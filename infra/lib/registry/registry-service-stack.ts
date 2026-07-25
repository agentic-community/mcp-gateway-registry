/**
 * RegistryServiceStack - Core ECS services + ALB + EFS + Secrets.
 *
 * Wraps the L3 constructs RegistryAlb, RegistryEfs, RegistrySecrets,
 * RegistryEcsService, McpServerService, and ObservabilityPipeline. The stack
 * itself only handles config plumbing, env-var mapping, and cross-stack SG
 * ingress.
 */

import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as servicediscovery from 'aws-cdk-lib/aws-servicediscovery';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';

import { RegistryConfig } from './registry-config';
import { RegistryNetworkStack } from './registry-network-stack';
import { RegistryDataStack } from './registry-data-stack';
import { RegistryAuthStack } from './registry-auth-stack';
import { RegistryEcsService } from './constructs/registry-ecs-service';
import { McpServerService } from './constructs/mcp-server-service';
import { ObservabilityPipeline } from './constructs/observability-pipeline';
import { RegistryAlb } from './constructs/registry-alb';
import { RegistryEfs } from './constructs/registry-efs';
import { RegistrySecrets } from './constructs/registry-secrets';
import { RegistryAlarms } from './constructs/registry-alarms';
import { CloudFrontOriginDistribution } from './constructs/cloudfront-distribution';
import { WafRules } from './constructs/waf-rules';
import * as route53 from 'aws-cdk-lib/aws-route53';
import * as route53targets from 'aws-cdk-lib/aws-route53-targets';

export interface RegistryServiceStackProps extends cdk.StackProps {
  readonly config: RegistryConfig;
  readonly networkStack: RegistryNetworkStack;
  readonly dataStack: RegistryDataStack;
  readonly authStack: RegistryAuthStack;
}

export class RegistryServiceStack extends cdk.Stack {
  public readonly ecsCluster: ecs.Cluster;
  public readonly registryEcsSg: ec2.SecurityGroup;
  public readonly authEcsSg: ec2.SecurityGroup;
  public readonly efsId: string;
  public readonly registryAlbDns: string;
  public readonly registryAlbArn: string;
  public readonly registryAlbSg: ec2.SecurityGroup;
  public readonly serviceDiscoveryNamespaceArn: string;
  public readonly appSecretsKmsKey: kms.Key;
  public readonly registryUrl: string;

  constructor(scope: Construct, id: string, props: RegistryServiceStackProps) {
    super(scope, id, props);

    const { config, networkStack, dataStack, authStack } = props;
    const { vpc, privateSubnets, publicSubnets } = networkStack;
    const namePrefix = config.name;

    const registryDomain = config.useRegionalDomains
      ? `${this.region}.${config.baseDomain}`
      : config.baseDomain;

    // Cloud Map service-discovery namespace
    const cloudMapNamespace = new servicediscovery.PrivateDnsNamespace(this, 'CloudMapNamespace', {
      name: `${namePrefix}.local`,
      description: 'Service discovery namespace for MCP Gateway Registry',
      vpc,
    });
    this.serviceDiscoveryNamespaceArn = cloudMapNamespace.namespaceArn;

    // ECS cluster
    this.ecsCluster = new ecs.Cluster(this, 'EcsCluster', {
      clusterName: `${namePrefix}-ecs-cluster`,
      vpc,
      containerInsights: true,
    });
    const cfnCluster = this.ecsCluster.node.defaultChild as ecs.CfnCluster;
    cfnCluster.capacityProviders = ['FARGATE'];
    cfnCluster.defaultCapacityProviderStrategy = [{ capacityProvider: 'FARGATE', base: 1, weight: 50 }];
    cfnCluster.addPropertyOverride('ServiceConnectDefaults', {
      Namespace: cloudMapNamespace.namespaceArn,
    });

    // ALB + listeners + target groups
    const alb = new RegistryAlb(this, 'Alb', { config, vpc, publicSubnets });
    this.registryAlbSg = alb.albSg;
    this.registryAlbDns = alb.alb.loadBalancerDnsName;
    this.registryAlbArn = alb.alb.loadBalancerArn;

    // CloudFront distribution fronting the registry ALB (Mode 1 & 3).
    let cfDistribution: CloudFrontOriginDistribution | undefined;
    let cfHostedZone: route53.IHostedZone | undefined;
    if (config.cloudfront.enabled) {
      cfHostedZone = config.enableRoute53Dns
        ? route53.HostedZone.fromLookup(this, 'HostedZone', {
            domainName: config.baseDomain,
          })
        : undefined;
      cfDistribution = new CloudFrontOriginDistribution(this, 'CloudFront', {
        config,
        albDns: alb.alb.loadBalancerDnsName,
        customDomain: config.enableRoute53Dns ? `registry.${config.baseDomain}` : '',
        hostedZone: cfHostedZone,
        comment: `${config.name} MCP Gateway Registry CloudFront Distribution`,
        logsPrefix: 'mcp-gateway/',
        emitCloudFrontForwardedProtoHeader: true,
      });

      // Optional WAFv2 Web ACL
      new WafRules(this, 'Waf', {
        config,
        mcpGatewayAlbArn: alb.alb.loadBalancerArn,
        keycloakAlbArn: undefined,
      });
    }

    // Route53 A-record for registry.<domain> — target ALB (Mode 2) or
    // CloudFront (Mode 3). Registry ALB alias-in-Mode-2 is created inside
    // RegistryAlb; here we only handle the Mode-3 case (target CloudFront).
    if (config.enableRoute53Dns && config.cloudfront.enabled && cfDistribution?.distribution && cfHostedZone) {
      new route53.ARecord(this, 'RegistryAliasRecord', {
        zone: cfHostedZone,
        recordName: `registry.${config.baseDomain}`,
        target: route53.RecordTarget.fromAlias(
          new route53targets.CloudFrontTarget(cfDistribution.distribution),
        ),
      });
    }

    if (cfDistribution) {
      this.registryUrl = cfDistribution.url;
    } else if (config.enableRoute53Dns || config.certificateArn !== '') {
      this.registryUrl = `https://${registryDomain}`;
    } else {
      this.registryUrl = `http://${alb.alb.loadBalancerDnsName}`;
    }

    // EFS + access points
    const efsResources = new RegistryEfs(this, 'Efs', { config, vpc, privateSubnets });
    this.efsId = efsResources.fileSystem.fileSystemId;
    const accessPoints = efsResources.accessPoints;

    // Application secrets bundle
    const secretsBundle = new RegistrySecrets(this, 'AppSecrets', {
      config,
      documentDbSecretArn: dataStack.documentDbSecretArn,
    });
    this.appSecretsKmsKey = secretsBundle.kmsKey;
    const secretsAccessStatements = secretsBundle.accessStatements;

    // Auth provider determination
    const authProvider = config.auth0.enabled ? 'auth0'
      : config.okta.enabled ? 'okta'
      : config.entra.enabled ? 'entra'
      : authStack.keycloakDomain !== '' ? 'keycloak'
      : 'default';

    // Container env shared by registry + auth-server (both need DocumentDB,
    // OAuth provider config, federation config, etc.)
    const sharedEnv: Record<string, string> = {
      REGISTRY_URL: this.registryUrl,
      AUTH_SERVER_URL: 'http://auth-server:8888',
      AUTH_SERVER_EXTERNAL_URL: this.registryUrl,
      AWS_REGION: config.awsRegion,
      AUTH_PROVIDER: authProvider,
      // Both registry AND auth-server need these *_ENABLED flags — auth-server
      // substitutes them into oauth2_providers.yml. Missing values leave the
      // yaml value as literal `${VAR}` which breaks provider registration
      // (Login page then shows "No login methods are currently configured").
      KEYCLOAK_ENABLED: authStack.keycloakDomain !== '' ? 'true' : 'false',
      COGNITO_ENABLED: 'false',
      GITHUB_ENABLED: 'false',
      GOOGLE_ENABLED: 'false',
      PINGFEDERATE_ENABLED: 'false',
      KEYCLOAK_URL: authStack.keycloakUrl,
      KEYCLOAK_REALM: 'mcp-gateway',
      KEYCLOAK_CLIENT_ID: 'mcp-gateway-web',
      ENTRA_ENABLED: String(config.entra.enabled),
      ENTRA_TENANT_ID: config.entra.tenantId,
      ENTRA_CLIENT_ID: config.entra.clientId,
      IDP_GROUP_FILTER_PREFIX: config.idpGroupFilterPrefix,
      OKTA_ENABLED: String(config.okta.enabled),
      OKTA_DOMAIN: config.okta.domain,
      OKTA_CLIENT_ID: config.okta.clientId,
      OKTA_M2M_CLIENT_ID: config.okta.m2mClientId,
      OKTA_AUTH_SERVER_ID: config.okta.authServerId,
      AUTH0_ENABLED: String(config.auth0.enabled),
      AUTH0_DOMAIN: config.auth0.domain,
      AUTH0_CLIENT_ID: config.auth0.clientId,
      AUTH0_AUDIENCE: config.auth0.audience,
      AUTH0_GROUPS_CLAIM: config.auth0.groupsClaim,
      AUTH0_M2M_CLIENT_ID: config.auth0.m2mClientId,
      AUTH0_MANAGEMENT_API_TOKEN: config.auth0.managementApiToken,
      SESSION_COOKIE_SECURE: String(config.session.cookieSecure),
      SESSION_COOKIE_DOMAIN: config.session.cookieDomain,
      // Exact-match allowlist of OAuth login/logout redirect URIs
      // (open-redirect hardening). Empty falls back to the weaker
      // cookie-domain heuristic. Read by the auth-server.
      OAUTH2_ALLOWED_REDIRECT_URIS: config.session.oauth2AllowedRedirectUris,
      OAUTH_STORE_TOKENS_IN_SESSION: String(config.session.oauthStoreTokensInSession),
      REGISTRY_STATIC_TOKEN_AUTH_ENABLED: String(config.staticTokenAuth.registryStaticTokenAuthEnabled),
      REGISTRY_API_TOKEN: config.staticTokenAuth.registryApiToken,
      M2M_DIRECT_REGISTRATION_ENABLED: String(config.staticTokenAuth.m2mDirectRegistrationEnabled),
      REGISTRY_ID: config.federation.registryId,
      FEDERATION_STATIC_TOKEN_AUTH_ENABLED: String(config.federation.staticTokenAuthEnabled),
      FEDERATION_STATIC_TOKEN: config.federation.staticToken,
      FEDERATION_ENCRYPTION_KEY: config.federation.encryptionKey,
      ANS_INTEGRATION_ENABLED: String(config.ans.integrationEnabled),
      ANS_API_ENDPOINT: config.ans.apiEndpoint,
      ANS_API_KEY: config.ans.apiKey,
      ANS_API_SECRET: config.ans.apiSecret,
      ANS_API_TIMEOUT_SECONDS: String(config.ans.apiTimeoutSeconds),
      ANS_SYNC_INTERVAL_HOURS: String(config.ans.syncIntervalHours),
      ANS_VERIFICATION_CACHE_TTL_SECONDS: String(config.ans.verificationCacheTtlSeconds),
      STORAGE_BACKEND: config.storageBackend,
      DOCUMENTDB_HOST: dataStack.documentDbCluster?.attrEndpoint ?? '',
      DOCUMENTDB_PORT: '27017',
      DOCUMENTDB_DATABASE: config.documentdb.database,
      DOCUMENTDB_NAMESPACE: config.documentdb.namespace,
      DOCUMENTDB_USE_TLS: String(config.documentdb.useTls),
      DOCUMENTDB_USE_IAM: String(config.documentdb.useIam),
      DOCUMENTDB_TLS_CA_FILE: '/app/certs/global-bundle.pem',
      AUDIT_LOG_ENABLED: String(config.audit.logEnabled),
      AUDIT_LOG_MONGODB_TTL_DAYS: String(config.audit.logTtlDays),
      METRICS_SERVICE_URL: config.enableObservability ? 'http://metrics-service:8890' : '',
    };

    const registryEnv: Record<string, string> = {
      ...sharedEnv,
      HOME: '/tmp',
      GATEWAY_ADDITIONAL_SERVER_NAMES: registryDomain,
      EC2_PUBLIC_DNS: registryDomain || alb.alb.loadBalancerDnsName,
      KEYCLOAK_ADMIN: 'admin',
      EMBEDDINGS_PROVIDER: config.embeddings.provider,
      EMBEDDINGS_MODEL_NAME: config.embeddings.modelName,
      EMBEDDINGS_MODEL_DIMENSIONS: String(config.embeddings.modelDimensions),
      EMBEDDINGS_AWS_REGION: config.embeddings.awsRegion,
      SECURITY_SCAN_ENABLED: String(config.security.scanEnabled),
      SECURITY_SCAN_ON_REGISTRATION: String(config.security.scanOnRegistration),
      SECURITY_BLOCK_UNSAFE_SERVERS: String(config.security.blockUnsafeServers),
      SECURITY_ANALYZERS: config.security.analyzers,
      SECURITY_SCAN_TIMEOUT: String(config.security.scanTimeout),
      SECURITY_ADD_PENDING_TAG: String(config.security.addPendingTag),
      REGISTRY_NAME: config.registryCard.name,
      REGISTRY_ORGANIZATION_NAME: config.registryCard.organizationName,
      REGISTRY_DESCRIPTION: config.registryCard.description,
      REGISTRY_CONTACT_EMAIL: config.registryCard.contactEmail,
      REGISTRY_CONTACT_URL: config.registryCard.contactUrl,
      AWS_REGISTRY_FEDERATION_ENABLED: String(config.federation.awsRegistryFederationEnabled),
      DEPLOYMENT_MODE: config.deploymentMode,
      REGISTRY_MODE: config.registryMode,
      SHOW_SERVERS_TAB: String(config.uiTabs.showServersTab),
      SHOW_VIRTUAL_SERVERS_TAB: String(config.uiTabs.showVirtualServersTab),
      SHOW_SKILLS_TAB: String(config.uiTabs.showSkillsTab),
      SHOW_AGENTS_TAB: String(config.uiTabs.showAgentsTab),
      MAX_TOKENS_PER_USER_PER_HOUR: String(config.staticTokenAuth.maxTokensPerUserPerHour),
      MCP_TELEMETRY_DISABLED: config.telemetry.disabled,
      MCP_TELEMETRY_OPT_OUT: config.telemetry.optOut,
      MCP_TELEMETRY_HEARTBEAT_INTERVAL_MINUTES: config.telemetry.heartbeatIntervalMinutes,
      TELEMETRY_DEBUG: config.telemetry.debug,
      DISABLE_AI_REGISTRY_TOOLS_SERVER: config.disableAiRegistryToolsServer,
      // PRM scopes_supported override. Live registry image advertises group-derived
      // internal scope names (registry-admins, federation-service, etc.) that
      // Keycloak DCR does not accept. Force the IdP-universal OIDC scopes so
      // Claude/MCP client DCR succeeds. Access is still group-derived at token
      // validation time, so this does not affect authorization.
      MCP_ADVERTISED_SCOPES: 'openid email profile offline_access',
      SERVICE_CONNECT_NAMESPACE: `${namePrefix}.local`,
      GITHUB_PAT: config.github.pat,
      GITHUB_APP_ID: config.github.appId,
      GITHUB_APP_INSTALLATION_ID: config.github.appInstallationId,
      GITHUB_APP_PRIVATE_KEY: config.github.appPrivateKey,
      GITHUB_EXTRA_HOSTS: config.github.extraHosts,
      GITHUB_API_BASE_URL: config.github.apiBaseUrl,
    };

    const authEnv: Record<string, string> = {
      ...sharedEnv,
      KEYCLOAK_EXTERNAL_URL: authStack.keycloakUrl,
      KEYCLOAK_M2M_CLIENT_ID: 'mcp-gateway-m2m',
    };

    // Container secrets (registry + auth share most of these)
    const docdbSecret = dataStack.documentDbSecretArn
      ? secretsmanager.Secret.fromSecretCompleteArn(this, 'DocDbSecretRef', dataStack.documentDbSecretArn)
      : undefined;

    const conditional: Array<[string, secretsmanager.ISecret | undefined]> = [
      ['ENTRA_CLIENT_SECRET', secretsBundle.entraClientSecret],
      ['OKTA_CLIENT_SECRET', secretsBundle.oktaClientSecret],
      ['OKTA_M2M_CLIENT_SECRET', secretsBundle.oktaM2mClientSecret],
      ['OKTA_API_TOKEN', secretsBundle.oktaApiToken],
      ['AUTH0_CLIENT_SECRET', secretsBundle.auth0ClientSecret],
      ['AUTH0_M2M_CLIENT_SECRET', secretsBundle.auth0M2mClientSecret],
      ['METRICS_API_KEY', secretsBundle.metricsApiKey],
    ];

    const sharedSecrets: Record<string, ecs.Secret> = {
      SECRET_KEY: ecs.Secret.fromSecretsManager(secretsBundle.secretKey),
      KEYCLOAK_CLIENT_SECRET: ecs.Secret.fromSecretsManager(secretsBundle.keycloakClientSecret, 'client_secret'),
      KEYCLOAK_M2M_CLIENT_SECRET: ecs.Secret.fromSecretsManager(secretsBundle.keycloakM2mClientSecret, 'client_secret'),
      ...(docdbSecret ? {
        DOCUMENTDB_USERNAME: ecs.Secret.fromSecretsManager(docdbSecret, 'username'),
        DOCUMENTDB_PASSWORD: ecs.Secret.fromSecretsManager(docdbSecret, 'password'),
      } : {}),
      ...Object.fromEntries(
        conditional
          .filter(([, s]) => s)
          .map(([k, s]) => [k, ecs.Secret.fromSecretsManager(s!)]),
      ),
    };

    const registrySecrets: Record<string, ecs.Secret> = {
      ...sharedSecrets,
      KEYCLOAK_ADMIN_PASSWORD: ecs.Secret.fromSecretsManager(secretsBundle.keycloakAdminPassword),
      EMBEDDINGS_API_KEY: ecs.Secret.fromSecretsManager(secretsBundle.embeddingsApiKey),
    };
    const authSecrets = sharedSecrets;

    // Optional Bedrock AgentCore policy for federation.
    //
    // Least-privilege: the registry federation client is READ-ONLY against the
    // bedrock-agentcore-control plane (list registries, list records, get record
    // -- see registry/services/federation/agentcore_client.py). It never
    // creates/updates/deletes AgentCore resources, so the action set is limited
    // to those three read operations.
    //
    // The read grant is split into two statements because the actions differ in
    // their IAM resource-level support (per the AWS Service Authorization
    // Reference):
    //   - ListRegistries has NO resource type, so IAM only accepts it on
    //     Resource "*". Scoping it to a registry ARN silently makes it a no-op
    //     (the action never matches) and boto3 gets AccessDenied at runtime.
    //   - ListRegistryRecords (resource type "registry") and GetRegistryRecord
    //     (resource type "registry-record") DO support resource-level
    //     permissions, so they are scoped to registries in the deploying
    //     account. Region is wildcarded so per-registry region overrides keep
    //     working; the record ARN (registry/<id>/record/<id>) is a child of the
    //     registry/* prefix.
    //
    // Cross-account federation assumes caller-supplied role ARNs. That grant is
    // only emitted when specific ARNs are configured; an empty list -> no
    // sts:AssumeRole statement (fail closed, no wildcard cross-account trust).
    const federationRoleArns = config.federation.awsRegistryFederationAssumeRoleArns ?? [];
    const agentCoreStatements: iam.PolicyStatement[] = [
      new iam.PolicyStatement({
        sid: 'BedrockAgentCoreListRegistries',
        effect: iam.Effect.ALLOW,
        actions: ['bedrock-agentcore:ListRegistries'],
        // ListRegistries has no IAM resource type; it must be granted on "*".
        // This is not a privilege-creep wildcard -- it is the only Resource
        // value AWS accepts for this single read/list action.
        resources: ['*'],
      }),
      new iam.PolicyStatement({
        sid: 'BedrockAgentCoreReadRecords',
        effect: iam.Effect.ALLOW,
        actions: [
          'bedrock-agentcore:ListRegistryRecords',
          'bedrock-agentcore:GetRegistryRecord',
        ],
        // Scope to registries (and their child records) in the deploying
        // account. registry/* also covers registry/<id>/record/<id>.
        resources: [`arn:${this.partition}:bedrock-agentcore:*:${this.account}:registry/*`],
      }),
    ];
    if (federationRoleArns.length > 0) {
      // Fail closed on a malformed ARN rather than silently synthesizing a
      // policy whose resource is rejected at deploy time (parity with the
      // Terraform variable's validation block).
      const roleArnPattern = /^arn:aws[a-z-]*:iam::[0-9]{12}:role\/.+$/;
      const invalidArns = federationRoleArns.filter((arn) => !roleArnPattern.test(arn));
      if (invalidArns.length > 0) {
        throw new Error(
          `federation.awsRegistryFederationAssumeRoleArns contains invalid IAM role ARNs: ${invalidArns.join(', ')}. ` +
            'Each entry must match arn:aws:iam::<account-id>:role/<name>.',
        );
      }
      agentCoreStatements.push(
        new iam.PolicyStatement({
          sid: 'StsAssumeRoleForCrossAccount',
          effect: iam.Effect.ALLOW,
          actions: ['sts:AssumeRole'],
          // Only the explicitly configured cross-account federation roles.
          resources: federationRoleArns,
          // Defense-in-depth: the target role must also carry the federation tag.
          conditions: { StringLike: { 'iam:ResourceTag/Purpose': 'agentcore-federation' } },
        }),
      );
    }
    const registryTaskRolePolicies: iam.IManagedPolicy[] = config.federation.awsRegistryFederationEnabled
      ? [new iam.ManagedPolicy(this, 'BedrockAgentCorePolicy', {
          statements: agentCoreStatements,
        })]
      : [];

    // Registry ECS service — nginx (:8080) fronts everything external. Gradio
    // (:7860) is loopback-bound and reached through nginx path routes.
    const registryService = new RegistryEcsService(this, 'RegistrySvc', {
      serviceName: 'registry',
      image: config.images.registry,
      cpu: 1024,
      memory: 2048,
      containerPort: 8080,
      additionalPorts: [
        { port: 8443, name: 'https' },
        { port: 7860, name: 'gradio-internal' },
      ],
      vpc,
      subnets: privateSubnets,
      cluster: this.ecsCluster,
      serviceConnectNamespaceArn: cloudMapNamespace.namespaceArn,
      serviceConnect: { port: 8080, dnsName: 'registry', portName: 'http', discoveryName: 'registry' },
      environment: registryEnv,
      secrets: registrySecrets,
      targetGroups: [
        { targetGroup: alb.registryTg, containerPort: 8080 },
      ],
      additionalTaskRolePolicies: registryTaskRolePolicies,
      additionalExecRoleStatements: secretsAccessStatements,
      healthCheckCommand: 'curl -f http://localhost:8080/health || exit 1',
      namePrefix,
      desiredCount: config.replicas.registry,
    });
    this.registryEcsSg = registryService.securityGroup;

    // Only :8080 (nginx) accepts traffic from the ALB. :8443/:7860 remain
    // inside the task for nginx-internal reverse-proxy to the app process.
    registryService.securityGroup.addIngressRule(
      alb.albSg, ec2.Port.tcp(8080), 'Registry nginx from ALB',
    );

    // Auth ECS service
    const authService = new RegistryEcsService(this, 'AuthSvc', {
      serviceName: 'auth-server',
      image: config.images.authServer,
      cpu: 512,
      memory: 1024,
      containerPort: 8888,
      vpc,
      subnets: privateSubnets,
      cluster: this.ecsCluster,
      serviceConnectNamespaceArn: cloudMapNamespace.namespaceArn,
      serviceConnect: { port: 8888, dnsName: 'auth-server', portName: 'auth-server', discoveryName: 'auth-server' },
      environment: authEnv,
      secrets: authSecrets,
      efsVolumes: [
        {
          volumeName: 'mcp-logs',
          fileSystemId: efsResources.fileSystem.fileSystemId,
          accessPointId: accessPoints['logs'].accessPointId,
          containerPath: '/app/logs',
        },
        {
          volumeName: 'auth-config',
          fileSystemId: efsResources.fileSystem.fileSystemId,
          accessPointId: accessPoints['authConfig'].accessPointId,
          containerPath: '/efs/auth_config',
        },
      ],
      // No public ALB attachment — auth-server is only reachable via Service
      // Connect from the registry container (nginx proxies /oauth2/*).
      targetGroups: [],
      additionalExecRoleStatements: secretsAccessStatements,
      healthCheckCommand: 'curl -f http://localhost:8888/health || exit 1',
      namePrefix,
      desiredCount: config.replicas.auth,
    });
    this.authEcsSg = authService.securityGroup;

    authService.securityGroup.addIngressRule(registryService.securityGroup, ec2.Port.tcp(8888), 'Allow registry to access auth server');

    // Registry nginx hard-fails if auth-server Service Connect DNS is not yet
    // registered on first deploy. Force CFN to create auth-server first.
    registryService.service.node.addDependency(authService.service);

    // Optional MCP servers / A2A agents
    new McpServerService(this, 'CurrenttimeSvc', {
      serviceName: 'currenttime-server',
      imageUri: config.images.currenttime,
      containerPort: 8000,
      vpc, subnets: privateSubnets, cluster: this.ecsCluster,
      serviceConnectNamespaceArn: cloudMapNamespace.namespaceArn,
      serviceConnectDnsName: 'currenttime-server',
      serviceConnectPortName: 'currenttime',
      environment: { PORT: '8000', MCP_TRANSPORT: 'streamable-http' },
      ingressSecurityGroup: registryService.securityGroup,
      namePrefix,
      desiredCount: config.replicas.currenttime,
    });

    const mcpgwService = new McpServerService(this, 'McpgwSvc', {
      serviceName: 'mcpgw-server',
      imageUri: config.images.mcpgw,
      containerPort: 8003,
      vpc, subnets: privateSubnets, cluster: this.ecsCluster,
      serviceConnectNamespaceArn: cloudMapNamespace.namespaceArn,
      serviceConnectDnsName: 'mcpgw-server',
      serviceConnectPortName: 'mcpgw',
      environment: {
        PORT: '8003',
        HOST: '0.0.0.0',
        REGISTRY_BASE_URL: 'http://registry:8080',
        REGISTRY_USERNAME: 'admin',
      },
      efsVolumes: [{
        volumeName: 'mcpgw-data',
        fileSystemId: efsResources.fileSystem.fileSystemId,
        accessPointId: accessPoints['mcpgwData'].accessPointId,
        containerPath: '/app/data',
      }],
      ingressSecurityGroup: registryService.securityGroup,
      additionalExecRoleStatements: secretsAccessStatements,
      namePrefix,
      desiredCount: config.replicas.mcpgw,
    });

    if (mcpgwService.securityGroup) {
      for (const port of [8080, 7860]) {
        registryService.securityGroup.addIngressRule(
          mcpgwService.securityGroup, ec2.Port.tcp(port), `Port ${port} from mcpgw`,
        );
      }
      // Auth-server mcp-proxy forwards to mcpgw:8003 (parity with terraform
      // auth_to_mcpgw rule at terraform/aws-ecs/modules/mcp-gateway/ecs-services.tf:1965)
      mcpgwService.securityGroup.addIngressRule(
        authService.securityGroup, ec2.Port.tcp(8003), 'Allow auth-server mcp-proxy to reach mcpgw',
      );
    }

    new McpServerService(this, 'RealServerFakeToolsSvc', {
      serviceName: 'realserverfaketools-server',
      imageUri: config.images.realserverfaketools,
      containerPort: 8002,
      vpc, subnets: privateSubnets, cluster: this.ecsCluster,
      serviceConnectNamespaceArn: cloudMapNamespace.namespaceArn,
      serviceConnectDnsName: 'realserverfaketools-server',
      serviceConnectPortName: 'realserverfaketools',
      environment: { PORT: '8002', MCP_TRANSPORT: 'streamable-http' },
      ingressSecurityGroup: registryService.securityGroup,
      namePrefix,
      desiredCount: config.replicas.realserverfaketools,
    });

    for (const agent of [
      { id: 'FlightBookingSvc', name: 'flight-booking-agent', image: config.images.flightBookingAgent, dnsName: 'flight-booking-agent', portName: 'flight-booking', count: config.replicas.flightBookingAgent },
      { id: 'TravelAssistantSvc', name: 'travel-assistant-agent', image: config.images.travelAssistantAgent, dnsName: 'travel-assistant-agent', portName: 'travel-assistant', count: config.replicas.travelAssistantAgent },
    ]) {
      new McpServerService(this, agent.id, {
        serviceName: agent.name,
        imageUri: agent.image,
        containerPort: 9000,
        vpc, subnets: privateSubnets, cluster: this.ecsCluster,
        serviceConnectNamespaceArn: cloudMapNamespace.namespaceArn,
        serviceConnectDnsName: agent.dnsName,
        serviceConnectPortName: agent.portName,
        environment: { AWS_REGION: config.awsRegion, AWS_DEFAULT_REGION: config.awsRegion },
        ingressCidr: config.vpcCidr,
        healthCheckCommand: 'curl -f http://localhost:9000/ping || exit 1',
        namePrefix,
        desiredCount: agent.count,
      });
    }

    // Cross-stack SG ingress (uses CfnSecurityGroupIngress to break the cycle:
    // Service depends on Data/Auth, so their SG objects can't reference Service SG)
    const crossStackIngress: Array<[string, ec2.ISecurityGroup, ec2.ISecurityGroup, number, string]> = [
      ['DocDbFromRegistry', dataStack.documentDbSg, registryService.securityGroup, 27017, 'DocumentDB ingress from registry'],
      ['DocDbFromAuth', dataStack.documentDbSg, authService.securityGroup, 27017, 'DocumentDB ingress from auth-server'],
      ['KeycloakAlbFromRegistry', authStack.keycloakAlbSg, registryService.securityGroup, 443, 'Keycloak ALB HTTPS from registry'],
      ['KeycloakAlbFromAuthSvc', authStack.keycloakAlbSg, authService.securityGroup, 443, 'Keycloak ALB HTTPS from auth-server'],
      ['KeycloakAlbHttpFromRegistry', authStack.keycloakAlbSg, registryService.securityGroup, 80, 'Keycloak ALB HTTP from registry'],
      ['KeycloakAlbHttpFromAuthSvc', authStack.keycloakAlbSg, authService.securityGroup, 80, 'Keycloak ALB HTTP from auth-server'],
    ];
    for (const [logicalId, target, source, port, description] of crossStackIngress) {
      new ec2.CfnSecurityGroupIngress(this, logicalId, {
        groupId: target.securityGroupId,
        ipProtocol: 'tcp',
        fromPort: port,
        toPort: port,
        sourceSecurityGroupId: source.securityGroupId,
        description,
      });
    }

    // CloudWatch alarms (no-op when monitoring.enabled=false)
    new RegistryAlarms(this, 'Alarms', {
      config,
      clusterName: this.ecsCluster.clusterName,
      registryServiceName: registryService.service.serviceName,
      authServiceName: authService.service.serviceName,
      alb: alb.alb,
      registryTargetGroup: alb.registryTg,
      documentDbClusterId: dataStack.documentDbCluster.ref,
    });

    // Observability (AMP + Grafana + ADOT) — no-op when disabled
    new ObservabilityPipeline(this, 'Observability', {
      config,
      vpc,
      privateSubnets,
      ecsCluster: this.ecsCluster,
      serviceConnectNamespaceArn: cloudMapNamespace.namespaceArn,
      alb: alb.alb,
      httpListener: alb.httpListener,
      httpsListener: alb.httpsListener,
      appSecretsKmsKey: this.appSecretsKmsKey,
      metricsApiKeySecret: secretsBundle.metricsApiKey,
      metricsKeyPepperSecret: secretsBundle.metricsKeyPepper,
      otlpExporterHeadersSecret: secretsBundle.otlpExporterHeaders,
      grafanaAdminPasswordSecret: secretsBundle.grafanaAdminPassword,
      secretsAccessStatements,
      registryServiceSg: registryService.securityGroup,
      authServiceSg: authService.securityGroup,
      albSg: alb.albSg,
      namePrefix,
    });

    // Tags
    cdk.Tags.of(this).add('Project', 'mcp-gateway-registry');
    cdk.Tags.of(this).add('Component', 'service');
    cdk.Tags.of(this).add('Environment', 'production');
    cdk.Tags.of(this).add('ManagedBy', 'cdk');

    // Outputs. Auth-server and Gradio are NOT exposed on the ALB — external
    // callers reach both via nginx path routes on the registry container.
    // OAuth callbacks go to ${REGISTRY_URL}/oauth2/callback/keycloak; the
    // Gradio UI is proxied at the registry root.
    new cdk.CfnOutput(this, 'RegistryUrl', { value: this.registryUrl, description: 'MCP Gateway Registry URL' });
    new cdk.CfnOutput(this, 'RegistryAlbDnsName', { value: this.registryAlbDns, description: 'Registry ALB DNS name' });
    new cdk.CfnOutput(this, 'KeycloakUrl', { value: authStack.keycloakUrl, description: 'Keycloak identity provider URL' });
    new cdk.CfnOutput(this, 'GradioUiUrl', {
      value: this.registryUrl,
      description: 'Gradio UI URL (proxied by registry nginx at the root path)',
    });
    if (config.enableObservability) {
      new cdk.CfnOutput(this, 'GrafanaUrl', { value: `${this.registryUrl}/grafana`, description: 'Grafana dashboard URL' });
    }
    new cdk.CfnOutput(this, 'ServiceEndpoints', {
      value: JSON.stringify({
        registry: this.registryUrl,
        registryApi: `${this.registryUrl}/api/v1`,
        registryHealth: `${this.registryUrl}/health`,
        keycloak: authStack.keycloakUrl,
        authServer: `${this.registryUrl}/oauth2`,
        gradioUi: this.registryUrl,
      }),
      description: 'All service endpoints as JSON',
    });
  }
}
