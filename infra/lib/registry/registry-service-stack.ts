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
import * as path from 'path';
import { RegistryEcsService } from './constructs/registry-ecs-service';
import { McpServerService } from './constructs/mcp-server-service';
import { ObservabilityPipeline } from './constructs/observability-pipeline';
import { RegistryAlb } from './constructs/registry-alb';
import { RegistryEfs } from './constructs/registry-efs';
import { RegistrySecrets } from './constructs/registry-secrets';
import { ScopesLoader } from './constructs/scopes-loader';
import { RegistryAlarms } from './constructs/registry-alarms';

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

    this.registryUrl =
      config.enableRoute53Dns || config.certificateArn !== ''
        ? `https://${registryDomain}`
        : `http://${alb.alb.loadBalancerDnsName}`;

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
      KEYCLOAK_ENABLED: authStack.keycloakDomain !== '' ? 'true' : 'false',
      KEYCLOAK_ADMIN: 'admin',
      SCOPES_CONFIG_PATH: '/app/auth_server/scopes.yml',
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
      SCOPES_CONFIG_PATH: '/efs/auth_config/scopes.yml',
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

    // Optional Bedrock AgentCore policy for federation
    const registryTaskRolePolicies: iam.IManagedPolicy[] = config.federation.awsRegistryFederationEnabled
      ? [new iam.ManagedPolicy(this, 'BedrockAgentCorePolicy', {
          statements: [
            new iam.PolicyStatement({
              sid: 'BedrockAgentCoreFullAccess',
              effect: iam.Effect.ALLOW,
              actions: ['bedrock-agentcore:*'],
              resources: ['*'],
            }),
            new iam.PolicyStatement({
              sid: 'StsAssumeRoleForCrossAccount',
              effect: iam.Effect.ALLOW,
              actions: ['sts:AssumeRole'],
              resources: ['*'],
              conditions: { StringLike: { 'iam:ResourceTag/Purpose': 'agentcore-federation' } },
            }),
          ],
        })]
      : [];

    // Registry ECS service
    const registryService = new RegistryEcsService(this, 'RegistrySvc', {
      serviceName: 'registry',
      image: config.images.registry,
      cpu: 1024,
      memory: 2048,
      containerPort: 8080,
      additionalPorts: [
        { port: 8443, name: 'https' },
        { port: 7860, name: 'registry' },
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
        { targetGroup: alb.gradioTg, containerPort: 7860 },
      ],
      additionalTaskRolePolicies: registryTaskRolePolicies,
      additionalExecRoleStatements: secretsAccessStatements,
      healthCheckCommand: 'curl -f http://localhost:7860/health || exit 1',
      namePrefix,
      desiredCount: config.replicas.registry,
    });
    this.registryEcsSg = registryService.securityGroup;

    for (const port of [8080, 8443, 7860]) {
      registryService.securityGroup.addIngressRule(
        alb.albSg, ec2.Port.tcp(port), `Port ${port} from ALB`,
      );
    }

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
      targetGroups: [{ targetGroup: alb.authTg, containerPort: 8888 }],
      additionalExecRoleStatements: secretsAccessStatements,
      healthCheckCommand: 'curl -f http://localhost:8888/health || exit 1',
      namePrefix,
      desiredCount: config.replicas.auth,
    });
    this.authEcsSg = authService.securityGroup;

    authService.securityGroup.addIngressRule(alb.albSg, ec2.Port.tcp(8888), 'Auth server port from ALB');
    authService.securityGroup.addIngressRule(registryService.securityGroup, ec2.Port.tcp(8888), 'Allow registry to access auth server');

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

    // UI-scope group docs into DocumentDB. Bridges the upstream-image gap:
    // init-documentdb-indexes.py only seeds `registry-admins`; this seeds the
    // rest defined in scopes.yml (mcp-registry-admin, etc.).
    if (config.storageBackend === 'documentdb' && dataStack.documentDbSecretArn) {
      new ScopesLoader(this, 'ScopesLoader', {
        vpc,
        privateSubnets,
        ingressSg: registryService.securityGroup,
        documentDbHost: dataStack.documentDbCluster.attrEndpoint,
        documentDbPort: 27017,
        documentDbDatabase: config.documentdb.database,
        documentDbNamespace: config.documentdb.namespace,
        documentDbSecretArn: dataStack.documentDbSecretArn,
        documentDbSecretKmsKeyArn: dataStack.documentDbKmsKey.keyArn,
        scopesYmlPath: path.join(__dirname, '..', '..', '..', 'auth_server', 'scopes.yml'),
        authConfigAccessPoint: accessPoints['authConfig'],
        namePrefix,
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

    // Outputs
    new cdk.CfnOutput(this, 'RegistryUrl', { value: this.registryUrl, description: 'MCP Gateway Registry URL' });
    new cdk.CfnOutput(this, 'RegistryAlbDnsName', { value: this.registryAlbDns, description: 'Registry ALB DNS name' });
    new cdk.CfnOutput(this, 'KeycloakUrl', { value: authStack.keycloakUrl, description: 'Keycloak identity provider URL' });
    new cdk.CfnOutput(this, 'GradioUiUrl', {
      value: `${this.registryUrl.replace(/:\d+$/, '')}:7860`,
      description: 'Gradio UI URL (port 7860)',
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
        authServer: `${this.registryUrl}:8888`,
        gradioUi: `${this.registryUrl.replace(/:\d+$/, '')}:7860`,
      }),
      description: 'All service endpoints as JSON',
    });
  }
}
