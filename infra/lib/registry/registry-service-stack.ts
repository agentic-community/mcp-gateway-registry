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
import {
  ADOT_SIDECAR_CPU,
  ADOT_SIDECAR_MEMORY_MIB,
  ObservabilityPipeline,
} from './constructs/observability-pipeline';
import { RegistryAlb } from './constructs/registry-alb';
import { RegistryEfs } from './constructs/registry-efs';
import { RegistrySecrets } from './constructs/registry-secrets';
import { RegistryAlarms } from './constructs/registry-alarms';
import { CloudFrontOriginDistribution } from './constructs/cloudfront-distribution';
import { WafRules } from './constructs/waf-rules';
import * as route53 from 'aws-cdk-lib/aws-route53';
import * as route53targets from 'aws-cdk-lib/aws-route53-targets';

/**
 * Port the auth-server application binds inside its task.
 *
 * Deliberately not 8888. Service Connect injects a proxy into every task in
 * the namespace and binds each client alias on loopback, so an app listening
 * on 8888 inside the auth-server's own task would collide with the proxy's
 * `auth-server:8888` alias. Callers still use 8888: the alias forwards to this
 * port. Parity with terraform/aws-ecs (AUTH_LISTEN_PORT=18888).
 */
const AUTH_SERVER_CONTAINER_PORT: number = 18888;

/** Service Connect alias port callers use to reach the auth-server. */
const AUTH_SERVER_ALIAS_PORT: number = 8888;

/**
 * Registry nginx port serving the internal egress-token vend. Never fronted by
 * the ALB; reachable task-to-task as `registry:8091` and gated by a security
 * group rule that admits only the auth-server.
 */
const REGISTRY_EGRESS_INTERNAL_PORT: number = 8091;

/**
 * Secret-path segment owning the auth-server's OBO exchange cache.
 *
 * The secret store encodes an auth_method name as unpadded base64url of its
 * NFC form, and the cache reserves the name "obo-cache", so entries live under
 * `<prefix>/b2JvLWNhY2hl/`. Hard-coded rather than computed so the grant and
 * the comment cannot drift apart; parity with
 * terraform/aws-ecs/modules/mcp-gateway/iam.tf.
 */
const OBO_CACHE_NAMESPACE_SEGMENT: string = 'b2JvLWNhY2hl';

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

    // Auth provider determination. Order matters when more than one provider
    // is enabled, and it is PingFederate first, matching
    // terraform/aws-ecs/modules/mcp-gateway/ecs-services.tf so the same
    // config picks the same provider on either stack.
    const authProvider = config.pingfederate.enabled ? 'pingfederate'
      : config.auth0.enabled ? 'auth0'
      : config.okta.enabled ? 'okta'
      : config.entra.enabled ? 'entra'
      : config.cognito.enabled ? 'cognito'
      : authStack.keycloakDomain !== '' ? 'keycloak'
      : 'default';

    // OTEL settings shared by every instrumented app container. The endpoint
    // is localhost because each task carries its own ADOT collector sidecar;
    // the sidecar, not the app, holds the AMP endpoint and signs the write.
    // Traces are off (`none`) in parity with Terraform: only metrics flow.
    const otelEnv = (serviceName: string): Record<string, string> =>
      config.enableObservability
        ? {
            OTEL_SERVICE_NAME: serviceName,
            OTEL_METRIC_EXPORT_INTERVAL_MS: '15000',
            OTEL_EXPORTER_OTLP_ENDPOINT: 'http://localhost:4317',
            OTEL_EXPORTER_OTLP_PROTOCOL: 'grpc',
            OTEL_TRACES_EXPORTER: 'none',
          }
        : {};

    // Egress settings both services read. The registry owns the vault and the
    // consent flow; the auth-server performs the OBO exchange and calls the
    // registry's internal vend. Keep the pool and OBO-cache numbers identical
    // across both, since they describe the same conversation.
    const egressSharedEnv: Record<string, string> = {
      EGRESS_AUTH_ENABLED: String(config.egress.enabled),
      SECRET_STORE_BACKEND: config.egress.secretStoreBackend,
      SECRETS_MANAGER_PATH_PREFIX: config.egress.secretsManagerPathPrefix,
      // Without this the application creates vault secrets under the
      // AWS-managed key even when an operator configured a CMK, so the
      // configured key would be granted in IAM and then never used.
      SECRETS_MANAGER_KMS_KEY_ID: config.egress.secretsManagerKmsKeyArn,
      EGRESS_OAUTH_TRUSTED_IDP_HOSTS: config.egress.oauthTrustedIdpHosts,
      EGRESS_HTTP_POOL_MAX_CONNECTIONS: String(config.egress.httpPoolMaxConnections),
      EGRESS_HTTP_POOL_MAX_KEEPALIVE: String(config.egress.httpPoolMaxKeepalive),
      EGRESS_HTTP_POOL_KEEPALIVE_EXPIRY_SECONDS: String(config.egress.httpPoolKeepaliveExpirySeconds),
      EGRESS_HTTP_POOL_CONNECT_RETRIES: String(config.egress.httpPoolConnectRetries),
      EGRESS_FORWARD_PROXY_ENABLED: String(config.egress.forwardProxyEnabled),
      EGRESS_FORWARD_PROXY_CA_BUNDLE: config.egress.forwardProxyCaBundle,
      EGRESS_OBO_CACHE_ENABLED: String(config.egress.oboCacheEnabled),
      EGRESS_OBO_CACHE_MAX_TTL_SECONDS: String(config.egress.oboCacheMaxTtlSeconds),
      EGRESS_OBO_CACHE_EXPIRY_SKEW_SECONDS: String(config.egress.oboCacheExpirySkewSeconds),
    };

    // Container env shared by registry + auth-server (both need DocumentDB,
    // OAuth provider config, federation config, etc.)
    const sharedEnv: Record<string, string> = {
      REGISTRY_URL: this.registryUrl,
      // The Service Connect alias, not the container port: the proxy forwards
      // this to AUTH_SERVER_CONTAINER_PORT inside the auth-server's task.
      AUTH_SERVER_URL: `http://auth-server:${AUTH_SERVER_ALIAS_PORT}`,
      AUTH_SERVER_EXTERNAL_URL: this.registryUrl,
      AWS_REGION: config.awsRegion,
      AUTH_PROVIDER: authProvider,
      // Both registry AND auth-server need these *_ENABLED flags — auth-server
      // substitutes them into oauth2_providers.yml. Missing values leave the
      // yaml value as literal `${VAR}` which breaks provider registration
      // (Login page then shows "No login methods are currently configured").
      KEYCLOAK_ENABLED: authStack.keycloakDomain !== '' ? 'true' : 'false',
      GITHUB_ENABLED: 'false',
      GOOGLE_ENABLED: 'false',
      // Amazon Cognito. The pool, client and issuer come from stack config
      // only: a request header must never select how a credential is
      // validated, which is why there is no header-derived fallback here.
      COGNITO_ENABLED: String(config.cognito.enabled),
      COGNITO_USER_POOL_ID: config.cognito.userPoolId,
      COGNITO_CLIENT_ID: config.cognito.clientId,
      COGNITO_DOMAIN: config.cognito.domain,
      PINGFEDERATE_ENABLED: String(config.pingfederate.enabled),
      PINGFEDERATE_BASE_URL: config.pingfederate.baseUrl,
      PINGFEDERATE_EXTERNAL_URL: config.pingfederate.externalUrl,
      PINGFEDERATE_CLIENT_ID: config.pingfederate.clientId,
      PINGFEDERATE_M2M_CLIENT_ID: config.pingfederate.m2mClientId,
      PINGFEDERATE_APPLICATION_ID_URI: config.pingfederate.applicationIdUri,
      PINGFEDERATE_GROUPS_CLAIM: config.pingfederate.groupsClaim,
      PINGFEDERATE_M2M_ALLOWED_AUDIENCES: config.pingfederate.m2mAllowedAudiences,
      KEYCLOAK_URL: authStack.keycloakUrl,
      KEYCLOAK_REALM: 'mcp-gateway',
      KEYCLOAK_CLIENT_ID: 'mcp-gateway-web',
      ENTRA_ENABLED: String(config.entra.enabled),
      ENTRA_TENANT_ID: config.entra.tenantId,
      ENTRA_CLIENT_ID: config.entra.clientId,
      // Optional Entra PRM scope-format config (issue #990). Only emit when set
      // so the provider default (v2 / api://<client-id>) applies otherwise.
      ...(config.entra.scopeFormat ? { ENTRA_SCOPE_FORMAT: config.entra.scopeFormat } : {}),
      ...(config.entra.applicationIdUri
        ? { ENTRA_APPLICATION_ID_URI: config.entra.applicationIdUri }
        : {}),
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
      SESSION_COOKIE_SECURE: String(config.session.cookieSecure),
      SESSION_COOKIE_DOMAIN: config.session.cookieDomain,
      SESSION_COOKIE_HOST_PREFIX: String(config.session.cookieHostPrefix),
      // Session lifetime in seconds. Must match on the registry and the
      // auth-server; sharedEnv reaches both containers.
      SESSION_MAX_AGE_SECONDS: String(config.session.maxAgeSeconds),
      // Exact-match allowlist of OAuth login/logout redirect URIs
      // (open-redirect hardening). Empty falls back to the weaker
      // cookie-domain heuristic. Read by the auth-server.
      OAUTH2_ALLOWED_REDIRECT_URIS: config.session.oauth2AllowedRedirectUris,
      OAUTH_STORE_TOKENS_IN_SESSION: String(config.session.oauthStoreTokensInSession),
      REGISTRY_STATIC_TOKEN_AUTH_ENABLED: String(config.staticTokenAuth.registryStaticTokenAuthEnabled),
      M2M_DIRECT_REGISTRATION_ENABLED: String(config.staticTokenAuth.m2mDirectRegistrationEnabled),
      REGISTRY_ID: config.federation.registryId,
      FEDERATION_STATIC_TOKEN_AUTH_ENABLED: String(config.federation.staticTokenAuthEnabled),
      ANS_INTEGRATION_ENABLED: String(config.ans.integrationEnabled),
      ANS_API_ENDPOINT: config.ans.apiEndpoint,
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
      ...egressSharedEnv,
      ...otelEnv('mcp-gateway-registry'),
      HOME: '/tmp',
      // nginx sits behind the ALB on every ECS deployment, so tell its realip
      // module to trust that hop. Without this, audit records and the per-IP
      // rate-limit zones key on the ALB's private address: every client
      // collapses into one bucket and the audited IP is never the real one.
      // Empty config value resolves to this stack's VPC CIDR, matching the
      // Terraform default.
      TRUSTED_REAL_IP_CIDRS: config.trustedRealIpCidrs || config.vpcCidr,
      // Registry-only egress settings: it owns the consent flow and the vault.
      EGRESS_OAUTH_CALLBACK_BASE_URL: config.egress.oauthCallbackBaseUrl,
      EGRESS_TOKEN_REFRESH_SKEW_SECONDS: String(config.egress.tokenRefreshSkewSeconds),
      EGRESS_STATE_TTL_SECONDS: String(config.egress.stateTtlSeconds),
      // Registration-time allowlist of the target audiences an obo_exchange
      // server may declare (registry/core/schemas.py). It decides which
      // audiences the gateway will ever exchange into. It is not the replay
      // check on exchanged tokens, which uses accepted_audiences().
      EGRESS_OBO_ALLOWED_AUDIENCES: config.egress.oboAllowedAudiences,
      // PingFederate admin API, registry only: it creates OAuth clients and
      // PCV users there. The auth-server never calls this API.
      ...(config.pingfederate.enabled
        ? {
            PF_ADMIN_URL: config.pingfederate.adminUrl,
            PF_ADMIN_USER: config.pingfederate.adminUser,
          }
        : {}),
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
      HEALTH_CHECK_INTERVAL_SECONDS: String(config.healthChecks.intervalSeconds),
      TOOL_REFRESH_ENABLED: String(config.healthChecks.toolRefreshEnabled),
      TOOL_REFRESH_EVERY_CYCLES: String(config.healthChecks.toolRefreshEveryCycles),
      TOOL_FETCH_MAX_CONCURRENCY: String(config.healthChecks.toolFetchMaxConcurrency),
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
      GITHUB_APP_ID: config.github.appId,
      GITHUB_APP_INSTALLATION_ID: config.github.appInstallationId,
      GITHUB_EXTRA_HOSTS: config.github.extraHosts,
      GITHUB_API_BASE_URL: config.github.apiBaseUrl,
      // Prevent AWS SDKs from falling back to EC2 IMDS. ECS delivers task-role
      // credentials through the container-credentials provider (169.254.170.2)
      // on both the Fargate and EC2 launch types; only the IMDS provider
      // (169.254.169.254) is disabled, so task-role creds survive regardless of
      // launch type. That is why there is no operator opt-out here (unlike the
      // Helm charts, where an EKS node-instance-profile-only setup can need
      // IMDS): on ECS the credential-only-via-IMDS footgun cannot occur. The
      // registry uses boto3 (Secrets Manager, DocumentDB IAM, Cognito,
      // AgentCore).
      //
      // NOT redundant with the url_guard egress SSRF check: boto3 uses its own
      // urllib3 stack and never passes through GuardedTransport. url_guard
      // blocks the app being tricked into fetching a URL that redirects to
      // IMDS; this env var closes boto3's own IMDS credential provider, a
      // vector url_guard structurally cannot see. Kept in parity with
      // terraform/aws-ecs registry env.
      AWS_EC2_METADATA_DISABLED: 'true',
    };

    const authEnv: Record<string, string> = {
      ...sharedEnv,
      ...egressSharedEnv,
      ...otelEnv('mcp-auth-server'),
      KEYCLOAK_EXTERNAL_URL: authStack.keycloakUrl,
      KEYCLOAK_M2M_CLIENT_ID: 'mcp-gateway-m2m',
      // Allowlist of Cognito M2M app client ids whose client-credentials
      // tokens the auth-server accepts. Auth-server only; the registry has no
      // use for it.
      COGNITO_M2M_CLIENT_IDS: config.cognito.m2mClientIds,
      // The auth-server reaches the registry's egress-token vend on the
      // dedicated internal listener (registry:8091), which the ALB never
      // fronts and only the auth-server security group can reach.
      EGRESS_REGISTRY_INTERNAL_URL: config.egress.registryInternalUrl,
      // The Service Connect Envoy proxy binds 127.0.0.1:8888 inside the task,
      // so the application listens on a different port and nginx/Service
      // Connect routes 8888 to it.
      AUTH_LISTEN_PORT: String(AUTH_SERVER_CONTAINER_PORT),
      // Prometheus pull endpoint for the auth-server, in parity with the
      // Terraform auth-server env. The ADOT sidecar does the AMP write.
      ...(config.enableObservability
        ? {
            OTEL_EXPORTER_PROMETHEUS_HOST: '0.0.0.0',
            OTEL_EXPORTER_PROMETHEUS_PORT: '9464',
          }
        : {}),
      // Prevent AWS SDKs from falling back to EC2 IMDS. ECS task-role
      // credentials remain available through the container credential URI.
      // Kept in parity with terraform/aws-ecs auth-server env.
      AWS_EC2_METADATA_DISABLED: 'true',
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
      ['COGNITO_CLIENT_SECRET', secretsBundle.cognitoClientSecret],
      ['PINGFEDERATE_CLIENT_SECRET', secretsBundle.pingfederateClientSecret],
      ['PINGFEDERATE_M2M_CLIENT_SECRET', secretsBundle.pingfederateM2mClientSecret],
      ['METRICS_API_KEY', secretsBundle.metricsApiKey],
    ];

    const sharedSecrets: Record<string, ecs.Secret> = {
      SECRET_KEY: ecs.Secret.fromSecretsManager(secretsBundle.secretKey),
      // The registry and the auth-server MUST read the same value: the
      // auth-server mints an internal token only when the marker nginx sent
      // matches its own. A mismatch, or a missing value, 401s every registry
      // API and UI request, and both services refuse to start without it
      // (registry/core/config.py). Injecting it here, in sharedSecrets, is
      // what guarantees one secret reaches both containers.
      AUTH_SERVER_NGINX_MARKER_SECRET: ecs.Secret.fromSecretsManager(secretsBundle.nginxMarkerSecret),
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
      // Operator-supplied credentials. Injected as secrets, never as plain
      // environment values, so the task definition carries only the ARN
      // (issue #2063).
      ...Object.fromEntries(
        Object.entries(secretsBundle.sharedContainerCredentials)
          .map(([k, s]) => [k, ecs.Secret.fromSecretsManager(s)]),
      ),
    };

    // The egress credential encryption key is the registry's: it owns the
    // vault. The auth-server needs it only to encrypt cached OBO tokens, so
    // it gets a copy only when that cache is on, and its execution role is
    // denied the secret otherwise. Same condition as
    // terraform/aws-ecs/modules/mcp-gateway/ecs-services.tf.
    const egressKey = secretsBundle.egressCredentialEncryptionKey;
    const authNeedsEgressKey = egressKey !== undefined && config.egress.oboCacheEnabled;

    const registrySecrets: Record<string, ecs.Secret> = {
      ...sharedSecrets,
      KEYCLOAK_ADMIN_PASSWORD: ecs.Secret.fromSecretsManager(secretsBundle.keycloakAdminPassword),
      EMBEDDINGS_API_KEY: ecs.Secret.fromSecretsManager(secretsBundle.embeddingsApiKey),
      ...(egressKey
        ? { EGRESS_CREDENTIAL_ENCRYPTION_KEY: ecs.Secret.fromSecretsManager(egressKey) }
        : {}),
      // Registry only: it drives the PingFederate admin API.
      ...(secretsBundle.pingfederateAdminPassword
        ? { PF_ADMIN_PASS: ecs.Secret.fromSecretsManager(secretsBundle.pingfederateAdminPassword) }
        : {}),
      // Registry only: GitHub credentials for skill and server fetches.
      ...Object.fromEntries(
        Object.entries(secretsBundle.registryContainerCredentials)
          .map(([k, s]) => [k, ecs.Secret.fromSecretsManager(s)]),
      ),
    };

    const authSecrets: Record<string, ecs.Secret> = {
      ...sharedSecrets,
      ...(authNeedsEgressKey
        ? { EGRESS_CREDENTIAL_ENCRYPTION_KEY: ecs.Secret.fromSecretsManager(egressKey!) }
        : {}),
    };

    // The auth-server's execution role is denied the secrets its containers
    // never receive, so a grant cannot outlive an injection.
    const authExecStatements = secretsBundle.accessStatementsExcept([
      secretsBundle.pingfederateAdminPassword,
      ...(authNeedsEgressKey ? [] : [egressKey]),
      ...Object.values(secretsBundle.registryContainerCredentials),
    ]);

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

    // Per-user egress credential vault. The registry does runtime CRUD on one
    // secret per user under the configured path prefix, so unlike the other
    // secrets in this stack these cannot be enumerated at synth time and the
    // grant has to be a prefix.
    //
    // The prefix is validated in registry-config.ts against a character set
    // that excludes `*`, so an operator cannot widen this grant past the
    // vault through config. Secrets Manager appends a random 6-character
    // suffix to every ARN, which is what the trailing wildcard covers.
    if (config.egress.enabled && config.egress.secretStoreBackend === 'secrets-manager') {  // pragma: allowlist secret
      const vaultStatements: iam.PolicyStatement[] = [
        new iam.PolicyStatement({
          sid: 'EgressCredentialVaultCrud',
          effect: iam.Effect.ALLOW,
          actions: [
            'secretsmanager:CreateSecret',
            'secretsmanager:GetSecretValue',
            'secretsmanager:PutSecretValue',
            'secretsmanager:DeleteSecret',
            'secretsmanager:DescribeSecret',
            'secretsmanager:ListSecretVersionIds',
          ],
          resources: [
            `arn:${this.partition}:secretsmanager:${this.region}:${this.account}:secret:${config.egress.secretsManagerPathPrefix}/*`,
          ],
        }),
      ];
      // Only for a customer-managed CMK. The AWS-managed Secrets Manager key
      // needs no explicit grant. registry-config.ts requires a KMS KEY ARN
      // here, not an alias ARN: IAM accepts only a key ARN in the Resource of
      // a statement granting cryptographic operations.
      if (config.egress.secretsManagerKmsKeyArn !== '') {
        vaultStatements.push(
          new iam.PolicyStatement({
            sid: 'EgressVaultCmkAccess',
            effect: iam.Effect.ALLOW,
            actions: ['kms:Decrypt', 'kms:GenerateDataKey'],
            resources: [config.egress.secretsManagerKmsKeyArn],
          }),
        );
      }
      registryTaskRolePolicies.push(
        new iam.ManagedPolicy(this, 'EgressVaultAccessPolicy', { statements: vaultStatements }),
      );
    }

    // The auth-server's OBO exchange cache lives in the same Secrets Manager
    // vault, under the reserved "obo-cache" namespace. The namespace is
    // base64url(NFC("obo-cache")) unpadded, which is the literal below, so
    // this grant reaches the cache entries and nothing else: the auth-server
    // cannot read a user's egress credentials with it.
    //
    // Without this grant the cache reads and writes return AccessDenied and
    // the auth-server falls back to a stateless exchange without saying so.
    const authTaskRolePolicies: iam.IManagedPolicy[] = [];
    if (
      config.egress.enabled
      && config.egress.oboCacheEnabled
      && config.egress.secretStoreBackend === 'secrets-manager'  // pragma: allowlist secret
    ) {
      authTaskRolePolicies.push(
        new iam.ManagedPolicy(this, 'AuthServerOboCacheVaultPolicy', {
          statements: [
            new iam.PolicyStatement({
              sid: 'OboCacheVaultCrud',
              effect: iam.Effect.ALLOW,
              // Exactly the four calls the secret store makes
              // (registry/secrets/secrets_manager/store.py), matching
              // terraform/aws-ecs/modules/mcp-gateway/iam.tf.
              actions: [
                'secretsmanager:CreateSecret',
                'secretsmanager:GetSecretValue',
                'secretsmanager:PutSecretValue',
                'secretsmanager:DeleteSecret',
              ],
              resources: [
                `arn:${this.partition}:secretsmanager:${this.region}:${this.account}:secret:${config.egress.secretsManagerPathPrefix}/${OBO_CACHE_NAMESPACE_SEGMENT}/*`,
              ],
            }),
            ...(config.egress.secretsManagerKmsKeyArn !== ''
              ? [new iam.PolicyStatement({
                  sid: 'OboCacheVaultCmkAccess',
                  effect: iam.Effect.ALLOW,
                  actions: ['kms:Decrypt', 'kms:GenerateDataKey'],
                  resources: [config.egress.secretsManagerKmsKeyArn],
                })]
              : []),
          ],
        }),
      );
    }

    // Read-only Cognito access for the registry's IAM management UI (list
    // groups and users). Scoped to the one configured user pool; the pool id
    // is format-validated in registry-config.ts because it lands in this ARN.
    if (config.cognito.enabled) {
      registryTaskRolePolicies.push(
        new iam.ManagedPolicy(this, 'CognitoIamReadPolicy', {
          statements: [
            new iam.PolicyStatement({
              sid: 'CognitoIdpReadOnly',
              effect: iam.Effect.ALLOW,
              actions: [
                'cognito-idp:ListGroups',
                'cognito-idp:ListUsers',
                'cognito-idp:AdminListGroupsForUser',
              ],
              resources: [
                `arn:${this.partition}:cognito-idp:${this.region}:${this.account}:userpool/${config.cognito.userPoolId}`,
              ],
            }),
          ],
        }),
      );
    }

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
        { port: REGISTRY_EGRESS_INTERNAL_PORT, name: 'egress-internal' },
      ],
      vpc,
      subnets: privateSubnets,
      cluster: this.ecsCluster,
      serviceConnectNamespaceArn: cloudMapNamespace.namespaceArn,
      serviceConnect: { port: 8080, dnsName: 'registry', portName: 'http', discoveryName: 'registry' },
      // Advertise the internal egress-token vend listener so the auth-server
      // reaches it as registry:8091. Separate from the :8080 alias the ALB
      // fronts, so this listener stays off the public path.
      additionalServiceConnect: [
        {
          port: REGISTRY_EGRESS_INTERNAL_PORT,
          dnsName: 'registry',
          portName: 'egress-internal',
          discoveryName: 'registry-egress-internal',
        },
      ],
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
      containerPort: AUTH_SERVER_CONTAINER_PORT,
      vpc,
      subnets: privateSubnets,
      cluster: this.ecsCluster,
      serviceConnectNamespaceArn: cloudMapNamespace.namespaceArn,
      serviceConnect: {
        port: AUTH_SERVER_ALIAS_PORT,
        dnsName: 'auth-server',
        portName: 'auth-server',
        discoveryName: 'auth-server',
      },
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
      additionalTaskRolePolicies: authTaskRolePolicies,
      additionalExecRoleStatements: authExecStatements,
      healthCheckCommand: `curl -f http://localhost:${AUTH_SERVER_CONTAINER_PORT}/health || exit 1`,
      namePrefix,
      desiredCount: config.replicas.auth,
    });
    this.authEcsSg = authService.securityGroup;

    // The container port, not the alias port: the Service Connect proxy's
    // ingress listener terminates on the container port, so that is what
    // arrives at the task's network interface.
    authService.securityGroup.addIngressRule(
      registryService.securityGroup,
      ec2.Port.tcp(AUTH_SERVER_CONTAINER_PORT),
      'Allow registry to access auth server',
    );

    // The auth-server calls the registry's internal egress-token vend on its
    // own listener. Only this one source: the port is never ALB-fronted, and
    // the vend mints credentials.
    if (config.egress.enabled) {
      registryService.securityGroup.addIngressRule(
        authService.securityGroup,
        ec2.Port.tcp(REGISTRY_EGRESS_INTERNAL_PORT),
        'Allow auth-server to reach the registry internal egress-token vend',
      );
    }

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
        // Without this, mcpgw sends no X-Forwarded-Host or X-Forwarded-Proto,
        // so URLs the registry builds for mcpgw tool results point at the
        // internal registry:8080 host instead of the public one.
        REGISTRY_EXTERNAL_URL: this.registryUrl,
        REGISTRY_USERNAME: 'admin',
        ...otelEnv('mcp-mcpgw'),
      },
      efsVolumes: [{
        volumeName: 'mcpgw-data',
        fileSystemId: efsResources.fileSystem.fileSystemId,
        accessPointId: accessPoints['mcpgwData'].accessPointId,
        containerPath: '/app/data',
      }],
      ingressSecurityGroup: registryService.securityGroup,
      // No secrets grant: McpServerService injects no container secrets, so
      // the whole bundle here would let mcpgw's execution role read the
      // PingFederate admin password and the egress credential key it never
      // receives. Terraform has the same over-grant; this does not.
      // Leave room for the ADOT sidecar attached below: this construct pins
      // the app container's CPU, and ECS rejects a task whose container CPU
      // values sum above the task CPU.
      ...(config.enableObservability
        ? {
            reservedSidecarCpu: ADOT_SIDECAR_CPU,
            reservedSidecarMemoryMiB: ADOT_SIDECAR_MEMORY_MIB,
          }
        : {}),
      namePrefix,
      desiredCount: config.replicas.mcpgw,
    });

    if (mcpgwService.securityGroup) {
      // :8080 (nginx) only. Terraform dropped the :7860 rule when the
      // registry's internal auth was hardened: 7860 is the app process, which
      // binds loopback and is meant to be reached only through nginx in the
      // same task.
      registryService.securityGroup.addIngressRule(
        mcpgwService.securityGroup, ec2.Port.tcp(8080), 'Port 8080 from mcpgw',
      );
      // Auth-server mcp-proxy forwards to mcpgw:8003 (parity with terraform
      // auth_to_mcpgw rule at terraform/aws-ecs/modules/mcp-gateway/ecs-services.tf:2460)
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
      metricsAdminApiKeySecret: secretsBundle.metricsAdminApiKey,
      otlpExporterHeadersSecret: secretsBundle.otlpExporterHeaders,
      grafanaAdminPasswordSecret: secretsBundle.grafanaAdminPassword,
      registryServiceSg: registryService.securityGroup,
      authServiceSg: authService.securityGroup,
      albSg: alb.albSg,
      // Issue #1122: the metrics-service collector only scrapes the metrics
      // service. These three push over OTLP, so each needs its own sidecar or
      // its metrics never reach AMP.
      otlpSidecarTargets: [
        {
          name: 'registry',
          taskDefinition: registryService.taskDefinition,
          appContainerName: 'registry',
        },
        {
          name: 'auth-server',
          taskDefinition: authService.taskDefinition,
          appContainerName: 'auth-server',
        },
        ...(mcpgwService.taskDefinition
          ? [{
              name: 'mcpgw',
              taskDefinition: mcpgwService.taskDefinition,
              appContainerName: 'mcpgw-server',
            }]
          : []),
      ],
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
