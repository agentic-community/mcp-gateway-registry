/**
 * RegistrySecrets - L3 construct for the application Secrets Manager bundle
 * and KMS key. Encapsulates conditional secrets (Entra/Okta/Auth0/observability).
 *
 * Translated from: terraform/aws-ecs/modules/mcp-gateway/secrets.tf
 */

import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';
import { RegistryConfig } from '../registry-config';

export interface RegistrySecretsProps {
  readonly config: RegistryConfig;
  /** ARN of the DocumentDB secret (so it's added to the IAM allowlist) */
  readonly documentDbSecretArn?: string;
}

export class RegistrySecrets extends Construct {
  public readonly kmsKey: kms.Key;
  public readonly secretKey: secretsmanager.Secret;
  /**
   * Shared marker proving a `/validate` call arrived through registry nginx.
   * Both the registry and the auth-server read it from the SAME secret: the
   * auth-server mints an internal token only when the marker it receives
   * matches its own, so a mismatch turns every registry API and UI request
   * into a 401.
   */
  public readonly nginxMarkerSecret: secretsmanager.Secret;
  public readonly keycloakClientSecret: secretsmanager.Secret;
  public readonly keycloakM2mClientSecret: secretsmanager.Secret;
  public readonly keycloakAdminPassword: secretsmanager.Secret;
  public readonly embeddingsApiKey: secretsmanager.Secret;
  public readonly entraClientSecret?: secretsmanager.Secret;
  public readonly oktaClientSecret?: secretsmanager.Secret;
  public readonly oktaM2mClientSecret?: secretsmanager.Secret;
  public readonly oktaApiToken?: secretsmanager.Secret;
  public readonly auth0ClientSecret?: secretsmanager.Secret;
  public readonly auth0M2mClientSecret?: secretsmanager.Secret;
  /**
   * Secrets referenced by ARN rather than created from a configured value.
   *
   * `_createSecret` writes a literal into `SecretString`, which lands in the
   * synthesized template and therefore sits in plaintext in `cdk.out` and in
   * the deployed CloudFormation stack. These four are imported instead, the
   * same way the DocumentDB secret already is, so no value passes through
   * synthesis. The remaining literal-valued secrets above predate this and
   * are tracked separately.
   */
  public readonly cognitoClientSecret?: secretsmanager.ISecret;
  public readonly pingfederateClientSecret?: secretsmanager.ISecret;
  public readonly pingfederateM2mClientSecret?: secretsmanager.ISecret;
  public readonly pingfederateAdminPassword?: secretsmanager.ISecret;
  public readonly egressCredentialEncryptionKey?: secretsmanager.ISecret;
  public readonly metricsApiKey?: secretsmanager.Secret;
  public readonly metricsKeyPepper?: secretsmanager.Secret;
  public readonly metricsAdminApiKey?: secretsmanager.Secret;
  public readonly otlpExporterHeaders?: secretsmanager.Secret;
  public readonly grafanaAdminPassword?: secretsmanager.Secret;
  /**
   * Operator-supplied credentials injected into containers, keyed by the
   * environment variable name. Only a configured (non-empty) value gets a
   * secret, so an unset credential injects nothing, as it did when these
   * were plain environment entries set to ''.
   *
   * These used to be plain `environment` values, which stored them in the ECS
   * task definition (issue #2063). The value still passes through synthesis,
   * like the other literal-valued secrets above.
   */
  public readonly sharedContainerCredentials: Record<string, secretsmanager.Secret>;
  /** Same as {@link sharedContainerCredentials}, for the registry only. */
  public readonly registryContainerCredentials: Record<string, secretsmanager.Secret>;

  /**
   * IAM statements granting GetSecretValue + KMS decrypt for every secret in
   * the bundle. Use {@link accessStatementsExcept} for a service that does not
   * receive all of them.
   */
  public readonly accessStatements: iam.PolicyStatement[];

  /** Every secret ARN in the bundle, including imported ones. */
  private readonly allSecretArns: string[];

  constructor(scope: Construct, id: string, props: RegistrySecretsProps) {
    super(scope, id);

    const { config } = props;
    const stack = cdk.Stack.of(this);
    const { name: namePrefix } = config;

    // KMS key
    this.kmsKey = new kms.Key(this, 'KmsKey', {
      description: 'KMS key for MCP Gateway application secrets encryption',
      enableKeyRotation: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      pendingWindow: cdk.Duration.days(7),
    });

    this.kmsKey.addToResourcePolicy(new iam.PolicyStatement({
      sid: 'AllowEcsTaskExecDecrypt',
      effect: iam.Effect.ALLOW,
      principals: [new iam.AnyPrincipal()],
      actions: ['kms:Decrypt', 'kms:DescribeKey'],
      resources: ['*'],
      conditions: {
        StringEquals: { 'aws:PrincipalAccount': stack.account },
        StringLike: { 'aws:PrincipalArn': `arn:aws:iam::${stack.account}:role/*task-exec*` },
      },
    }));
    this.kmsKey.addToResourcePolicy(new iam.PolicyStatement({
      sid: 'AllowCloudWatchLogs',
      effect: iam.Effect.ALLOW,
      principals: [new iam.ServicePrincipal(`logs.${stack.region}.amazonaws.com`)],
      actions: ['kms:Encrypt', 'kms:Decrypt', 'kms:ReEncrypt*', 'kms:GenerateDataKey*', 'kms:CreateGrant', 'kms:DescribeKey'],
      resources: ['*'],
      conditions: {
        ArnLike: { 'kms:EncryptionContext:aws:logs:arn': `arn:aws:logs:${stack.region}:${stack.account}:log-group:*` },
      },
    }));
    new kms.Alias(this, 'KmsAlias', { aliasName: `alias/${namePrefix}-secrets`, targetKey: this.kmsKey });

    // Always-on secrets
    this.secretKey = _createSecret(this, 'SecretKey', {
      description: 'Secret key for MCP Gateway Registry',
      kmsKey: this.kmsKey,
      generateString: { passwordLength: 64, excludePunctuation: false },
    });

    // Generated, not operator supplied: the registry and the auth-server only
    // have to agree with each other, and nothing outside the stack needs the
    // value. 64 characters of letters and digits clear all three checks in
    // registry/core/config.py: non-empty, at least 32 bytes, and every
    // character printable ASCII other than `$` (nginx renders it into a config
    // file and sends it as a header).
    this.nginxMarkerSecret = _createSecret(this, 'NginxMarkerSecret', {
      description:
        'Shared marker proving an auth-server /validate call arrived through registry nginx',
      kmsKey: this.kmsKey,
      generateString: { passwordLength: 64, excludePunctuation: true },
    });

    this.keycloakClientSecret = _createSecret(this, 'KeycloakClientSecret', {
      fixedName: 'mcp-gateway-keycloak-client-secret',
      description: 'Keycloak web client secret (updated by init-keycloak.sh)',
      kmsKey: this.kmsKey,
      stringValue: JSON.stringify({ client_secret: 'placeholder-will-be-updated-by-init-script' }),
    });
    this.keycloakM2mClientSecret = _createSecret(this, 'KeycloakM2mClientSecret', {
      fixedName: 'mcp-gateway-keycloak-m2m-client-secret',
      description: 'Keycloak M2M client secret (updated by init-keycloak.sh)',
      kmsKey: this.kmsKey,
      stringValue: JSON.stringify({ client_secret: 'placeholder-will-be-updated-by-init-script' }),
    });

    this.keycloakAdminPassword = _createSecret(this, 'KeycloakAdminPassword', {
      description: 'Keycloak admin password for Management API operations',
      kmsKey: this.kmsKey,
      stringValue: config.keycloak.adminPassword,
    });

    this.embeddingsApiKey = _createSecret(this, 'EmbeddingsApiKey', {
      description: 'API key for embeddings provider',
      kmsKey: this.kmsKey,
      stringValue: config.embeddings.apiKey || 'not-configured',
    });

    // Conditional IdP secrets
    if (config.entra.enabled) {
      this.entraClientSecret = _createSecret(this, 'EntraClientSecret', {
        description: 'Microsoft Entra ID client secret',
        kmsKey: this.kmsKey,
        stringValue: config.entra.clientSecret,
      });
    }

    if (config.okta.enabled) {
      this.oktaClientSecret = _createSecret(this, 'OktaClientSecret', {
        description: 'Okta client secret',
        kmsKey: this.kmsKey,
        stringValue: config.okta.clientSecret,
      });
      this.oktaM2mClientSecret = _createSecret(this, 'OktaM2mClientSecret', {
        description: 'Okta M2M client secret',
        kmsKey: this.kmsKey,
        stringValue: config.okta.m2mClientSecret,
      });
      this.oktaApiToken = _createSecret(this, 'OktaApiToken', {
        description: 'Okta API token',
        kmsKey: this.kmsKey,
        stringValue: config.okta.apiToken,
      });
    }

    if (config.auth0.enabled) {
      this.auth0ClientSecret = _createSecret(this, 'Auth0ClientSecret', {
        description: 'Auth0 client secret',
        kmsKey: this.kmsKey,
        stringValue: config.auth0.clientSecret,
      });
      this.auth0M2mClientSecret = _createSecret(this, 'Auth0M2mClientSecret', {
        description: 'Auth0 M2M client secret',
        kmsKey: this.kmsKey,
        stringValue: config.auth0.m2mClientSecret,
      });
    }

    // Imported, not created: see the note on the fields above.
    if (config.cognito.enabled) {
      this.cognitoClientSecret = secretsmanager.Secret.fromSecretCompleteArn(
        this, 'CognitoClientSecret', config.cognito.clientSecretArn,
      );
    }

    if (config.pingfederate.enabled) {
      this.pingfederateClientSecret = secretsmanager.Secret.fromSecretCompleteArn(
        this, 'PingFederateClientSecret', config.pingfederate.clientSecretArn,
      );
      // Only when a SEPARATE M2M secret is configured. The provider falls back
      // to the web client secret otherwise
      // (auth_server/providers/pingfederate.py), so creating an empty secret
      // here would both be rejected by CloudFormation (SecretString has a
      // minimum length of 1) and override a working fallback.
      if (config.pingfederate.m2mClientSecretArn !== '') {
        this.pingfederateM2mClientSecret = secretsmanager.Secret.fromSecretCompleteArn(
          this, 'PingFederateM2mClientSecret', config.pingfederate.m2mClientSecretArn,
        );
      }
      // Registry only. It drives the PingFederate admin API to create OAuth
      // clients and PCV users; the auth-server never calls that API.
      this.pingfederateAdminPassword = secretsmanager.Secret.fromSecretCompleteArn(
        this, 'PingFederateAdminPassword', config.pingfederate.adminPasswordArn,
      );
    }

    // Empty leaves credentials in plaintext, which is the application's legacy
    // behavior, so an unset key injects nothing rather than an empty value.
    if (config.egress.enabled && config.egress.credentialEncryptionKeyArn !== '') {
      this.egressCredentialEncryptionKey = secretsmanager.Secret.fromSecretCompleteArn(
        this, 'EgressCredentialEncryptionKey', config.egress.credentialEncryptionKeyArn,
      );
    }

    if (config.enableObservability) {
      this.metricsApiKey = _createSecret(this, 'MetricsApiKey', {
        description: 'API key for metrics-service',
        kmsKey: this.kmsKey,
        generateString: { passwordLength: 48, excludePunctuation: true },
      });
      // Per-deployment HMAC pepper for API-key hashing. The metrics-service
      // refuses to start without a strong value. Generated (not operator
      // supplied) so a fresh deploy does not fail closed; rotating it
      // invalidates existing API-key hashes.
      this.metricsKeyPepper = _createSecret(this, 'MetricsKeyPepper', {
        description: 'Per-deployment HMAC pepper for metrics-service API-key hashing',
        kmsKey: this.kmsKey,
        generateString: { passwordLength: 64, excludePunctuation: true },
      });
      // Admin API key gating the metrics-service /admin/* endpoints (retention,
      // cleanup, database stats). Privilege-separated from ingest: an ingest key
      // is rejected on /admin/*, and /admin/* denies until this is set.
      // Generated separately from the ingest key so it is guaranteed distinct.
      this.metricsAdminApiKey = _createSecret(this, 'MetricsAdminApiKey', {
        description: 'Admin API key gating metrics-service /admin/* endpoints (distinct from ingest key)',
        kmsKey: this.kmsKey,
        generateString: { passwordLength: 48, excludePunctuation: true },
      });
      // Issue #1325: store the Grafana admin password in Secrets Manager instead
      // of injecting it as a plaintext container env value.
      this.grafanaAdminPassword = _createSecret(this, 'GrafanaAdminPassword', {
        description: 'Grafana admin password',
        kmsKey: this.kmsKey,
        stringValue: config.grafanaAdminPassword,
      });
    }

    if (config.enableObservability && config.otel.otlpEndpoint !== '') {
      this.otlpExporterHeaders = _createSecret(this, 'OtlpExporterHeaders', {
        description: 'OTLP exporter authentication headers',
        kmsKey: this.kmsKey,
        stringValue: config.otel.exporterOtlpHeaders,
      });
    }

    // Operator-supplied container credentials. Same set and same split as
    // terraform/aws-ecs/modules/mcp-gateway/secrets.tf (container_secret_values).
    this.sharedContainerCredentials = _createCredentialSecrets(this, this.kmsKey, [
      { envName: 'AUTH0_MANAGEMENT_API_TOKEN', id: 'Auth0ManagementApiToken', value: config.auth0.managementApiToken },
      { envName: 'REGISTRY_API_TOKEN', id: 'RegistryApiToken', value: config.staticTokenAuth.registryApiToken },
      { envName: 'FEDERATION_STATIC_TOKEN', id: 'FederationStaticToken', value: config.federation.staticToken },
      { envName: 'FEDERATION_ENCRYPTION_KEY', id: 'FederationEncryptionKey', value: config.federation.encryptionKey },
      { envName: 'ANS_API_KEY', id: 'AnsApiKey', value: config.ans.apiKey },
      { envName: 'ANS_API_SECRET', id: 'AnsApiSecret', value: config.ans.apiSecret },
    ]);
    this.registryContainerCredentials = _createCredentialSecrets(this, this.kmsKey, [
      { envName: 'GITHUB_PAT', id: 'GithubPat', value: config.github.pat },
      { envName: 'GITHUB_APP_PRIVATE_KEY', id: 'GithubAppPrivateKey', value: config.github.appPrivateKey },
    ]);

    // IAM statements
    const arns: string[] = [
      this.secretKey.secretArn,
      this.nginxMarkerSecret.secretArn,
      this.keycloakClientSecret.secretArn,
      this.keycloakM2mClientSecret.secretArn,
      this.keycloakAdminPassword.secretArn,
      this.embeddingsApiKey.secretArn,
    ];
    if (props.documentDbSecretArn) arns.push(props.documentDbSecretArn);
    for (const s of [
      this.entraClientSecret, this.oktaClientSecret, this.oktaM2mClientSecret,
      this.oktaApiToken, this.auth0ClientSecret, this.auth0M2mClientSecret,
      this.cognitoClientSecret, this.pingfederateClientSecret,
      this.pingfederateM2mClientSecret, this.pingfederateAdminPassword,
      this.egressCredentialEncryptionKey,
      this.metricsApiKey, this.metricsKeyPepper, this.metricsAdminApiKey,
      this.otlpExporterHeaders, this.grafanaAdminPassword,
    ]) if (s) arns.push(s.secretArn);
    for (const s of Object.values(this.sharedContainerCredentials)) arns.push(s.secretArn);
    for (const s of Object.values(this.registryContainerCredentials)) arns.push(s.secretArn);

    this.allSecretArns = arns;
    this.accessStatements = this.accessStatementsExcept();
  }

  /**
   * Read access to every secret in the bundle except the ones listed.
   *
   * An execution role only needs the secrets its own containers are given, and
   * two of these are single-service: the PingFederate admin password is the
   * registry's, and the egress credential key reaches the auth-server only
   * when its OBO cache is on. Pass the ones a service does not receive so its
   * role cannot read them.
   */
  public accessStatementsExcept(
    exclude: Array<secretsmanager.ISecret | undefined> = [],
  ): iam.PolicyStatement[] {
    const excluded = new Set(
      exclude.filter((s): s is secretsmanager.ISecret => s !== undefined).map((s) => s.secretArn),
    );
    return [
      new iam.PolicyStatement({
        sid: 'SecretsManagerAccess',
        effect: iam.Effect.ALLOW,
        actions: ['secretsmanager:GetSecretValue'],
        resources: this.allSecretArns.filter((arn) => !excluded.has(arn)),
      }),
      // This stack's own key only. An IMPORTED secret (Cognito, PingFederate,
      // egress credential key, DocumentDB) is encrypted with whatever key its
      // creator chose. The AWS-managed Secrets Manager key needs no explicit
      // grant, but a customer-managed one does, and CDK cannot discover it
      // from the secret ARN. config.yaml says so where the ARNs are set.
      new iam.PolicyStatement({
        sid: 'KmsDecrypt',
        effect: iam.Effect.ALLOW,
        actions: ['kms:Decrypt', 'kms:DescribeKey'],
        resources: [this.kmsKey.keyArn],
      }),
    ];
  }
}

interface CreateSecretOptions {
  description: string;
  kmsKey: kms.IKey;
  fixedName?: string;
  stringValue?: string;
  generateString?: { passwordLength: number; excludePunctuation: boolean };
}

interface CredentialSecretSpec {
  /** Environment variable the container reads the value from. */
  envName: string;
  /** Construct id of the secret. */
  id: string;
  /** Operator-supplied value. Empty means not configured. */
  value: string;
}

/**
 * Create one secret per configured credential, keyed by its env var name.
 *
 * An empty value is skipped: CloudFormation rejects an empty SecretString,
 * and the application treats an unset variable the same as an empty one.
 */
function _createCredentialSecrets(
  scope: Construct,
  kmsKey: kms.IKey,
  specs: CredentialSecretSpec[],
): Record<string, secretsmanager.Secret> {
  const secrets: Record<string, secretsmanager.Secret> = {};
  for (const spec of specs) {
    if (spec.value === '') continue;
    secrets[spec.envName] = _createSecret(scope, spec.id, {
      description: `${spec.envName} for MCP Gateway Registry containers`,
      kmsKey,
      stringValue: spec.value,
    });
  }
  return secrets;
}

function _createSecret(scope: Construct, id: string, opts: CreateSecretOptions): secretsmanager.Secret {
  const props: secretsmanager.SecretProps = {
    description: opts.description,
    encryptionKey: opts.kmsKey,
    removalPolicy: cdk.RemovalPolicy.DESTROY,
    ...(opts.fixedName ? { secretName: opts.fixedName } : {}),
    ...(opts.generateString
      ? { generateSecretString: opts.generateString }
      : {}),
  };

  const secret = new secretsmanager.Secret(scope, id, props);

  // For static-string secrets, force the literal value via L1 (skipping the
  // GenerateSecretString CDK adds by default).
  if (opts.stringValue !== undefined && !opts.generateString) {
    const cfn = secret.node.defaultChild as secretsmanager.CfnSecret;
    cfn.addPropertyOverride('SecretString', opts.stringValue);
    cfn.addPropertyDeletionOverride('GenerateSecretString');
  }

  return secret;
}
