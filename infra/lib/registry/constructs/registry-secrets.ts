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
  public readonly metricsApiKey?: secretsmanager.Secret;
  public readonly metricsKeyPepper?: secretsmanager.Secret;
  public readonly otlpExporterHeaders?: secretsmanager.Secret;
  public readonly grafanaAdminPassword?: secretsmanager.Secret;

  /** IAM statements granting GetSecretValue + KMS decrypt for all secrets above */
  public readonly accessStatements: iam.PolicyStatement[];

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

    // IAM statements
    const arns: string[] = [
      this.secretKey.secretArn,
      this.keycloakClientSecret.secretArn,
      this.keycloakM2mClientSecret.secretArn,
      this.keycloakAdminPassword.secretArn,
      this.embeddingsApiKey.secretArn,
    ];
    if (props.documentDbSecretArn) arns.push(props.documentDbSecretArn);
    for (const s of [
      this.entraClientSecret, this.oktaClientSecret, this.oktaM2mClientSecret,
      this.oktaApiToken, this.auth0ClientSecret, this.auth0M2mClientSecret,
      this.metricsApiKey, this.metricsKeyPepper, this.otlpExporterHeaders,
      this.grafanaAdminPassword,
    ]) if (s) arns.push(s.secretArn);

    this.accessStatements = [
      new iam.PolicyStatement({
        sid: 'SecretsManagerAccess',
        effect: iam.Effect.ALLOW,
        actions: ['secretsmanager:GetSecretValue'],
        resources: arns,
      }),
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
