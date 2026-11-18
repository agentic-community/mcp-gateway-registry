/**
 * Parity and startup-safety assertions for RegistryServiceStack.
 *
 * These pin the gaps found in issue #1961, so a later change that drops one
 * fails here rather than as a crash-looping task or a silent 401.
 */

import { Match, Template } from 'aws-cdk-lib/assertions';

import {
  buildStacks,
  container,
  containers,
  emptySecretStrings,
  envOf,
  grantedSecretArns,
  secretArn,
  secretIdentity,
  secretsOf,
  testConfig,
} from './helpers';

const AUTH_CONTAINER_PORT = 18888;
const AUTH_ALIAS_PORT = 8888;
const EGRESS_INTERNAL_PORT = 8091;

describe('nginx marker secret', () => {
  let template: Template;

  beforeAll(() => {
    template = buildStacks(testConfig()).template;
  });

  it('injects AUTH_SERVER_NGINX_MARKER_SECRET into both services', () => {
    for (const name of ['registry', 'auth-server']) {
      expect(secretsOf(container(template, name))).toHaveProperty(
        'AUTH_SERVER_NGINX_MARKER_SECRET',
      );
    }
  });

  it('gives both services the SAME marker secret', () => {
    // The auth-server mints an internal token only when the marker nginx sent
    // matches its own, so two different secrets 401 every request.
    const registry = secretsOf(container(template, 'registry'));
    const auth = secretsOf(container(template, 'auth-server'));
    expect(registry.AUTH_SERVER_NGINX_MARKER_SECRET).toEqual(
      auth.AUTH_SERVER_NGINX_MARKER_SECRET,
    );
  });

  it('generates the marker with no punctuation and at least 32 characters', () => {
    // registry/core/config.py rejects a marker under 32 bytes or carrying a
    // character outside printable ASCII minus `$`.
    template.hasResourceProperties('AWS::SecretsManager::Secret', {
      Description: Match.stringLikeRegexp('marker'),
      GenerateSecretString: {
        PasswordLength: Match.exact(64),
        ExcludePunctuation: true,
      },
    });
  });
});

describe('secret grants', () => {
  it('grants read on every secret any container references', () => {
    // An injected secret the execution role cannot read stops the task before
    // the application logs anything, so the grant list and the injected list
    // have to stay matched.
    const { template } = buildStacks(testConfig());
    const granted = grantedSecretArns(template);

    const referenced = new Map<string, string>();
    for (const [key, def] of Object.entries(containers(template))) {
      for (const secret of def.Secrets ?? []) {
        referenced.set(secretIdentity(secret.ValueFrom), `${key} ${secret.Name}`);
      }
    }

    expect(referenced.size).toBeGreaterThan(0);
    const ungranted = [...referenced.entries()]
      .filter(([identity]) => !granted.has(identity))
      .map(([identity, where]) => `${where} -> ${identity}`);
    expect(ungranted).toEqual([]);
  });
});

describe('trusted real IP CIDRs', () => {
  it('defaults to the VPC CIDR on the registry', () => {
    const config = testConfig();
    const { template } = buildStacks(config);
    expect(envOf(container(template, 'registry')).TRUSTED_REAL_IP_CIDRS).toEqual(config.vpcCidr);
  });

  it('honors an explicit override', () => {
    const config = testConfig();
    config.trustedRealIpCidrs = '10.0.0.0/16,130.176.0.0/16';
    const { template } = buildStacks(config);
    expect(envOf(container(template, 'registry')).TRUSTED_REAL_IP_CIDRS).toEqual(
      '10.0.0.0/16,130.176.0.0/16',
    );
  });
});

describe('auth-server listen port', () => {
  let template: Template;

  beforeAll(() => {
    template = buildStacks(testConfig()).template;
  });

  it('binds the container port the Service Connect proxy does not hold', () => {
    const auth = container(template, 'auth-server');
    expect(envOf(auth).AUTH_LISTEN_PORT).toEqual(String(AUTH_CONTAINER_PORT));
    expect(auth.PortMappings?.map((p) => p.ContainerPort)).toEqual([AUTH_CONTAINER_PORT]);
  });

  it('health-checks the port the app actually listens on', () => {
    const command = container(template, 'auth-server').HealthCheck?.Command ?? [];
    expect(command.join(' ')).toContain(`localhost:${AUTH_CONTAINER_PORT}`);
  });

  it('keeps callers on the 8888 alias', () => {
    // Callers must not learn the container port: the alias is the contract.
    for (const name of ['registry', 'auth-server']) {
      expect(envOf(container(template, name)).AUTH_SERVER_URL).toEqual(
        `http://auth-server:${AUTH_ALIAS_PORT}`,
      );
    }
    template.hasResourceProperties('AWS::ECS::Service', {
      ServiceConnectConfiguration: {
        Services: Match.arrayWith([
          Match.objectLike({
            PortName: 'auth-server',
            ClientAliases: Match.arrayWith([Match.objectLike({ Port: AUTH_ALIAS_PORT })]),
          }),
        ]),
      },
    });
  });

  it('opens the security group on the container port, not the alias', () => {
    template.hasResourceProperties('AWS::EC2::SecurityGroupIngress', {
      FromPort: AUTH_CONTAINER_PORT,
      ToPort: AUTH_CONTAINER_PORT,
      Description: 'Allow registry to access auth server',
    });
  });
});

describe('egress auth disabled (the default)', () => {
  let template: Template;

  beforeAll(() => {
    template = buildStacks(testConfig()).template;
  });

  it('reports egress off to both services', () => {
    for (const name of ['registry', 'auth-server']) {
      expect(envOf(container(template, name)).EGRESS_AUTH_ENABLED).toEqual('false');
    }
  });

  it('creates no credential encryption key and no vault grant', () => {
    expect(secretsOf(container(template, 'registry'))).not.toHaveProperty(
      'EGRESS_CREDENTIAL_ENCRYPTION_KEY',
    );
    const vaultActions = JSON.stringify(template.findResources('AWS::IAM::ManagedPolicy'));
    expect(vaultActions).not.toContain('secretsmanager:CreateSecret');
  });

  it('opens no ingress on the internal vend port', () => {
    const ingress = JSON.stringify(template.findResources('AWS::EC2::SecurityGroupIngress'));
    const sgs = JSON.stringify(template.findResources('AWS::EC2::SecurityGroup'));
    expect(ingress + sgs).not.toContain(`"FromPort":${EGRESS_INTERNAL_PORT}`);
  });
});

describe('egress auth enabled', () => {
  let template: Template;

  beforeAll(() => {
    const config = testConfig();
    config.storageBackend = 'documentdb';
    config.egress = {
      ...config.egress,
      enabled: true,
      credentialEncryptionKeyArn: secretArn('mcp/egress-cred-key'),
      // Whitespace-separated: the application splits on whitespace, so a
      // comma-separated list would parse as one audience and match nothing.
      oboAllowedAudiences: 'api://upstream-one api://upstream-two',
      secretsManagerPathPrefix: 'mcp/egress',  // pragma: allowlist secret
    };
    template = buildStacks(config).template;
  });

  it('maps the internal vend port and advertises it on Service Connect', () => {
    const registry = container(template, 'registry');
    const mapping = registry.PortMappings?.find((p) => p.ContainerPort === EGRESS_INTERNAL_PORT);
    expect(mapping?.Name).toEqual('egress-internal');

    template.hasResourceProperties('AWS::ECS::Service', {
      ServiceConnectConfiguration: {
        Services: Match.arrayWith([
          Match.objectLike({
            PortName: 'egress-internal',
            DiscoveryName: 'registry-egress-internal',
            ClientAliases: Match.arrayWith([
              Match.objectLike({ Port: EGRESS_INTERNAL_PORT, DnsName: 'registry' }),
            ]),
          }),
        ]),
      },
    });
  });

  it('lets only the auth-server reach the vend port', () => {
    template.hasResourceProperties('AWS::EC2::SecurityGroupIngress', {
      FromPort: EGRESS_INTERNAL_PORT,
      ToPort: EGRESS_INTERNAL_PORT,
      Description: Match.stringLikeRegexp('egress-token vend'),
    });
  });

  it('points the auth-server at the internal vend URL', () => {
    expect(envOf(container(template, 'auth-server')).EGRESS_REGISTRY_INTERNAL_URL).toEqual(
      `http://registry:${EGRESS_INTERNAL_PORT}`,
    );
  });

  it('gives the encryption key to the registry only, as a secret', () => {
    // The registry owns the vault. The auth-server needs the key solely to
    // encrypt cached OBO tokens, so with the cache off it gets neither the
    // value nor a grant to read it.
    const registry = container(template, 'registry');
    expect(secretsOf(registry)).toHaveProperty('EGRESS_CREDENTIAL_ENCRYPTION_KEY');
    expect(envOf(registry)).not.toHaveProperty('EGRESS_CREDENTIAL_ENCRYPTION_KEY');

    const auth = container(template, 'auth-server');
    expect(secretsOf(auth)).not.toHaveProperty('EGRESS_CREDENTIAL_ENCRYPTION_KEY');
    expect(envOf(auth)).not.toHaveProperty('EGRESS_CREDENTIAL_ENCRYPTION_KEY');
  });

  it('passes the OBO audience allowlist to the registry, whitespace separated', () => {
    expect(envOf(container(template, 'registry')).EGRESS_OBO_ALLOWED_AUDIENCES).toEqual(
      'api://upstream-one api://upstream-two',
    );
  });

  it('passes the configured CMK to both containers', () => {
    // The grant alone is not enough: without the env var the application
    // creates vault secrets under the AWS-managed key and the CMK is never
    // used.
    for (const name of ['registry', 'auth-server']) {
      expect(envOf(container(template, name))).toHaveProperty('SECRETS_MANAGER_KMS_KEY_ID');
    }
  });

  it('adds no OBO cache policy when the cache is off', () => {
    const policies = JSON.stringify(template.findResources('AWS::IAM::ManagedPolicy'));
    expect(policies).not.toContain('OboCacheVaultCrud');
  });

  it('scopes the vault grant to the configured path prefix', () => {
    const policies = template.findResources('AWS::IAM::ManagedPolicy');
    const vault = Object.values(policies).find((p) =>
      JSON.stringify(p).includes('EgressCredentialVaultCrud'),
    );
    expect(vault).toBeDefined();
    const serialized = JSON.stringify(vault);
    expect(serialized).toContain('secret:mcp/egress/*');
    // No wildcard resource, and no KMS grant without a configured CMK.
    expect(serialized).not.toContain('"Resource":"*"');
    expect(serialized).not.toContain('kms:GenerateDataKey');
  });
});

describe('egress with the OBO cache enabled', () => {
  let template: Template;

  beforeAll(() => {
    const config = testConfig();
    config.storageBackend = 'documentdb';
    config.egress = {
      ...config.egress,
      enabled: true,
      credentialEncryptionKeyArn: secretArn('mcp/egress-cred-key'),
      oboCacheEnabled: true,
      secretsManagerKmsKeyArn:
        'arn:aws:kms:us-west-2:123456789012:key/11111111-2222-3333-4444-555555555555',
    };
    template = buildStacks(config).template;
  });

  it('gives the auth-server the encryption key once the cache needs it', () => {
    expect(secretsOf(container(template, 'auth-server')))
      .toHaveProperty('EGRESS_CREDENTIAL_ENCRYPTION_KEY');
  });

  it('grants the auth-server its own cache namespace and nothing more', () => {
    // base64url(NFC("obo-cache")) unpadded. Without this grant the cache
    // reads and writes get AccessDenied and the auth-server silently falls
    // back to stateless exchanges.
    const policies = template.findResources('AWS::IAM::ManagedPolicy');
    const cachePolicy = Object.values(policies).find((p) =>
      JSON.stringify(p).includes('OboCacheVaultCrud'),
    );
    expect(cachePolicy).toBeDefined();
    const serialized = JSON.stringify(cachePolicy);
    expect(serialized).toContain('secret:mcp/egress/b2JvLWNhY2hl/*');
    // Not the whole vault: the auth-server must not reach a user's
    // egress credentials with this.
    expect(serialized).not.toContain('secret:mcp/egress/*');
    expect(serialized).not.toContain('"Resource":"*"');
    // The CMK is configured, so the cache needs decrypt on it too.
    expect(serialized).toContain('11111111-2222-3333-4444-555555555555');
  });

  it('grants the cache exactly the four calls the secret store makes', () => {
    // registry/secrets/secrets_manager/store.py calls get, put, create and
    // delete, and nothing else. Terraform's OBO policy grants the same four.
    const policies = template.findResources('AWS::IAM::ManagedPolicy');
    const cachePolicy = Object.values(policies).find((p) =>
      JSON.stringify(p).includes('OboCacheVaultCrud'),
    );
    const statements = cachePolicy?.Properties.PolicyDocument.Statement as Array<{
      Sid: string;
      Action: string[];
    }>;
    const crud = statements.find((s) => s.Sid === 'OboCacheVaultCrud');
    expect([...(crud?.Action ?? [])].sort()).toEqual([
      'secretsmanager:CreateSecret',
      'secretsmanager:DeleteSecret',
      'secretsmanager:GetSecretValue',
      'secretsmanager:PutSecretValue',
    ]);
  });
});

describe('cognito and pingfederate', () => {
  it('reports both providers off by default', () => {
    const { template } = buildStacks(testConfig());
    const env = envOf(container(template, 'auth-server'));
    expect(env.COGNITO_ENABLED).toEqual('false');
    expect(env.PINGFEDERATE_ENABLED).toEqual('false');
  });

  it('wires Cognito into both services and scopes the IAM read to one pool', () => {
    const config = testConfig();
    config.cognito = {
      enabled: true,
      userPoolId: 'us-west-2_AbC123xyz',
      clientId: 'cognito-client-id',
      clientSecretArn: secretArn('mcp/cognito-client-secret'),
      // A PREFIX, not a hostname: the auth-server appends
      // ".auth.<region>.amazoncognito.com".
      domain: 'my-pool',
      m2mClientIds: 'machine-client-one',
    };
    const { template } = buildStacks(config);

    for (const name of ['registry', 'auth-server']) {
      const env = envOf(container(template, name));
      expect(env.COGNITO_ENABLED).toEqual('true');
      expect(env.COGNITO_USER_POOL_ID).toEqual('us-west-2_AbC123xyz');
      expect(secretsOf(container(template, name))).toHaveProperty('COGNITO_CLIENT_SECRET');
    }

    // Auth-server only: the registry has no use for the M2M allowlist.
    expect(envOf(container(template, 'auth-server')).COGNITO_M2M_CLIENT_IDS).toEqual(
      'machine-client-one',
    );
    expect(envOf(container(template, 'registry'))).not.toHaveProperty('COGNITO_M2M_CLIENT_IDS');

    const policies = JSON.stringify(template.findResources('AWS::IAM::ManagedPolicy'));
    expect(policies).toContain('userpool/us-west-2_AbC123xyz');
    expect(policies).not.toContain('userpool/*');
  });

  function pingConfig(overrides: Record<string, unknown> = {}) {
    const config = testConfig();
    config.pingfederate = {
      ...config.pingfederate,
      enabled: true,
      baseUrl: 'https://ping.internal',
      externalUrl: 'https://sso.example.com',
      clientId: 'ping-client',
      clientSecretArn: secretArn('mcp/ping-client-secret'),
      adminPasswordArn: secretArn('mcp/ping-admin-password'),
      m2mAllowedAudiences: 'api://ping-upstream',
      ...overrides,
    };
    return config;
  }

  it('wires PingFederate into both services', () => {
    const { template } = buildStacks(pingConfig({
      m2mClientId: 'ping-m2m',
      m2mClientSecretArn: secretArn('mcp/ping-m2m-secret'),
    }));

    for (const name of ['registry', 'auth-server']) {
      const env = envOf(container(template, name));
      expect(env.PINGFEDERATE_ENABLED).toEqual('true');
      expect(env.PINGFEDERATE_M2M_ALLOWED_AUDIENCES).toEqual('api://ping-upstream');
      const secrets = secretsOf(container(template, name));
      expect(secrets).toHaveProperty('PINGFEDERATE_CLIENT_SECRET');
      expect(secrets).toHaveProperty('PINGFEDERATE_M2M_CLIENT_SECRET');
    }
  });

  it('gives the registry the admin API settings, and not the auth-server', () => {
    // Without PF_ADMIN_PASS the registry's PingFederate admin client raises,
    // so login works while user administration and service-account creation
    // fail. The auth-server never calls that API.
    const { template } = buildStacks(pingConfig());
    const registry = container(template, 'registry');
    expect(envOf(registry).PF_ADMIN_URL).toEqual('https://pingfederate:9999');
    expect(envOf(registry).PF_ADMIN_USER).toEqual('administrator');
    expect(secretsOf(registry)).toHaveProperty('PF_ADMIN_PASS');

    const auth = container(template, 'auth-server');
    expect(envOf(auth)).not.toHaveProperty('PF_ADMIN_URL');
    expect(secretsOf(auth)).not.toHaveProperty('PF_ADMIN_PASS');
  });

  it('wins the provider race, as it does in Terraform', () => {
    // Terraform evaluates PingFederate first, so the same config has to pick
    // the same provider here.
    const config = pingConfig();
    config.auth0 = { ...config.auth0, enabled: true, domain: 'x.auth0.com', clientId: 'a' };
    const { template } = buildStacks(config);
    expect(envOf(container(template, 'auth-server')).AUTH_PROVIDER).toEqual('pingfederate');
  });

  it('injects no M2M secret when none is configured', () => {
    // The provider falls back to the web client secret, and creating an empty
    // secret would both be rejected by CloudFormation and override a working
    // fallback.
    const { template } = buildStacks(pingConfig());
    for (const name of ['registry', 'auth-server']) {
      const secrets = secretsOf(container(template, name));
      expect(secrets).toHaveProperty('PINGFEDERATE_CLIENT_SECRET');
      expect(secrets).not.toHaveProperty('PINGFEDERATE_M2M_CLIENT_SECRET');
    }
    expect(emptySecretStrings(template)).toEqual([]);
  });
});

describe('no secret value reaches the synthesized template', () => {
  it('creates no SecretString for an ARN-referenced secret', () => {
    // A literal value in the template sits in plaintext in cdk.out and in the
    // deployed stack. These four are referenced, never embedded.
    const config = testConfig();
    config.storageBackend = 'documentdb';
    config.cognito = {
      enabled: true,
      userPoolId: 'us-west-2_AbC123xyz',
      clientId: 'cid',
      clientSecretArn: secretArn('mcp/cognito-client-secret'),
      domain: 'my-pool',
      m2mClientIds: '',
    };
    config.pingfederate = {
      ...config.pingfederate,
      enabled: true,
      baseUrl: 'https://ping.internal',
      clientId: 'pf',
      clientSecretArn: secretArn('mcp/ping-client-secret'),
      m2mClientSecretArn: secretArn('mcp/ping-m2m-secret'),
      adminPasswordArn: secretArn('mcp/ping-admin-password'),
    };
    config.egress = {
      ...config.egress,
      enabled: true,
      credentialEncryptionKeyArn: secretArn('mcp/egress-cred-key'),
    };

    const { template } = buildStacks(config);
    const secretResources = template.findResources('AWS::SecretsManager::Secret');
    const names = Object.keys(secretResources).join(' ');
    for (const id of ['Cognito', 'PingFederate', 'EgressCredential']) {
      expect(names).not.toContain(id);
    }

    // And nothing anywhere carries an empty SecretString, which CloudFormation
    // rejects.
    expect(emptySecretStrings(template)).toEqual([]);
  });
});

describe('operator credentials reach containers as secrets (issue #2063)', () => {
  const SHARED = [
    'AUTH0_MANAGEMENT_API_TOKEN',
    'REGISTRY_API_TOKEN',
    'FEDERATION_STATIC_TOKEN',
    'FEDERATION_ENCRYPTION_KEY',
    'ANS_API_KEY',
    'ANS_API_SECRET',
  ];
  const REGISTRY_ONLY = ['GITHUB_PAT', 'GITHUB_APP_PRIVATE_KEY'];
  const ALL = [...SHARED, ...REGISTRY_ONLY];
  const CANARY = 'canary-credential-value-0123456789';  // pragma: allowlist secret

  function _configWithAllCredentials() {
    const config = testConfig();
    config.auth0 = { ...config.auth0, managementApiToken: CANARY };
    config.staticTokenAuth = { ...config.staticTokenAuth, registryApiToken: CANARY };
    config.federation = { ...config.federation, staticToken: CANARY, encryptionKey: CANARY };
    config.ans = { ...config.ans, apiKey: CANARY, apiSecret: CANARY };
    config.github = { ...config.github, pat: CANARY, appPrivateKey: CANARY };
    return config;
  }

  let template: Template;

  beforeAll(() => {
    template = buildStacks(_configWithAllCredentials()).template;
  });

  it('puts no credential in any container environment', () => {
    for (const def of Object.values(containers(template))) {
      const env = envOf(def);
      for (const name of ALL) {
        expect(env).not.toHaveProperty(name);
      }
      expect(JSON.stringify(def.Environment ?? [])).not.toContain(CANARY);
    }
  });

  it('injects the shared credentials into both services', () => {
    for (const service of ['registry', 'auth-server']) {
      const secrets = secretsOf(container(template, service));
      for (const name of SHARED) {
        expect(secrets).toHaveProperty(name);
      }
    }
  });

  it('injects the GitHub credentials into the registry only', () => {
    const registry = secretsOf(container(template, 'registry'));
    const auth = secretsOf(container(template, 'auth-server'));
    for (const name of REGISTRY_ONLY) {
      expect(registry).toHaveProperty(name);
      expect(auth).not.toHaveProperty(name);
    }
  });

  it('denies the auth-server execution role the GitHub credentials', () => {
    const registry = secretsOf(container(template, 'registry'));
    const githubIds = REGISTRY_ONLY.map(
      (name) => secretIdentity(JSON.parse(String(registry[name]))),
    );
    for (const [logicalId, policy] of Object.entries(
      template.findResources('AWS::IAM::Policy'),
    )) {
      if (!logicalId.toLowerCase().includes('auth')) continue;
      const serialized = JSON.stringify(policy);
      for (const id of githubIds) {
        expect(serialized).not.toContain(id.replace(/^Ref:/, ''));
      }
    }
  });

  it('creates no secret and injects nothing when a credential is unset', () => {
    const { template: unset } = buildStacks(testConfig());
    for (const service of ['registry', 'auth-server']) {
      const def = container(unset, service);
      for (const name of ALL) {
        expect(secretsOf(def)).not.toHaveProperty(name);
        expect(envOf(def)).not.toHaveProperty(name);
      }
    }
    expect(emptySecretStrings(unset)).toEqual([]);
  });
});

describe('mcpgw', () => {
  it('knows the registry external URL', () => {
    const { template, serviceStack } = buildStacks(testConfig());
    // Without it, mcpgw sends no X-Forwarded-Host and the registry builds
    // tool-result URLs pointing at registry:8080.
    expect(envOf(container(template, 'mcpgw-server')).REGISTRY_EXTERNAL_URL).toBeDefined();
    expect(serviceStack.registryUrl).toBeTruthy();
  });

  it('is not allowed to reach the registry app port directly', () => {
    const { template } = buildStacks(testConfig());
    const serialized = JSON.stringify(template.findResources('AWS::EC2::SecurityGroup'))
      + JSON.stringify(template.findResources('AWS::EC2::SecurityGroupIngress'));
    expect(serialized).not.toContain('Port 7860 from mcpgw');
  });
});

describe('per-service ADOT sidecars', () => {
  it('adds an OTLP collector beside the registry, auth-server and mcpgw', () => {
    const config = testConfig();
    config.enableObservability = true;
    const { template } = buildStacks(config);

    const sidecars = Object.entries(containers(template)).filter(
      ([key]) => key.endsWith('/adot-collector'),
    );
    const families = sidecars.map(([key]) => key.split('/')[0]);
    for (const service of ['registry', 'auth-server', 'mcpgw-server']) {
      expect(families.some((family) => family.endsWith(service))).toBe(true);
    }

    // The app pushes to a co-located collector, so the endpoint is localhost
    // and the collector, not the app, signs the AMP write.
    for (const name of ['registry', 'auth-server', 'mcpgw-server']) {
      expect(envOf(container(template, name)).OTEL_EXPORTER_OTLP_ENDPOINT).toEqual(
        'http://localhost:4317',
      );
    }

    // The sidecar's own config must RECEIVE over OTLP. The metrics-service
    // collector scrapes Prometheus instead, so a copy-paste of that config
    // here would collect nothing.
    const otlpSidecar = sidecars.find(([key]) => key.includes('registry'))?.[1];
    const sidecarConfig = JSON.stringify(envOf(otlpSidecar!));
    expect(sidecarConfig).toContain('otlp');
    expect(sidecarConfig).toContain('0.0.0.0:4317');
    expect(sidecarConfig).toContain('prometheusremotewrite');
    expect(sidecarConfig).not.toContain('scrape_configs');
  });

  it('adds no sidecar when observability is off', () => {
    const config = testConfig();
    config.enableObservability = false;
    const { template } = buildStacks(config);
    const sidecars = Object.keys(containers(template)).filter((k) => k.endsWith('/adot-collector'));
    expect(sidecars).toEqual([]);
    expect(envOf(container(template, 'registry'))).not.toHaveProperty(
      'OTEL_EXPORTER_OTLP_ENDPOINT',
    );
  });
});

describe('template hygiene', () => {
  it('creates no secret with an empty SecretString', () => {
    // CloudFormation's minimum length for SecretString is 1, so one of these
    // fails the deploy after a clean synth. The check that used to live here
    // searched the serialized template for '"SecretString": ""', which
    // JSON.stringify never emits, so it passed unconditionally.
    const { template } = buildStacks(testConfig());
    expect(emptySecretStrings(template)).toEqual([]);
  });

  it('grants secret read only to roles whose containers receive secrets', () => {
    // mcpgw injects no container secrets, so an execution-role grant over the
    // whole bundle would let it read the PingFederate admin password and the
    // egress credential key.
    const config = testConfig();
    config.storageBackend = 'documentdb';
    config.pingfederate = {
      ...config.pingfederate,
      enabled: true,
      baseUrl: 'https://ping.internal',
      clientId: 'pf',
      clientSecretArn: secretArn('mcp/ping-client-secret'),
      adminPasswordArn: secretArn('mcp/ping-admin-password'),
    };
    const { template } = buildStacks(config);

    const adminPasswordArn = secretArn('mcp/ping-admin-password');
    for (const [logicalId, policy] of Object.entries(
      template.findResources('AWS::IAM::Policy'),
    )) {
      const serialized = JSON.stringify(policy);
      if (!serialized.includes(adminPasswordArn)) continue;
      // Only the registry's roles may name this secret.
      expect(logicalId.toLowerCase()).toMatch(/registry/);
    }
  });
});

describe('container environment hygiene', () => {
  it('never renders a literal "undefined" value', () => {
    // A missing config field reaches the task as the string "undefined",
    // which the app reads as a real value rather than as absent.
    const { template } = buildStacks(testConfig());
    for (const [key, def] of Object.entries(containers(template))) {
      for (const [name, value] of Object.entries(envOf(def))) {
        expect(`${key} ${name}=${value}`).not.toMatch(/=undefined$/);
      }
    }
  });
});
