/**
 * Synth-time validation of the provider and egress config blocks.
 *
 * Each case covers a setting whose wrong value would either crash-loop the
 * task or widen a grant, so it has to fail at synth instead.
 */

import { secretArn, testConfig } from './helpers';
import { DEFAULT_REGISTRY_CONFIG, RegistryConfig } from '../lib/registry/registry-config';

/**
 * `loadRegistryConfig` reads a file, so the validators are exercised through a
 * fresh require of the module with a stubbed YAML read. Simpler: call the
 * loader against a temp file.
 */
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as yaml from 'js-yaml';

import { loadRegistryConfig } from '../lib/registry/registry-config';

/** Load a config from verbatim YAML text, with the required secrets in env. */
function loadRaw(yamlText: string): RegistryConfig {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cdk-config-test-'));
  const file = path.join(dir, 'config.yaml');
  fs.writeFileSync(file, yamlText);
  const base = testConfig();
  const previous = { ...process.env };
  // The loader's own required-secret check reads these; set them here so a
  // raw-YAML fixture does not have to restate them.
  process.env.CDK_KEYCLOAK_ADMIN_PASSWORD = base.keycloak.adminPassword;
  process.env.CDK_KEYCLOAK_DATABASE_PASSWORD = base.keycloak.databasePassword;
  process.env.CDK_DOCUMENTDB_ADMIN_PASSWORD = base.documentdb.adminPassword;
  process.env.CDK_GRAFANA_ADMIN_PASSWORD = base.grafanaAdminPassword;
  try {
    return loadRegistryConfig(file);
  } finally {
    process.env = previous;
  }
}

function loadWith(overrides: Record<string, unknown>): RegistryConfig {
  const base = testConfig();
  return loadRaw(
    yaml.dump({
      registry: {
        awsAccountId: base.awsAccountId,
        enableRoute53Dns: false,
        cloudfront: { enabled: true },
        ...overrides,
      },
    }),
  );
}

describe('YAML keys written with no value', () => {
  // `oboAllowedAudiences:` with nothing after it parses as null. It is one
  // keystroke from a commented-out example, and assigning null either crashed
  // a string method during validation or reached a container as the literal
  // `Value: null`. The default has to win instead.
  it('keeps the default rather than taking null', () => {
    const config = loadRaw(
      'registry:\n  egress:\n    oboAllowedAudiences:\n    oauthTrustedIdpHosts:\n'
      + '    secretsManagerPathPrefix:\n',
    );
    expect(config.egress.oboAllowedAudiences).toEqual('');
    expect(config.egress.oauthTrustedIdpHosts).toEqual('');
    expect(config.egress.secretsManagerPathPrefix).toEqual('mcp/egress');
  });

  it('does not crash validation on a null string field', () => {
    expect(() =>
      loadRaw('registry:\n  trustedRealIpCidrs:\n  egress:\n    secretStoreBackend:\n'),
    ).not.toThrow();
  });

  it('still honors an explicitly empty value', () => {
    const config = loadRaw('registry:\n  egress:\n    oboAllowedAudiences: ""\n');
    expect(config.egress.oboAllowedAudiences).toEqual('');
  });
});

describe('defaults', () => {
  it('leaves every new provider and egress block off', () => {
    expect(DEFAULT_REGISTRY_CONFIG.cognito.enabled).toBe(false);
    expect(DEFAULT_REGISTRY_CONFIG.pingfederate.enabled).toBe(false);
    expect(DEFAULT_REGISTRY_CONFIG.egress.enabled).toBe(false);
    expect(DEFAULT_REGISTRY_CONFIG.trustedRealIpCidrs).toEqual('');
  });

  it('loads a default config without throwing', () => {
    expect(() => loadWith({})).not.toThrow();
  });
});

describe('values that reach an IAM policy Resource', () => {
  // One wildcard in any of these positions widens the grant silently, so each
  // field gets the same hostile inputs. A `*` in a secret name is the worst
  // case: every Secrets Manager ARN ends in a dash plus six characters, so
  // `secret:*-AbCdEf` as a Resource matches every secret in the account.
  //
  // Keep this list matched to the one above `_requireIamResourceSegment` in
  // registry-config.ts.
  const HOSTILE = ['*', '?', 'a:b', 'has space'];

  const validCognito = {
    enabled: true,
    userPoolId: 'us-west-2_AbC123xyz',
    clientId: 'client',
    clientSecretArn: secretArn('mcp/cognito'),
    domain: 'my-pool',
  };
  const validPing = {
    enabled: true,
    baseUrl: 'https://ping.internal',
    clientId: 'c',
    clientSecretArn: secretArn('mcp/ping'),
    adminPasswordArn: secretArn('mcp/ping-admin'),
  };

  /** Each secret-ARN field, with the config that makes it reachable. */
  const SECRET_ARN_FIELDS: Array<[string, (v: string) => Record<string, unknown>]> = [
    ['cognito.clientSecretArn',
      (v) => ({ cognito: { ...validCognito, clientSecretArn: v } })],
    ['pingfederate.clientSecretArn',
      (v) => ({ pingfederate: { ...validPing, clientSecretArn: v } })],
    ['pingfederate.m2mClientSecretArn',
      (v) => ({ pingfederate: { ...validPing, m2mClientSecretArn: v } })],
    ['pingfederate.adminPasswordArn',
      (v) => ({ pingfederate: { ...validPing, adminPasswordArn: v } })],
    ['egress.credentialEncryptionKeyArn',
      (v) => ({
        storageBackend: 'documentdb',
        egress: { enabled: true, credentialEncryptionKeyArn: v },
      })],
  ];

  describe.each(SECRET_ARN_FIELDS)('%s', (_name, build) => {
    it.each(HOSTILE)('rejects a secret name of %p', (hostile) => {
      const arn = `arn:aws:secretsmanager:us-west-2:123456789012:secret:${hostile}-AbCdEf`;
      expect(() => loadWith(build(arn))).toThrow();
    });

    it('accepts a well-formed ARN', () => {
      expect(() => loadWith(build(secretArn('mcp/ok')))).not.toThrow();
    });
  });

  it.each(HOSTILE)('rejects a Cognito user pool id of %p', (hostile) => {
    expect(() => loadWith({ cognito: { ...validCognito, userPoolId: hostile } })).toThrow();
  });

  it.each(HOSTILE)('rejects an egress path prefix of %p', (hostile) => {
    expect(() =>
      loadWith({
        storageBackend: 'documentdb',
        egress: { enabled: true, secretsManagerPathPrefix: hostile },
      }),
    ).toThrow();
  });

  it.each(HOSTILE)('rejects a KMS key ARN of %p', (hostile) => {
    expect(() =>
      loadWith({
        storageBackend: 'documentdb',
        egress: { enabled: true, secretsManagerKmsKeyArn: hostile },
      }),
    ).toThrow();
  });

  it('names the wildcard rather than the field format', () => {
    // The operator should learn WHY, not just that a regex did not match.
    expect(() =>
      loadWith({
        storageBackend: 'documentdb',
        egress: { enabled: true, secretsManagerPathPrefix: 'mcp/*' },  // pragma: allowlist secret
      }),
    ).toThrow(/must not contain "\*"/);
  });

  it.each(['/mcp/egress', 'mcp/egress/', 'mcp//egress', '/'])(
    'rejects the edge-slash prefix %p',
    (prefix) => {
      // "secret:<prefix>/*" would carry a doubled or leading separator that no
      // secret the application writes can match, so every vault operation
      // would fail with AccessDenied.
      expect(() =>
        loadWith({
          storageBackend: 'documentdb',
          egress: { enabled: true, secretsManagerPathPrefix: prefix },
        }),
      ).toThrow(/must not start or end with "\/"|must be a Secrets Manager name prefix/);
    },
  );

  it('accepts the default prefix', () => {
    const config = loadWith({
      storageBackend: 'documentdb',
      egress: { enabled: true, secretsManagerPathPrefix: 'mcp/egress' },  // pragma: allowlist secret
    });
    expect(config.egress.secretsManagerPathPrefix).toEqual('mcp/egress');
  });
});

describe('the application contract this config has to match', () => {
  // These are the values the running services accept. Each failure here means
  // CDK would pass something the application rejects at startup, which is the
  // shape of defect that template assertions cannot see.
  it('offers only storage backends the registry treats as Mongo-family', () => {
    // registry/core/config.py, MONGODB_BACKENDS
    const config = loadWith({
      storageBackend: 'documentdb',
      egress: { enabled: true, credentialEncryptionKeyArn: secretArn('k') },
    });
    expect(config.storageBackend).toEqual('documentdb');
    for (const backend of ['file', 'sqlite', 'mongo']) {
      expect(() => loadWith({ storageBackend: backend, egress: { enabled: true } }))
        .toThrow(/Mongo-family storageBackend/);
    }
  });

  it('offers only secret-store backends the registry accepts', () => {
    // registry/core/config.py, secret_store_backend: "secrets-manager, openbao".  // pragma: allowlist secret
    // openbao is excluded because this stack provisions none.
    expect(loadWith({}).egress.secretStoreBackend).toEqual('secrets-manager');
    for (const backend of ['openbao', 'documentdb', 'vault']) {
      expect(() => loadWith({ egress: { secretStoreBackend: backend } }))
        .toThrow(/secretStoreBackend must be one of/);
    }
  });

  it('treats the Cognito domain as a prefix, as the auth-server does', () => {
    // auth_server/providers/cognito.py appends .auth.<region>.amazoncognito.com
    expect(() =>
      loadWith({
        cognito: {
          enabled: true,
          userPoolId: 'us-west-2_AbC123xyz',
          clientId: 'c',
          clientSecretArn: secretArn('s'),
          domain: 'my-pool.auth.us-west-2.amazoncognito.com',
        },
      }),
    ).toThrow(/must be the hosted-UI domain PREFIX/);
  });

  it('rejects a comma-separated OBO audience list', () => {
    // registry/core/schemas.py splits on whitespace, so commas match nothing.
    expect(() =>
      loadWith({
        storageBackend: 'documentdb',
        egress: {
          enabled: true,
          credentialEncryptionKeyArn: secretArn('k'),
          oboAllowedAudiences: 'api://one,api://two',
        },
      }),
    ).toThrow(/whitespace-separated, not comma-separated/);
  });
});

describe('cognito validation', () => {
  const validCognito = {
    enabled: true,
    userPoolId: 'us-west-2_AbC123xyz',
    clientId: 'client',
    clientSecretArn: secretArn('mcp/cognito'),
    domain: 'my-pool',
  };

  it('rejects an incomplete config', () => {
    expect(() => loadWith({ cognito: { enabled: true, userPoolId: 'us-west-2_AbC123' } }))
      .toThrow(/cognito.clientId/);
  });

  it('rejects a malformed user pool id', () => {
    // A wildcard is caught earlier by the shared IAM-segment guard and
    // covered in its own block; this is the plain format check.
    expect(() => loadWith({ cognito: { ...validCognito, userPoolId: 'not-a-pool-id' } }))
      .toThrow(/must be a Cognito user pool id/);
  });

  it('requires a complete secret ARN, not a value', () => {
    expect(() => loadWith({ cognito: { ...validCognito, clientSecretArn: 'my-secret-value' } }))  // pragma: allowlist secret
      .toThrow(/complete Secrets Manager secret ARN/);
    // A partial ARN would read the wrong secret rather than failing.
    expect(() =>
      loadWith({
        cognito: {
          ...validCognito,
          clientSecretArn: 'arn:aws:secretsmanager:us-west-2:123456789012:secret:mcp/cognito',  // pragma: allowlist secret
        },
      }),
    ).toThrow(/complete Secrets Manager secret ARN/);
  });

  it('accepts an empty domain, which the auth-server derives from the pool id', () => {
    const config = loadWith({ cognito: { ...validCognito, domain: '' } });
    expect(config.cognito.enabled).toBe(true);
    expect(config.cognito.domain).toEqual('');
  });

  it('accepts a well-formed config', () => {
    const config = loadWith({ cognito: { ...validCognito, domain: 'my-pool' } });
    expect(config.cognito.clientSecretArn).toEqual(secretArn('mcp/cognito'));
  });
});

describe('pingfederate validation', () => {
  const validPing = {
    enabled: true,
    baseUrl: 'https://ping.internal',
    clientId: 'c',
    clientSecretArn: secretArn('mcp/ping'),
    adminPasswordArn: secretArn('mcp/ping-admin'),
  };

  it('accepts an M2M client with no extra audience allowlist', () => {
    // The provider always validates aud against the client ids and
    // applicationIdUri, so an empty extra list is already fail-closed and
    // Terraform permits it.
    const config = loadWith({
      pingfederate: { ...validPing, m2mClientId: 'm', m2mAllowedAudiences: '' },
    });
    expect(config.pingfederate.m2mClientId).toEqual('m');
  });

  it('requires the admin password, which the registry raises without', () => {
    expect(() => loadWith({ pingfederate: { ...validPing, adminPasswordArn: '' } }))
      .toThrow(/pingfederate.adminPasswordArn/);
  });

  it('rejects an empty groups claim', () => {
    expect(() => loadWith({ pingfederate: { ...validPing, groupsClaim: '' } }))
      .toThrow(/groupsClaim/);
  });

  it('accepts a config with no separate M2M secret', () => {
    // The provider falls back to the web client secret, so this is valid and
    // must not synthesize an empty secret.
    const config = loadWith({ pingfederate: validPing });
    expect(config.pingfederate.m2mClientSecretArn).toEqual('');
  });

  it('validates an M2M secret ARN when one is given', () => {
    expect(() => loadWith({ pingfederate: { ...validPing, m2mClientSecretArn: 'not-an-arn' } }))  // pragma: allowlist secret
      .toThrow(/complete Secrets Manager secret ARN/);
  });
});

describe('egress validation', () => {
  const validEgress = {
    enabled: true,
    oboAllowedAudiences: 'api://upstream',
    credentialEncryptionKeyArn: secretArn('mcp/egress-cred-key'),
  };

  function loadEgress(overrides: Record<string, unknown> = {}) {
    return loadWith({ storageBackend: 'documentdb', egress: { ...validEgress, ...overrides } });
  }

  it('rejects a non-Mongo storage backend', () => {
    expect(() => loadWith({ storageBackend: 'file', egress: validEgress }))
      .toThrow(/Mongo-family storageBackend/);
  });

  it('accepts an empty OBO audience allowlist', () => {
    // Empty is not "allow anything": the application falls back to a shape
    // rule that rejects every https:// host URL. Requiring it here would
    // block a configuration Terraform and the application both support.
    const config = loadEgress({ oboAllowedAudiences: '' });
    expect(config.egress.oboAllowedAudiences).toEqual('');
  });

  it('requires an ARN for the encryption key, not the key material', () => {
    expect(() => loadEgress({ credentialEncryptionKeyArn: 'a-literal-32-byte-key-value-here!' }))
      .toThrow(/complete Secrets Manager secret ARN/);
  });

  it('never echoes a value passed where the ARN belongs', () => {
    const value = 'definitely-not-an-arn-but-secret-looking';
    try {
      loadEgress({ credentialEncryptionKeyArn: value });
      throw new Error('expected a validation error');
    } catch (err) {
      expect(String(err)).not.toContain(value);
    }
  });

  it('accepts an empty encryption key ARN', () => {
    // Legacy behavior: credentials are stored in plaintext. Supported by the
    // application and by Terraform, so synth must not block it.
    const config = loadEgress({ credentialEncryptionKeyArn: '' });
    expect(config.egress.credentialEncryptionKeyArn).toEqual('');
  });

  it('rejects a path prefix that could widen the vault grant', () => {
    expect(() => loadEgress({ secretsManagerPathPrefix: 'mcp/*' }))  // pragma: allowlist secret
      .toThrow(/secretsManagerPathPrefix/);
  });

  it('rejects any KMS value that is not a key ARN', () => {
    // IAM accepts only a key ARN in the Resource element of a statement
    // granting cryptographic operations, so each of these would grant
    // nothing (or, for the wildcard, far too much).
    for (const value of [
      '*',
      '11111111-2222-3333-4444-555555555555',
      'alias/mcp-egress-vault',
      // An alias ARN is valid only in a statement governing alias
      // operations. Used here it silently grants no decrypt at all, so every
      // vault write would fail with AccessDenied.
      'arn:aws:kms:us-west-2:123456789012:alias/mcp-egress-vault',
    ]) {
      expect(() => loadEgress({ secretsManagerKmsKeyArn: value }))
        .toThrow(/must be a KMS KEY ARN/);
    }
  });

  it('accepts a KMS key ARN', () => {
    const arn = 'arn:aws:kms:us-west-2:123456789012:key/11111111-2222-3333-4444-555555555555';
    expect(loadEgress({ secretsManagerKmsKeyArn: arn }).egress.secretsManagerKmsKeyArn)
      .toEqual(arn);
  });

  it('rejects the OBO cache without egress', () => {
    expect(() => loadWith({ egress: { enabled: false, oboCacheEnabled: true } }))
      .toThrow(/oboCacheEnabled=true needs egress.enabled=true/);
  });

  it('rejects an expiry skew at or above the max TTL', () => {
    expect(() =>
      loadEgress({
        oboCacheEnabled: true,
        oboCacheMaxTtlSeconds: 60,
        oboCacheExpirySkewSeconds: 60,
      }),
    ).toThrow(/born expired/);
  });

  it('accepts a complete config', () => {
    const config = loadEgress();
    expect(config.egress.enabled).toBe(true);
    expect(config.egress.registryInternalUrl).toEqual('http://registry:8091');
  });
});
