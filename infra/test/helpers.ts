/**
 * Shared helpers for the CDK stack tests.
 *
 * Builds the real stack graph from DEFAULT_REGISTRY_CONFIG so a test asserts
 * against the template that would deploy, not against a hand-written fixture.
 */

import * as cdk from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';

import { DEFAULT_REGISTRY_CONFIG, RegistryConfig } from '../lib/registry/registry-config';
import { RegistryNetworkStack } from '../lib/registry/registry-network-stack';
import { RegistryDataStack } from '../lib/registry/registry-data-stack';
import { RegistryAuthStack } from '../lib/registry/registry-auth-stack';
import { RegistryServiceStack } from '../lib/registry/registry-service-stack';

/** Placeholder secrets. Long enough to clear the synth-time length checks. */
const TEST_SECRET = 'test-only-not-a-real-secret-0123456789abcdef';  // pragma: allowlist secret

/**
 * Build a complete Secrets Manager ARN for a pre-existing secret.
 *
 * `Secret.fromSecretCompleteArn` requires the six-character suffix the
 * service appends, so a test fixture has to carry one too.
 */
export function secretArn(name: string, suffix = 'AbCdEf'): string {
  return `arn:aws:secretsmanager:us-west-2:123456789012:secret:${name}-${suffix}`;
}

export interface BuiltStacks {
  readonly app: cdk.App;
  readonly serviceStack: RegistryServiceStack;
  readonly template: Template;
}

/**
 * A config that passes every synth-time check, with CloudFront as the HTTPS
 * front so no Route53 hosted-zone lookup is needed.
 */
export function testConfig(overrides: Partial<RegistryConfig> = {}): RegistryConfig {
  const base: RegistryConfig = JSON.parse(JSON.stringify(DEFAULT_REGISTRY_CONFIG));
  base.awsAccountId = '123456789012';
  base.awsRegion = 'us-west-2';
  base.enableRoute53Dns = false;
  base.cloudfront = { ...base.cloudfront, enabled: true };
  base.keycloak = { ...base.keycloak, adminPassword: TEST_SECRET, databasePassword: TEST_SECRET };
  base.documentdb = { ...base.documentdb, adminPassword: TEST_SECRET };
  // Required once observability is on, which the default config enables.
  base.grafanaAdminPassword = TEST_SECRET;
  // The optional services are skipped entirely when their image is empty, so
  // set the ones these tests assert on.
  base.images = {
    ...base.images,
    registry: 'example.invalid/registry:test',
    mcpgw: 'example.invalid/mcpgw:test',
    metricsService: 'example.invalid/metrics:test',
    grafana: 'example.invalid/grafana:test',
  };
  return { ...base, ...overrides };
}

/** Synthesize the service stack (and the stacks it depends on) for a config. */
export function buildStacks(config: RegistryConfig): BuiltStacks {
  const app = new cdk.App();
  const env: cdk.Environment = { account: config.awsAccountId, region: config.awsRegion };

  const networkStack = new RegistryNetworkStack(app, 'Registry-Network', { config, env });
  const dataStack = new RegistryDataStack(app, 'Registry-Data', { config, networkStack, env });
  const authStack = new RegistryAuthStack(app, 'Registry-Auth', {
    config,
    vpc: networkStack.vpc,
    privateSubnets: networkStack.privateSubnets,
    publicSubnets: networkStack.publicSubnets,
    keycloakDbSg: dataStack.keycloakDbSg,
    rdsKmsKey: dataStack.rdsKmsKey,
    keycloakDbSecretArn: dataStack.keycloakDbSecret.secretArn,
    env,
  });
  const serviceStack = new RegistryServiceStack(app, 'Registry-Service', {
    config,
    networkStack,
    dataStack,
    authStack,
    env,
  });

  return { app, serviceStack, template: Template.fromStack(serviceStack) };
}

export interface ContainerDef {
  Name: string;
  Environment?: Array<{ Name: string; Value: unknown }>;
  Secrets?: Array<{ Name: string; ValueFrom: unknown }>;
  PortMappings?: Array<{ ContainerPort: number; Name?: string }>;
  HealthCheck?: { Command: string[] };
  DependsOn?: Array<{ ContainerName: string; Condition: string }>;
}

/**
 * Every container in the template, keyed by `<task-definition-family>/<container-name>`.
 *
 * Reading the synthesized template rather than the construct tree is the point:
 * a value that a config branch silently dropped shows up as missing here.
 */
export function containers(template: Template): Record<string, ContainerDef> {
  const taskDefs = template.findResources('AWS::ECS::TaskDefinition');
  const out: Record<string, ContainerDef> = {};
  for (const resource of Object.values(taskDefs)) {
    const props = (resource as { Properties: Record<string, unknown> }).Properties;
    const family = String(props.Family ?? 'unknown');
    for (const container of (props.ContainerDefinitions ?? []) as ContainerDef[]) {
      out[`${family}/${container.Name}`] = container;
    }
  }
  return out;
}

/** The named container, or a failure naming what was found instead. */
export function container(template: Template, suffix: string): ContainerDef {
  const all = containers(template);
  const matches = Object.keys(all).filter((key) => key.endsWith(`/${suffix}`));
  if (matches.length !== 1) {
    throw new Error(
      `expected exactly one container named '${suffix}', found ${matches.length}: ` +
      `${matches.join(', ')} (all: ${Object.keys(all).join(', ')})`,
    );
  }
  return all[matches[0]];
}

/** Environment variables of a container as a plain map. */
export function envOf(def: ContainerDef): Record<string, unknown> {
  return Object.fromEntries((def.Environment ?? []).map((e) => [e.Name, e.Value]));
}

/** Secret references of a container as a plain map of name to ValueFrom. */
export function secretsOf(def: ContainerDef): Record<string, unknown> {
  return Object.fromEntries((def.Secrets ?? []).map((s) => [s.Name, JSON.stringify(s.ValueFrom)]));
}

/**
 * The CloudFormation identity a secret reference resolves to.
 *
 * A container's `ValueFrom` is a `Ref` (or a `Fn::ImportValue`, or a
 * `Fn::Join` wrapping one to append a JSON key). The matching IAM grant names
 * the same secret, so comparing the underlying logical id or export name is
 * what links the two without string surgery on the ARN.
 */
export function secretIdentity(valueFrom: unknown): string {
  if (typeof valueFrom === 'string') return valueFrom;
  if (valueFrom === null || typeof valueFrom !== 'object') return String(valueFrom);
  const obj = valueFrom as Record<string, unknown>;
  if ('Ref' in obj) return `Ref:${obj.Ref}`;
  if ('Fn::ImportValue' in obj) return `ImportValue:${obj['Fn::ImportValue']}`;
  if ('Fn::Join' in obj) {
    const parts = (obj['Fn::Join'] as [string, unknown[]])[1];
    for (const part of parts) {
      const nested = secretIdentity(part);
      if (nested.startsWith('Ref:') || nested.startsWith('ImportValue:')) return nested;
    }
  }
  return JSON.stringify(valueFrom);
}

/**
 * Logical ids of every `AWS::SecretsManager::Secret` whose `SecretString` is
 * empty.
 *
 * Reads the parsed template rather than its serialized form. An earlier
 * version of this check did `JSON.stringify(template).includes('"SecretString": ""')`,
 * which can never match: `JSON.stringify` emits no space after the colon, so
 * the assertion passed unconditionally while an empty secret sat in the
 * template. CloudFormation's minimum length for the property is 1.
 */
export function emptySecretStrings(template: Template): string[] {
  const out: string[] = [];
  for (const [logicalId, resource] of Object.entries(
    template.findResources('AWS::SecretsManager::Secret'),
  )) {
    const props = (resource as { Properties?: Record<string, unknown> }).Properties ?? {};
    if (props.SecretString === '') out.push(logicalId);
  }
  return out;
}

/**
 * Secret identities that at least one IAM policy in the template grants
 * `secretsmanager:GetSecretValue` on.
 */
export function grantedSecretArns(template: Template): Set<string> {
  const out = new Set<string>();
  for (const type of ['AWS::IAM::Policy', 'AWS::IAM::ManagedPolicy']) {
    for (const resource of Object.values(template.findResources(type))) {
      const props = (resource as { Properties: Record<string, any> }).Properties;
      for (const statement of props.PolicyDocument?.Statement ?? []) {
        const actions = [statement.Action].flat();
        if (!actions.includes('secretsmanager:GetSecretValue')) continue;
        for (const res of [statement.Resource].flat()) {
          out.add(secretIdentity(res));
        }
      }
    }
  }
  return out;
}
