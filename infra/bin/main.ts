#!/usr/bin/env node

/**
 * CDK entry point for MCP Gateway Registry infrastructure.
 *
 * Reads configuration from config.yaml (with environment variable overrides
 * for sensitive values) and instantiates the appropriate stacks.
 */

import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import * as path from 'path';
import { loadRegistryConfig } from '../lib/registry/registry-config';
import { RegistryNetworkStack } from '../lib/registry/registry-network-stack';
import { RegistryDataStack } from '../lib/registry/registry-data-stack';
import { RegistryAuthStack } from '../lib/registry/registry-auth-stack';
import { RegistryServiceStack } from '../lib/registry/registry-service-stack';
import { RegistryOpsStack } from '../lib/registry/registry-ops-stack';
import { RegistryCdnStack } from '../lib/registry/registry-cdn-stack';
import { RegistryBuildStack } from '../lib/registry/registry-build-stack';

// ---------------------------------------------------------------------------
// Load configuration
// ---------------------------------------------------------------------------

const configPath = path.join(__dirname, '..', 'config.yaml');
const config = loadRegistryConfig(configPath);

// ---------------------------------------------------------------------------
// CDK App
// ---------------------------------------------------------------------------

const app = new cdk.App();

if (config.enabled) {
  const env: cdk.Environment = {
    account: config.awsAccountId || process.env.CDK_DEFAULT_ACCOUNT,
    region: config.awsRegion || process.env.CDK_DEFAULT_REGION,
  };

  const networkStack = new RegistryNetworkStack(app, 'Registry-Network', {
    config,
    env,
    description: 'MCP Gateway Registry - VPC, subnets, NAT gateways, and VPC endpoints',
  });

  const dataStack = new RegistryDataStack(app, 'Registry-Data', {
    config,
    networkStack,
    env,
    description: 'MCP Gateway Registry - DocumentDB + Aurora MySQL for Keycloak',
  });

  const authStack = new RegistryAuthStack(app, 'Registry-Auth', {
    config,
    vpc: networkStack.vpc,
    privateSubnets: networkStack.privateSubnets,
    publicSubnets: networkStack.publicSubnets,
    keycloakDbSg: dataStack.keycloakDbSg,
    rdsKmsKey: dataStack.rdsKmsKey,
    keycloakDbSecretArn: dataStack.keycloakDbSecret.secretArn,
    env,
    description: 'MCP Gateway Registry - Keycloak ECS + ALB + DNS',
  });
  authStack.addDependency(networkStack);
  authStack.addDependency(dataStack);

  // Ops must deploy before Auth so that secret rotation completes before
  // Keycloak starts — Keycloak reads DB creds from Secrets Manager and the
  // rotation Lambda updates both SM and Aurora on first deploy.
  const opsStack = new RegistryOpsStack(app, 'Registry-Ops', {
    config,
    networkStack,
    dataStack,
    env,
    description: 'MCP Gateway Registry - Secret rotation Lambdas for DocumentDB and RDS',
  });
  opsStack.addDependency(networkStack);
  opsStack.addDependency(dataStack);

  authStack.addDependency(opsStack);

  const serviceStack = new RegistryServiceStack(app, 'Registry-Service', {
    config,
    networkStack,
    dataStack,
    authStack,
    env,
    description: 'MCP Gateway Registry - Core ECS services, ALB, EFS, and Secrets',
  });
  serviceStack.addDependency(networkStack);
  serviceStack.addDependency(dataStack);
  serviceStack.addDependency(authStack);

  const cdnStack = new RegistryCdnStack(app, 'Registry-Cdn', {
    config,
    serviceStack,
    authStack,
    env,
    description: 'MCP Gateway Registry - CloudFront distributions + WAFv2 Web ACLs',
  });
  cdnStack.addDependency(serviceStack);
  cdnStack.addDependency(authStack);

  const buildStack = new RegistryBuildStack(app, 'Registry-Build', {
    config,
    env,
    description: 'MCP Gateway Registry - ECR repositories + CodeBuild pipeline',
  });
}

app.synth();
