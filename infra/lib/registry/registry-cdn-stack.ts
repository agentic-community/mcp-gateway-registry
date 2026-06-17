/**
 * RegistryCdnStack - CloudFront distributions and WAFv2 Web ACLs.
 *
 * Instantiates CloudFrontDistribution and WafRules L3 constructs.
 *
 * This stack depends on RegistryServiceStack (for the registry ALB)
 * and RegistryAuthStack (for the Keycloak ALB).
 */

import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { RegistryConfig } from './registry-config';
import { RegistryServiceStack } from './registry-service-stack';
import { RegistryAuthStack } from './registry-auth-stack';
import { CloudFrontDistribution } from './constructs/cloudfront-distribution';
import { WafRules } from './constructs/waf-rules';

// ---------------------------------------------------------------------------
// Stack props
// ---------------------------------------------------------------------------

export interface RegistryCdnStackProps extends cdk.StackProps {
  readonly config: RegistryConfig;
  readonly serviceStack: RegistryServiceStack;
  readonly authStack: RegistryAuthStack;
}

// ---------------------------------------------------------------------------
// Stack
// ---------------------------------------------------------------------------

export class RegistryCdnStack extends cdk.Stack {
  /** MCP Gateway CloudFront distribution domain name */
  public readonly mcpGatewayDistributionDomain: string;

  /** Keycloak CloudFront distribution domain name */
  public readonly keycloakDistributionDomain: string;

  /** MCP Gateway WAF Web ACL ARN */
  public readonly mcpGatewayWebAclArn: string;

  /** Keycloak WAF Web ACL ARN */
  public readonly keycloakWebAclArn: string;

  constructor(scope: Construct, id: string, props: RegistryCdnStackProps) {
    super(scope, id, props);

    const { config, serviceStack, authStack } = props;

    // ------------------------------------------------------------------
    // CloudFront distributions
    // ------------------------------------------------------------------

    const cf = new CloudFrontDistribution(this, 'CloudFront', {
      config,
      registryAlbDns: serviceStack.registryAlbDns,
      keycloakAlbDns: authStack.keycloakAlbDns,
      hostedZoneDomain: config.baseDomain,
    });

    this.mcpGatewayDistributionDomain = cf.mcpGatewayDistributionDomainName;
    this.keycloakDistributionDomain = cf.keycloakDistributionDomainName;

    // ------------------------------------------------------------------
    // WAF Web ACLs
    // ------------------------------------------------------------------

    const waf = new WafRules(this, 'Waf', {
      config,
      mcpGatewayAlbArn: serviceStack.registryAlbArn,
      keycloakAlbArn: authStack.keycloakDomain !== '' ? authStack.keycloakAlbArn : undefined,
    });

    this.mcpGatewayWebAclArn = waf.mcpGatewayWebAclArn;
    this.keycloakWebAclArn = waf.keycloakWebAclArn;

    // ------------------------------------------------------------------
    // Common tags
    // ------------------------------------------------------------------

    cdk.Tags.of(this).add('Project', 'mcp-gateway-registry');
    cdk.Tags.of(this).add('Component', 'cdn');
    cdk.Tags.of(this).add('Environment', 'production');
    cdk.Tags.of(this).add('ManagedBy', 'cdk');
  }
}
