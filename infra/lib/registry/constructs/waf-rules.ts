/**
 * WafRules - L3 construct that creates WAFv2 Web ACLs for MCP Gateway
 * and Keycloak ALBs with managed rule groups and rate limiting.
 *
 * Translated from: terraform/aws-ecs/waf.tf
 *
 * Each Web ACL contains 4 rules:
 *   1. AWSManagedRulesCommonRuleSet (priority 1)
 *   2. AWSManagedRulesKnownBadInputsRuleSet (priority 2)
 *   3. IP-based rate limit: 100 requests / 5 minutes per IP (priority 3)
 *   4. Global rate limit: 2000 requests / 5 minutes (priority 4)
 *
 * This construct is a no-op when config.enableWaf is false.
 */

import * as cdk from 'aws-cdk-lib';
import * as wafv2 from 'aws-cdk-lib/aws-wafv2';
import * as logs from 'aws-cdk-lib/aws-logs';
import { Construct } from 'constructs';
import { RegistryConfig } from '../registry-config';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const IP_RATE_LIMIT = 100;
const GLOBAL_RATE_LIMIT = 2000;
const WAF_LOG_RETENTION_DAYS = 30;

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface WafRulesProps {
  readonly config: RegistryConfig;
  /** ARN of the MCP Gateway ALB to associate with WAF */
  readonly mcpGatewayAlbArn: string;
  /** ARN of the Keycloak ALB to associate with WAF (optional) */
  readonly keycloakAlbArn?: string;
}

// ---------------------------------------------------------------------------
// Construct
// ---------------------------------------------------------------------------

export class WafRules extends Construct {
  /** ARN of the MCP Gateway WAF Web ACL */
  public readonly mcpGatewayWebAclArn: string;

  /** ARN of the Keycloak WAF Web ACL */
  public readonly keycloakWebAclArn: string;

  constructor(scope: Construct, id: string, props: WafRulesProps) {
    super(scope, id);

    const { config, mcpGatewayAlbArn, keycloakAlbArn } = props;

    // No-op when WAF is disabled
    if (!config.enableWaf) {
      this.mcpGatewayWebAclArn = '';
      this.keycloakWebAclArn = '';
      return;
    }

    // ------------------------------------------------------------------
    // MCP Gateway WAF
    // ------------------------------------------------------------------

    const mcpGatewayWaf = _createWebAcl(
      this,
      'McpGateway',
      config,
      `${config.name}-mcp-gateway-waf`,
      'WAF protection for MCP Gateway ALB',
    );

    _createWebAclAssociation(this, 'McpGatewayAssoc', mcpGatewayWaf, mcpGatewayAlbArn);
    _createWafLogging(this, 'McpGatewayLogs', config, mcpGatewayWaf, 'mcp-gateway');

    this.mcpGatewayWebAclArn = mcpGatewayWaf.attrArn;

    // ------------------------------------------------------------------
    // Keycloak WAF
    // ------------------------------------------------------------------

    if (keycloakAlbArn) {
      const keycloakWaf = _createWebAcl(
        this,
        'Keycloak',
        config,
        `${config.name}-keycloak-waf`,
        'WAF protection for Keycloak ALB',
      );

      _createWebAclAssociation(this, 'KeycloakAssoc', keycloakWaf, keycloakAlbArn);
      _createWafLogging(this, 'KeycloakLogs', config, keycloakWaf, 'keycloak');

      this.keycloakWebAclArn = keycloakWaf.attrArn;
    } else {
      this.keycloakWebAclArn = '';
    }
  }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/**
 * Build the standard array of 4 WAF rules used by both MCP Gateway and Keycloak ACLs.
 */
function _buildWafRules(): wafv2.CfnWebACL.RuleProperty[] {
  return [
    // Rule 1: AWS Managed Rules - Common Rule Set
    {
      name: 'AWSManagedRulesCommonRuleSet',
      priority: 1,
      overrideAction: { none: {} },
      statement: {
        managedRuleGroupStatement: {
          name: 'AWSManagedRulesCommonRuleSet',
          vendorName: 'AWS',
        },
      },
      visibilityConfig: {
        cloudWatchMetricsEnabled: true,
        metricName: 'AWSManagedRulesCommonRuleSetMetric',
        sampledRequestsEnabled: true,
      },
    },
    // Rule 2: AWS Managed Rules - Known Bad Inputs
    {
      name: 'AWSManagedRulesKnownBadInputsRuleSet',
      priority: 2,
      overrideAction: { none: {} },
      statement: {
        managedRuleGroupStatement: {
          name: 'AWSManagedRulesKnownBadInputsRuleSet',
          vendorName: 'AWS',
        },
      },
      visibilityConfig: {
        cloudWatchMetricsEnabled: true,
        metricName: 'AWSManagedRulesKnownBadInputsRuleSetMetric',
        sampledRequestsEnabled: true,
      },
    },
    // Rule 3: IP-based rate limiting (100 req / 5 min per IP)
    {
      name: 'IPRateLimitRule',
      priority: 3,
      action: { block: {} },
      statement: {
        rateBasedStatement: {
          limit: IP_RATE_LIMIT,
          aggregateKeyType: 'IP',
        },
      },
      visibilityConfig: {
        cloudWatchMetricsEnabled: true,
        metricName: 'IPRateLimitRuleMetric',
        sampledRequestsEnabled: true,
      },
    },
    // Rule 4: Global rate limiting (2000 req / 5 min total)
    {
      name: 'GlobalRateLimitRule',
      priority: 4,
      action: { block: {} },
      statement: {
        rateBasedStatement: {
          limit: GLOBAL_RATE_LIMIT,
          aggregateKeyType: 'CONSTANT',
        },
      },
      visibilityConfig: {
        cloudWatchMetricsEnabled: true,
        metricName: 'GlobalRateLimitRuleMetric',
        sampledRequestsEnabled: true,
      },
    },
  ];
}

/**
 * Create a WAFv2 Web ACL with the standard 4-rule set.
 */
function _createWebAcl(
  scope: Construct,
  id: string,
  config: RegistryConfig,
  aclName: string,
  purpose: string,
): wafv2.CfnWebACL {
  const webAcl = new wafv2.CfnWebACL(scope, `${id}WebAcl`, {
    name: aclName,
    scope: 'REGIONAL',
    defaultAction: { allow: {} },
    rules: _buildWafRules(),
    visibilityConfig: {
      cloudWatchMetricsEnabled: true,
      metricName: aclName,
      sampledRequestsEnabled: true,
    },
    tags: [
      { key: 'Purpose', value: purpose },
      { key: 'Component', value: 'security' },
    ],
  });

  return webAcl;
}

/**
 * Associate a WAFv2 Web ACL with an ALB.
 */
function _createWebAclAssociation(
  scope: Construct,
  id: string,
  webAcl: wafv2.CfnWebACL,
  resourceArn: string,
): wafv2.CfnWebACLAssociation {
  const association = new wafv2.CfnWebACLAssociation(scope, id, {
    resourceArn,
    webAclArn: webAcl.attrArn,
  });

  return association;
}

/**
 * Create CloudWatch log group and WAF logging configuration.
 * CloudWatch log group name must start with "aws-waf-logs-" per AWS requirements.
 */
function _createWafLogging(
  scope: Construct,
  id: string,
  config: RegistryConfig,
  webAcl: wafv2.CfnWebACL,
  component: string,
): void {
  const logGroup = new logs.LogGroup(scope, `${id}LogGroup`, {
    logGroupName: `aws-waf-logs-${config.name}-${component}`,
    retention: WAF_LOG_RETENTION_DAYS,
    removalPolicy: cdk.RemovalPolicy.DESTROY,
  });

  cdk.Tags.of(logGroup).add('Purpose', `WAF logs for ${component}`);
  cdk.Tags.of(logGroup).add('Component', 'security');

  new wafv2.CfnLoggingConfiguration(scope, `${id}LogConfig`, {
    resourceArn: webAcl.attrArn,
    logDestinationConfigs: [logGroup.logGroupArn],
    redactedFields: [
      {
        singleHeader: { name: 'authorization' },
      },
    ],
  });
}
