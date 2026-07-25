/**
 * WafRules - L3 construct that creates WAFv2 Web ACLs for MCP Gateway
 * and Keycloak ALBs with managed rule groups and rate limiting.
 *
 * Translated from: terraform/aws-ecs/waf.tf
 *
 * Each Web ACL contains 5 rules:
 *   1. AWSManagedRulesCommonRuleSet   (priority 1) — everywhere EXCEPT the
 *      Keycloak anonymous DCR endpoint. Full block on all sub-rules.
 *   2. AWSManagedRulesCommonRuleSet   (priority 2) — ONLY the Keycloak DCR
 *      endpoint. Same managed group but with count-mode overrides on the
 *      two sub-rules that false-positive on RFC 8252 loopback redirect_uris
 *      (EC2MetaDataSSRF_BODY, GenericRFI_BODY).
 *   3. AWSManagedRulesKnownBadInputsRuleSet (priority 3)
 *   4. IP-based rate limit: 100 requests / 5 minutes per IP (priority 4)
 *   5. Global rate limit: 2000 requests / 5 minutes (priority 5)
 *
 * Rules 1 and 2 share the same managed group but have mutually-exclusive
 * scopeDownStatements. Together they preserve full CRS protection on every
 * path except the one endpoint (Keycloak anonymous DCR) that has a documented
 * false-positive on loopback callback URIs.
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

// Keycloak anonymous Dynamic Client Registration endpoint. Requests to this
// path legitimately carry loopback callback URIs (http://localhost:<port>/...)
// per RFC 8252, which trip EC2MetaDataSSRF_BODY (and, once that is downgraded,
// GenericRFI_BODY as well). Elsewhere both rules stay at Block.
const KEYCLOAK_DCR_PATH_PREFIX = '/realms/mcp-gateway/clients-registrations/';

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
    // Guard on mcpGatewayAlbArn (mirrors the Keycloak block below). CloudFront
    // + WAF is now embedded per-stack, so this construct is instantiated in
    // BOTH the Auth stack (mcpGatewayAlbArn: '') and the Service stack (real
    // registry ALB ARN). Without this guard the Auth stack would ALSO build a
    // Web ACL named `${config.name}-mcp-gateway-waf` — a WAFv2 REGIONAL name
    // collision with the Service stack's ACL — and associate it to an empty
    // resourceArn (a deploy failure). Only the stack that owns the registry ALB
    // builds this ACL.

    if (mcpGatewayAlbArn) {
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
    } else {
      this.mcpGatewayWebAclArn = '';
    }

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
  // Reusable byte match: URI path starts with the Keycloak DCR endpoint.
  const dcrPathMatch: wafv2.CfnWebACL.StatementProperty = {
    byteMatchStatement: {
      fieldToMatch: { uriPath: {} },
      positionalConstraint: 'STARTS_WITH',
      searchString: KEYCLOAK_DCR_PATH_PREFIX,
      textTransformations: [{ priority: 0, type: 'NONE' }],
    },
  };

  return [
    // Rule 1: CommonRuleSet EVERYWHERE EXCEPT Keycloak DCR.
    // Full managed group at Block. AWS requires ManagedRuleGroupStatement at
    // the rule's top level (cannot nest under AndStatement/NotStatement), so
    // the path exclusion goes via managedRuleGroupStatement.scopeDownStatement.
    {
      name: 'CommonRuleSet',
      priority: 1,
      overrideAction: { none: {} },
      statement: {
        managedRuleGroupStatement: {
          name: 'AWSManagedRulesCommonRuleSet',
          vendorName: 'AWS',
          // Managed group only evaluates when scope-down matches.
          // scope-down = NOT starts_with('/realms/*/clients-registrations/').
          scopeDownStatement: {
            notStatement: { statement: dcrPathMatch },
          },
        },
      },
      visibilityConfig: {
        cloudWatchMetricsEnabled: true,
        metricName: 'CommonRuleSetMetric',
        sampledRequestsEnabled: true,
      },
    },
    // Rule 2: CommonRuleSet on Keycloak DCR endpoint ONLY, with count-mode
    // overrides for the two sub-rules that false-positive on RFC 8252 loopback
    // callback URIs. Every other CommonRuleSet sub-rule (LFI, XSS, SQLi, size,
    // bad bots, etc.) still Blocks on this endpoint.
    //
    //   EC2MetaDataSSRF_BODY  — matches 'localhost' / '127.0.0.1' / '169.254.*'
    //                           in the JSON body. DCR redirect_uris legitimately
    //                           contain these per RFC 8252.
    //   GenericRFI_BODY       — matches '127.0.0.1' / '169.254.*' in the body.
    //                           Shadowed while _BODY was terminating; now that
    //                           _BODY is Count on this path, GenericRFI_BODY
    //                           becomes the next terminating match unless also
    //                           downgraded.
    {
      name: 'CommonRuleSetForDcr',
      priority: 2,
      overrideAction: { none: {} },
      statement: {
        managedRuleGroupStatement: {
          name: 'AWSManagedRulesCommonRuleSet',
          vendorName: 'AWS',
          ruleActionOverrides: [
            { name: 'EC2MetaDataSSRF_BODY', actionToUse: { count: {} } },
            { name: 'GenericRFI_BODY', actionToUse: { count: {} } },
          ],
          scopeDownStatement: dcrPathMatch,
        },
      },
      visibilityConfig: {
        cloudWatchMetricsEnabled: true,
        metricName: 'CommonRuleSetForDcrMetric',
        sampledRequestsEnabled: true,
      },
    },
    // Rule 3: AWS Managed Rules - Known Bad Inputs
    {
      name: 'AWSManagedRulesKnownBadInputsRuleSet',
      priority: 3,
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
    // Rule 4: IP-based rate limiting (100 req / 5 min per IP)
    {
      name: 'IPRateLimitRule',
      priority: 4,
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
    // Rule 5: Global rate limiting (2000 req / 5 min total across all clients).
    // WAF requires a scopeDownStatement when aggregateKeyType=CONSTANT. Match-all
    // achieved via byte_match on URI path containing "/" — every HTTP request
    // has a URI, so all traffic counts toward the global bucket.
    {
      name: 'GlobalRateLimitRule',
      priority: 5,
      action: { block: {} },
      statement: {
        rateBasedStatement: {
          limit: GLOBAL_RATE_LIMIT,
          aggregateKeyType: 'CONSTANT',
          scopeDownStatement: {
            byteMatchStatement: {
              fieldToMatch: { uriPath: {} },
              positionalConstraint: 'CONTAINS',
              searchString: '/',
              textTransformations: [{ priority: 0, type: 'NONE' }],
            },
          },
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
 * Create a WAFv2 Web ACL with the standard 5-rule set (see file header).
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
        singleHeader: { Name: 'authorization' },
      },
    ],
  });
}
