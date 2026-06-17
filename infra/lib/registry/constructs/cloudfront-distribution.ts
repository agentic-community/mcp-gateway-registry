/**
 * CloudFrontDistribution - L3 construct that creates CloudFront distributions
 * for MCP Gateway and Keycloak with optional cross-region ACM certificates
 * and an S3 logging bucket.
 *
 * Translated from:
 *   - terraform/aws-ecs/cloudfront.tf
 *   - terraform/aws-ecs/cloudfront-acm.tf
 *   - terraform/aws-ecs/cloudfront-logging.tf
 *
 * This construct is a no-op when config.cloudfront.enabled is false.
 */

import * as cdk from 'aws-cdk-lib';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import * as acm from 'aws-cdk-lib/aws-certificatemanager';
import * as route53 from 'aws-cdk-lib/aws-route53';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';
import { RegistryConfig } from '../registry-config';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface CloudFrontDistributionProps {
  readonly config: RegistryConfig;
  /** DNS name of the MCP Gateway ALB */
  readonly registryAlbDns: string;
  /** DNS name of the Keycloak ALB */
  readonly keycloakAlbDns: string;
  /** Root hosted zone domain (e.g. mycorp.click) for cross-region certificates */
  readonly hostedZoneDomain: string;
}

// ---------------------------------------------------------------------------
// Construct
// ---------------------------------------------------------------------------

export class CloudFrontDistribution extends Construct {
  /** Domain name of the MCP Gateway CloudFront distribution */
  public readonly mcpGatewayDistributionDomainName: string;

  /** Domain name of the Keycloak CloudFront distribution */
  public readonly keycloakDistributionDomainName: string;

  constructor(scope: Construct, id: string, props: CloudFrontDistributionProps) {
    super(scope, id);

    const { config, registryAlbDns, keycloakAlbDns, hostedZoneDomain } = props;

    // No-op when CloudFront is disabled
    if (!config.cloudfront.enabled) {
      this.mcpGatewayDistributionDomainName = '';
      this.keycloakDistributionDomainName = '';
      return;
    }

    // ------------------------------------------------------------------
    // S3 bucket for CloudFront access logs
    // ------------------------------------------------------------------

    const logsBucket = _createLogsBucket(this, config);

    // ------------------------------------------------------------------
    // Cross-region ACM certificates (us-east-1) - only when Route53 is also enabled
    // ------------------------------------------------------------------

    // Look up the Route53 hosted zone when custom domains are enabled
    const hostedZone = config.enableRoute53Dns
      ? route53.HostedZone.fromLookup(this, 'HostedZone', {
          domainName: hostedZoneDomain,
        })
      : undefined;

    const registryCert = _createCrossRegionCert(
      this,
      'RegistryCert',
      config,
      hostedZone,
      `registry.${hostedZoneDomain}`,
    );

    const keycloakDomain = config.useRegionalDomains
      ? `kc.${config.awsRegion}.${config.baseDomain}`
      : config.keycloak.domain;

    const keycloakCert = _createCrossRegionCert(
      this,
      'KeycloakCert',
      config,
      hostedZone,
      keycloakDomain,
    );

    // ------------------------------------------------------------------
    // MCP Gateway CloudFront distribution
    // ------------------------------------------------------------------

    const mcpGatewayDistribution = _createDistribution(
      this,
      'McpGateway',
      {
        config,
        albDns: registryAlbDns,
        originId: 'mcp-gateway-alb',
        comment: `${config.name} MCP Gateway Registry CloudFront Distribution`,
        logsBucket,
        logsPrefix: 'mcp-gateway/',
        certificate: registryCert,
        domainAlias: config.enableRoute53Dns ? `registry.${hostedZoneDomain}` : undefined,
        includeCloudFrontForwardedProtoHeader: true,
      },
    );

    this.mcpGatewayDistributionDomainName = mcpGatewayDistribution.distributionDomainName;

    // ------------------------------------------------------------------
    // Keycloak CloudFront distribution
    // ------------------------------------------------------------------

    const keycloakDistribution = _createDistribution(
      this,
      'Keycloak',
      {
        config,
        albDns: keycloakAlbDns,
        originId: 'keycloak-alb',
        comment: `${config.name} Keycloak CloudFront Distribution`,
        logsBucket,
        logsPrefix: 'keycloak/',
        certificate: keycloakCert,
        domainAlias: config.enableRoute53Dns ? keycloakDomain : undefined,
        includeCloudFrontForwardedProtoHeader: false,
      },
    );

    this.keycloakDistributionDomainName = keycloakDistribution.distributionDomainName;
  }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/**
 * Create the S3 bucket for CloudFront access logs with full security hardening.
 */
function _createLogsBucket(scope: Construct, config: RegistryConfig): s3.Bucket {
  const bucket = new s3.Bucket(scope, 'LogsBucket', {
    bucketName: `ai-registry-${config.awsRegion}-cloudfront-logs`,
    blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
    encryption: s3.BucketEncryption.S3_MANAGED,
    versioned: true,
    objectOwnership: s3.ObjectOwnership.BUCKET_OWNER_PREFERRED,
    removalPolicy: cdk.RemovalPolicy.DESTROY,
    autoDeleteObjects: true,
    lifecycleRules: [
      {
        id: 'delete-old-logs',
        enabled: true,
        expiration: cdk.Duration.days(90),
      },
    ],
  });

  cdk.Tags.of(bucket).add('Purpose', 'CloudFront access logs');
  cdk.Tags.of(bucket).add('Component', 'logging');

  return bucket;
}

/**
 * Create a cross-region ACM certificate in us-east-1 for CloudFront.
 * Only created when both CloudFront and Route53 DNS are enabled.
 * Returns undefined when Route53 is disabled.
 */
function _createCrossRegionCert(
  scope: Construct,
  id: string,
  config: RegistryConfig,
  hostedZone: route53.IHostedZone | undefined,
  domainName: string,
): acm.ICertificate | undefined {
  if (!config.enableRoute53Dns || !hostedZone) {
    return undefined;
  }

  // DnsValidatedCertificate handles cross-region provisioning in us-east-1
  // and automatic DNS validation via Route53.
  const cert = new acm.DnsValidatedCertificate(scope, id, {
    domainName,
    hostedZone,
    region: 'us-east-1',
    cleanupRoute53Records: true,
  });

  return cert;
}

/**
 * Configuration for a single CloudFront distribution.
 */
interface DistributionConfig {
  readonly config: RegistryConfig;
  readonly albDns: string;
  readonly originId: string;
  readonly comment: string;
  readonly logsBucket: s3.IBucket;
  readonly logsPrefix: string;
  readonly certificate: acm.ICertificate | undefined;
  readonly domainAlias: string | undefined;
  readonly includeCloudFrontForwardedProtoHeader: boolean;
}

/**
 * Create a CloudFront distribution for an ALB origin.
 *
 * - HTTP-only origin to avoid TLS certificate mismatch with ALB DNS names.
 * - Caching disabled; AllViewer origin request policy forwards all headers.
 * - PriceClass_100 (NA + EU edge locations).
 * - Compression enabled.
 */
function _createDistribution(
  scope: Construct,
  id: string,
  distConfig: DistributionConfig,
): cloudfront.Distribution {
  // Build custom origin headers
  const customHeaders: Record<string, string> = {
    'X-Forwarded-Proto': 'https',
  };

  if (distConfig.includeCloudFrontForwardedProtoHeader) {
    customHeaders['X-Cloudfront-Forwarded-Proto'] = 'https';
  }

  const origin = new origins.HttpOrigin(distConfig.albDns, {
    protocolPolicy: cloudfront.OriginProtocolPolicy.HTTP_ONLY,
    httpPort: 80,
    httpsPort: 443,
    customHeaders,
  });

  // Determine viewer certificate configuration
  const certificateConfig: {
    certificate?: acm.ICertificate;
    domainNames?: string[];
    sslSupportMethod?: cloudfront.SSLMethod;
    minimumProtocolVersion?: cloudfront.SecurityPolicyProtocol;
  } = {};

  if (distConfig.certificate && distConfig.domainAlias) {
    certificateConfig.certificate = distConfig.certificate;
    certificateConfig.domainNames = [distConfig.domainAlias];
    certificateConfig.sslSupportMethod = cloudfront.SSLMethod.SNI;
    certificateConfig.minimumProtocolVersion = cloudfront.SecurityPolicyProtocol.TLS_V1_2_2021;
  }

  const distribution = new cloudfront.Distribution(scope, `${id}Distribution`, {
    enabled: true,
    comment: distConfig.comment,
    priceClass: cloudfront.PriceClass.PRICE_CLASS_100,
    enableLogging: true,
    logBucket: distConfig.logsBucket,
    logFilePrefix: distConfig.logsPrefix,
    logIncludesCookies: false,
    defaultBehavior: {
      origin,
      allowedMethods: cloudfront.AllowedMethods.ALLOW_ALL,
      cachedMethods: cloudfront.CachedMethods.CACHE_GET_HEAD,
      cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED,
      originRequestPolicy: cloudfront.OriginRequestPolicy.ALL_VIEWER,
      viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
      compress: true,
    },
    ...certificateConfig,
  });

  cdk.Tags.of(distribution).add('Component', id.toLowerCase());

  return distribution;
}
