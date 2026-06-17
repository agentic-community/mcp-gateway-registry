/**
 * RegistryAlb - L3 construct for the registry ALB, S3 access logs bucket,
 * security group, target groups (registry / auth / gradio), and listeners.
 *
 * Translated from: terraform/aws-ecs/modules/mcp-gateway/networking.tf
 *                  terraform/aws-ecs/alb-logging.tf
 */

import * as cdk from 'aws-cdk-lib';
import * as acm from 'aws-cdk-lib/aws-certificatemanager';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as route53 from 'aws-cdk-lib/aws-route53';
import * as route53targets from 'aws-cdk-lib/aws-route53-targets';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';
import { RegistryConfig } from '../registry-config';

export interface RegistryAlbProps {
  readonly config: RegistryConfig;
  readonly vpc: ec2.IVpc;
  readonly publicSubnets: ec2.ISubnet[];
}

export class RegistryAlb extends Construct {
  public readonly alb: elbv2.ApplicationLoadBalancer;
  public readonly albSg: ec2.SecurityGroup;
  public readonly registryTg: elbv2.ApplicationTargetGroup;
  public readonly authTg: elbv2.ApplicationTargetGroup;
  public readonly gradioTg: elbv2.ApplicationTargetGroup;
  public readonly httpListener: elbv2.ApplicationListener;
  public readonly httpsListener?: elbv2.ApplicationListener;

  constructor(scope: Construct, id: string, props: RegistryAlbProps) {
    super(scope, id);

    const { config, vpc, publicSubnets } = props;
    const { name: namePrefix } = config;
    const stack = cdk.Stack.of(this);

    // S3 access-logs bucket
    const logsBucket = new s3.Bucket(this, 'LogsBucket', {
      bucketName: `${namePrefix}-${stack.region}-${stack.account}-alb-logs`,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      lifecycleRules: [{ id: 'delete-old-logs', enabled: true, expiration: cdk.Duration.days(90) }],
    });

    logsBucket.addToResourcePolicy(new iam.PolicyStatement({
      sid: 'EnforceTLS',
      effect: iam.Effect.DENY,
      principals: [new iam.AnyPrincipal()],
      actions: ['s3:*'],
      resources: [logsBucket.bucketArn, `${logsBucket.bucketArn}/*`],
      conditions: { Bool: { 'aws:SecureTransport': 'false' } },
    }));
    logsBucket.addToResourcePolicy(new iam.PolicyStatement({
      sid: 'AWSLogDeliveryWrite',
      effect: iam.Effect.ALLOW,
      principals: [new iam.ServicePrincipal('logdelivery.elasticloadbalancing.amazonaws.com')],
      actions: ['s3:PutObject'],
      resources: [`${logsBucket.bucketArn}/*`],
      conditions: { StringEquals: { 's3:x-amz-acl': 'bucket-owner-full-control' } },
    }));
    logsBucket.addToResourcePolicy(new iam.PolicyStatement({
      sid: 'AWSLogDeliveryAclCheck',
      effect: iam.Effect.ALLOW,
      principals: [new iam.ServicePrincipal('logdelivery.elasticloadbalancing.amazonaws.com')],
      actions: ['s3:GetBucketAcl'],
      resources: [logsBucket.bucketArn],
    }));

    // SG with ingress on 80/443/8888/7860 from configured CIDRs
    this.albSg = new ec2.SecurityGroup(this, 'Sg', {
      vpc,
      securityGroupName: `${namePrefix}-alb`,
      description: 'Security group for MCP Gateway Registry ALB',
      allowAllOutbound: true,
    });
    cdk.Tags.of(this.albSg).add('Name', `${namePrefix}-alb`);

    for (const cidr of config.ingressCidrBlocks) {
      for (const port of [80, 443, 8888, 7860]) {
        this.albSg.addIngressRule(
          ec2.Peer.ipv4(cidr),
          ec2.Port.tcp(port),
          `Ingress from ${cidr} to port ${port}`,
        );
      }
    }

    // ALB (internet-facing matches Terraform default)
    this.alb = new elbv2.ApplicationLoadBalancer(this, 'Alb', {
      loadBalancerName: `${namePrefix}-alb`,
      vpc,
      internetFacing: true,
      vpcSubnets: { subnets: publicSubnets },
      securityGroup: this.albSg,
      dropInvalidHeaderFields: true,
      deletionProtection: false,
    });
    this.alb.logAccessLogs(logsBucket);

    // Target groups
    const tgHealth = {
      enabled: true,
      healthyThresholdCount: 2,
      unhealthyThresholdCount: 2,
      timeout: cdk.Duration.seconds(5),
      interval: cdk.Duration.seconds(30),
      path: '/health',
      healthyHttpCodes: '200',
      protocol: elbv2.Protocol.HTTP,
    } as const;

    this.registryTg = new elbv2.ApplicationTargetGroup(this, 'RegistryTg', {
      targetGroupName: `${namePrefix}-registry-tg`,
      port: 8080,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targetType: elbv2.TargetType.IP,
      vpc,
      deregistrationDelay: cdk.Duration.seconds(5),
      healthCheck: { ...tgHealth, port: '8080' },
    });

    this.authTg = new elbv2.ApplicationTargetGroup(this, 'AuthTg', {
      targetGroupName: `${namePrefix}-auth-tg`,
      port: 8888,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targetType: elbv2.TargetType.IP,
      vpc,
      deregistrationDelay: cdk.Duration.seconds(5),
      healthCheck: tgHealth,
    });

    this.gradioTg = new elbv2.ApplicationTargetGroup(this, 'GradioTg', {
      targetGroupName: `${namePrefix}-gradio-tg`,
      port: 7860,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targetType: elbv2.TargetType.IP,
      vpc,
      deregistrationDelay: cdk.Duration.seconds(5),
      healthCheck: tgHealth,
    });

    // Auto-issue ACM cert + Route53 alias when enableRoute53Dns is true and no
    // CloudFront fronts the ALB. Mirrors terraform/aws-ecs/registry-dns.tf.
    let certificateArn = config.certificateArn;
    if (config.enableRoute53Dns && !config.cloudfront.enabled && !certificateArn) {
      const hostedZoneDomain = config.baseDomain;
      const registryDomain = `registry.${hostedZoneDomain}`;
      const hostedZone = route53.HostedZone.fromLookup(this, 'HostedZone', {
        domainName: hostedZoneDomain,
      });
      const cert = new acm.Certificate(this, 'Certificate', {
        domainName: registryDomain,
        validation: acm.CertificateValidation.fromDns(hostedZone),
      });
      certificateArn = cert.certificateArn;
      new route53.ARecord(this, 'AliasRecord', {
        zone: hostedZone,
        recordName: registryDomain,
        target: route53.RecordTarget.fromAlias(new route53targets.LoadBalancerTarget(this.alb)),
      });
    }

    // Listeners — HTTPS only when a cert is configured.
    this.httpListener = this.alb.addListener('HttpListener', {
      port: 80,
      protocol: elbv2.ApplicationProtocol.HTTP,
      defaultTargetGroups: [this.registryTg],
    });

    const enableHttps = certificateArn !== '';
    if (enableHttps) {
      this.httpsListener = this.alb.addListener('HttpsListener', {
        port: 443,
        protocol: elbv2.ApplicationProtocol.HTTPS,
        sslPolicy: elbv2.SslPolicy.TLS13_RES,
        certificates: [elbv2.ListenerCertificate.fromArn(certificateArn)],
        defaultTargetGroups: [this.registryTg],
      });
    }

    this.alb.addListener('AuthListener', {
      port: 8888,
      protocol: enableHttps ? elbv2.ApplicationProtocol.HTTPS : elbv2.ApplicationProtocol.HTTP,
      ...(enableHttps
        ? {
            sslPolicy: elbv2.SslPolicy.TLS13_RES,
            certificates: [elbv2.ListenerCertificate.fromArn(certificateArn)],
          }
        : {}),
      defaultTargetGroups: [this.authTg],
    });

    this.alb.addListener('GradioListener', {
      port: 7860,
      protocol: elbv2.ApplicationProtocol.HTTP,
      defaultTargetGroups: [this.gradioTg],
    });
  }
}
