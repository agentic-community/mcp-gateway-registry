/**
 * KeycloakService - L3 construct that creates the full Keycloak deployment.
 *
 * Translates the following Terraform resources into CDK:
 *   - keycloak-ecr.tf     (ECR repository + lifecycle policy)
 *   - keycloak-ecs.tf     (ECS cluster, task definition, service, autoscaling, SSM, IAM)
 *   - keycloak-alb.tf     (ALB, target group, HTTP/HTTPS listeners)
 *   - keycloak-dns.tf     (Route53 hosted zone lookup, ACM certificate, A record)
 *   - keycloak-security-groups.tf (ECS SG, ALB SG, CloudFront SG, DB SG rules)
 */

import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as route53 from 'aws-cdk-lib/aws-route53';
import * as route53targets from 'aws-cdk-lib/aws-route53-targets';
import * as acm from 'aws-cdk-lib/aws-certificatemanager';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import * as cr from 'aws-cdk-lib/custom-resources';
import { Construct } from 'constructs';
import { RegistryConfig } from '../registry-config';
import { putSecureSsmParam } from './_lib';

// ---------------------------------------------------------------------------
// Construct props
// ---------------------------------------------------------------------------

export interface KeycloakServiceProps {
  readonly config: RegistryConfig;
  readonly vpc: ec2.IVpc;
  readonly privateSubnets: ec2.ISubnet[];
  readonly publicSubnets: ec2.ISubnet[];
  /** Keycloak database security group from the data stack */
  readonly keycloakDbSg: ec2.ISecurityGroup;
  /** KMS key used for RDS / SSM encryption from the data stack */
  readonly rdsKmsKey: kms.IKey;
  /** ARN of the Secrets Manager secret containing Keycloak DB credentials (username, password) */
  readonly keycloakDbSecretArn: string;
}

// ---------------------------------------------------------------------------
// Construct
// ---------------------------------------------------------------------------

export class KeycloakService extends Construct {
  /** ECR repository for Keycloak container images */
  public readonly ecrRepo: ecr.Repository;

  /** ECS cluster running Keycloak */
  public readonly ecsCluster: ecs.Cluster;

  /** Security group attached to Keycloak ECS tasks */
  public readonly ecsSg: ec2.SecurityGroup;

  /** Security group attached to the Keycloak ALB */
  public readonly albSg: ec2.SecurityGroup;

  /** Application Load Balancer fronting Keycloak */
  public readonly alb: elbv2.ApplicationLoadBalancer;

  /** Resolved Keycloak domain name */
  public readonly keycloakDomain: string;

  /** Full HTTPS URL for Keycloak */
  public readonly keycloakUrl: string;

  constructor(scope: Construct, id: string, props: KeycloakServiceProps) {
    super(scope, id);

    const { config, vpc, privateSubnets, publicSubnets, keycloakDbSg, rdsKmsKey, keycloakDbSecretArn } = props;

    // Import the DB secret by ARN to avoid cross-stack cyclic dependency
    const keycloakDbSecret = secretsmanager.Secret.fromSecretCompleteArn(this, 'ImportedDbSecret', keycloakDbSecretArn);
    const region = config.awsRegion;

    // ------------------------------------------------------------------
    // Computed domain values (mirrors Terraform locals)
    // ------------------------------------------------------------------

    this.keycloakDomain = config.useRegionalDomains
      ? `kc.${region}.${config.baseDomain}`
      : config.keycloak.domain;

    const hostedZoneDomain = config.useRegionalDomains
      ? config.baseDomain
      : config.keycloak.rootDomain;

    const cloudfrontPrefixListName = config.cloudfront.prefixListName !== ''
      ? config.cloudfront.prefixListName
      : (config.cloudfront.enabled
        ? 'com.amazonaws.global.cloudfront.origin-facing'
        : '');

    // keycloakUrl depends on the ALB DNS name (created later in this
    // constructor), so we use cdk.Lazy to defer resolution until synth.
    let resolvedKeycloakUrl = '';
    this.keycloakUrl = cdk.Lazy.string({
      produce: () => resolvedKeycloakUrl,
    });

    // ------------------------------------------------------------------
    // ECR Repository
    // ------------------------------------------------------------------

    this.ecrRepo = new ecr.Repository(this, 'EcrRepo', {
      repositoryName: 'keycloak',
      imageScanOnPush: true,
      imageTagMutability: ecr.TagMutability.MUTABLE,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: true,
      lifecycleRules: [
        {
          rulePriority: 10,
          description: 'Keep last 10 git SHA tagged images',
          tagPrefixList: ['sha-'],
          maxImageCount: 10,
        },
        {
          rulePriority: 20,
          description: 'Expire untagged images older than 7 days',
          tagStatus: ecr.TagStatus.UNTAGGED,
          maxImageAge: cdk.Duration.days(7),
        },
      ],
    });

    // ECR repository policy - allow ECS pull and account push
    this.ecrRepo.addToResourcePolicy(new iam.PolicyStatement({
      sid: 'AllowECSPull',
      effect: iam.Effect.ALLOW,
      principals: [new iam.ServicePrincipal('ecs-tasks.amazonaws.com')],
      actions: [
        'ecr:GetDownloadUrlForLayer',
        'ecr:BatchGetImage',
        'ecr:BatchCheckLayerAvailability',
      ],
    }));

    this.ecrRepo.addToResourcePolicy(new iam.PolicyStatement({
      sid: 'AllowPush',
      effect: iam.Effect.ALLOW,
      principals: [new iam.AccountRootPrincipal()],
      actions: [
        'ecr:GetDownloadUrlForLayer',
        'ecr:BatchGetImage',
        'ecr:BatchCheckLayerAvailability',
        'ecr:PutImage',
        'ecr:InitiateLayerUpload',
        'ecr:UploadLayerPart',
        'ecr:CompleteLayerUpload',
      ],
    }));

    // ------------------------------------------------------------------
    // ECS Cluster
    // ------------------------------------------------------------------

    this.ecsCluster = new ecs.Cluster(this, 'EcsCluster', {
      clusterName: 'keycloak',
      vpc,
      containerInsights: true,
    });

    const cfnCluster = this.ecsCluster.node.defaultChild as ecs.CfnCluster;
    cfnCluster.capacityProviders = ['FARGATE', 'FARGATE_SPOT'];
    cfnCluster.defaultCapacityProviderStrategy = [
      { capacityProvider: 'FARGATE', base: 1, weight: 100 },
      { capacityProvider: 'FARGATE_SPOT', base: 0, weight: 0 },
    ];

    // ------------------------------------------------------------------
    // CloudWatch Log Group
    // ------------------------------------------------------------------

    const logGroup = new logs.LogGroup(this, 'LogGroup', {
      logGroupName: '/ecs/keycloak',
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // ------------------------------------------------------------------
    // SSM Parameters (SecureString - Keycloak admin credentials)
    // Note: CfnParameter is used because CDK L2 does not support SecureString
    // The database SSM parameters are created by the data stack.
    // ------------------------------------------------------------------

    putSecureSsmParam(this, 'SsmKeycloakAdmin', '/keycloak/admin', config.keycloak.adminUser, rdsKmsKey);
    putSecureSsmParam(this, 'SsmKeycloakAdminPassword', '/keycloak/admin_password', config.keycloak.adminPassword, rdsKmsKey);

    // Build ARNs for Keycloak SSM parameters (admin creds + DB URL)
    // DB username/password now come from Secrets Manager (rotation-safe)
    const ssmParamArns = [
      cdk.Stack.of(this).formatArn({
        service: 'ssm',
        resource: 'parameter',
        resourceName: 'keycloak/admin',
      }),
      cdk.Stack.of(this).formatArn({
        service: 'ssm',
        resource: 'parameter',
        resourceName: 'keycloak/admin_password',
      }),
      cdk.Stack.of(this).formatArn({
        service: 'ssm',
        resource: 'parameter',
        resourceName: 'keycloak/database/url',
      }),
    ];

    // ------------------------------------------------------------------
    // IAM - Task Execution Role
    // ------------------------------------------------------------------

    const taskExecRole = new iam.Role(this, 'TaskExecRole', {
      roleName: `keycloak-task-exec-role-${region}`,
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonECSTaskExecutionRolePolicy'),
      ],
    });

    // Inline policy: read SSM parameters
    taskExecRole.addToPolicy(new iam.PolicyStatement({
      sid: 'SSMGetParameters',
      effect: iam.Effect.ALLOW,
      actions: ['ssm:GetParameter', 'ssm:GetParameters'],
      resources: ssmParamArns,
    }));

    // Inline policy: KMS decrypt (wildcard required - key ARN determined at runtime by SSM)
    taskExecRole.addToPolicy(new iam.PolicyStatement({
      sid: 'KMSDecrypt',
      effect: iam.Effect.ALLOW,
      actions: ['kms:Decrypt'],
      resources: ['*'],
    }));

    // Inline policy: read Keycloak DB credentials from Secrets Manager
    taskExecRole.addToPolicy(new iam.PolicyStatement({
      sid: 'SecretsManagerGetDbCreds',
      effect: iam.Effect.ALLOW,
      actions: ['secretsmanager:GetSecretValue'],
      resources: [keycloakDbSecretArn],
    }));

    // Inline policy: CloudWatch logs
    taskExecRole.addToPolicy(new iam.PolicyStatement({
      sid: 'CloudWatchLogs',
      effect: iam.Effect.ALLOW,
      actions: ['logs:CreateLogStream', 'logs:PutLogEvents'],
      resources: [logGroup.logGroupArn + ':*'],
    }));

    // ------------------------------------------------------------------
    // IAM - Task Role
    // ------------------------------------------------------------------

    const taskRole = new iam.Role(this, 'TaskRole', {
      roleName: `keycloak-task-role-${region}`,
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
    });

    // Inline policy: SSM Session Manager (ECS Exec)
    taskRole.addToPolicy(new iam.PolicyStatement({
      sid: 'SSMSessionManager',
      effect: iam.Effect.ALLOW,
      actions: [
        'ssmmessages:CreateControlChannel',
        'ssmmessages:CreateDataChannel',
        'ssmmessages:OpenControlChannel',
        'ssmmessages:OpenDataChannel',
      ],
      resources: ['*'],
    }));

    // ------------------------------------------------------------------
    // Security Groups
    // ------------------------------------------------------------------

    // Keycloak ALB security group
    this.albSg = new ec2.SecurityGroup(this, 'AlbSg', {
      vpc,
      securityGroupName: 'keycloak-lb',
      description: 'Security group for Keycloak load balancer',
      allowAllOutbound: false,
    });

    // Keycloak ECS security group
    this.ecsSg = new ec2.SecurityGroup(this, 'EcsSg', {
      vpc,
      securityGroupName: 'keycloak-ecs',
      description: 'Security group for Keycloak ECS tasks',
      allowAllOutbound: false,
    });

    // --- ALB SG Rules ---

    for (const cidr of config.ingressCidrBlocks) {
      for (const port of [80, 443]) {
        this.albSg.addIngressRule(ec2.Peer.ipv4(cidr), ec2.Port.tcp(port), `Ingress from ${cidr}`);
      }
    }

    // ALB egress: port 8080 to ECS SG (application traffic)
    this.albSg.addEgressRule(
      this.ecsSg,
      ec2.Port.tcp(8080),
      'Egress from load balancer to Keycloak ECS task',
    );

    // ALB egress: port 9000 to ECS SG (management port - metrics, health)
    this.albSg.addEgressRule(
      this.ecsSg,
      ec2.Port.tcp(9000),
      'Egress from load balancer to Keycloak management port',
    );

    // --- ECS SG Rules ---

    // ECS egress: HTTPS to internet
    this.ecsSg.addEgressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(443),
      'Egress from Keycloak ECS task to internet (HTTPS)',
    );

    // ECS egress: DNS UDP
    this.ecsSg.addEgressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.udp(53),
      'Egress from Keycloak ECS task for DNS',
    );

    // ECS egress: MySQL 3306 to Keycloak DB SG
    this.ecsSg.addEgressRule(
      keycloakDbSg,
      ec2.Port.tcp(3306),
      'Egress from Keycloak ECS task to database',
    );

    // ECS ingress: port 8080 from ALB SG (application traffic)
    this.ecsSg.addIngressRule(
      this.albSg,
      ec2.Port.tcp(8080),
      'Ingress from load balancer to Keycloak ECS task',
    );

    // ECS ingress: port 9000 from ALB SG (health checks on management port)
    this.ecsSg.addIngressRule(
      this.albSg,
      ec2.Port.tcp(9000),
      'Ingress from load balancer to Keycloak management port for health checks',
    );

    // CfnSecurityGroupIngress avoids cross-stack cyclic dependency
    // (Auth depends on Data, so Data SG cannot reference Auth SG)
    new ec2.CfnSecurityGroupIngress(this, 'DbFromEcs', {
      groupId: (keycloakDbSg as ec2.SecurityGroup).securityGroupId,
      ipProtocol: 'tcp',
      fromPort: 3306,
      toPort: 3306,
      sourceSecurityGroupId: this.ecsSg.securityGroupId,
      description: 'Ingress to database from Keycloak ECS task',
    });

    // --- CloudFront SG (conditional) ---
    // Store reference so we can attach it to the ALB after ALB creation
    let cloudfrontAlbSg: ec2.SecurityGroup | undefined;

    if (cloudfrontPrefixListName !== '') {
      const prefixListLookup = new cr.AwsCustomResource(this, 'PrefixListLookup', {
        onCreate: {
          service: 'EC2',
          action: 'describeManagedPrefixLists',
          parameters: {
            Filters: [{ Name: 'prefix-list-name', Values: [cloudfrontPrefixListName] }],
          },
          physicalResourceId: cr.PhysicalResourceId.of('CloudFrontPrefixListLookup'),
        },
        onUpdate: {
          service: 'EC2',
          action: 'describeManagedPrefixLists',
          parameters: {
            Filters: [{ Name: 'prefix-list-name', Values: [cloudfrontPrefixListName] }],
          },
          physicalResourceId: cr.PhysicalResourceId.of('CloudFrontPrefixListLookup'),
        },
        policy: cr.AwsCustomResourcePolicy.fromSdkCalls({
          resources: cr.AwsCustomResourcePolicy.ANY_RESOURCE,
        }),
      });

      const prefixListId = prefixListLookup.getResponseField('PrefixLists.0.PrefixListId');

      cloudfrontAlbSg = new ec2.SecurityGroup(this, 'AlbCloudFrontSg', {
        vpc,
        securityGroupName: 'keycloak-lb-cloudfront',
        description: 'Security group for CloudFront access to Keycloak ALB',
        allowAllOutbound: false,
      });

      new ec2.CfnSecurityGroupIngress(this, 'CfAlbIngressPrefixList', {
        groupId: cloudfrontAlbSg.securityGroupId,
        ipProtocol: 'tcp',
        fromPort: 80,
        toPort: 80,
        sourcePrefixListId: prefixListId,
        description: 'Ingress from prefix list to load balancer (HTTP) - CloudFront origin-facing IPs',
      });

      cloudfrontAlbSg.addEgressRule(
        this.ecsSg,
        ec2.Port.tcp(8080),
        'Egress from CloudFront SG to Keycloak ECS task',
      );

      this.ecsSg.addIngressRule(
        cloudfrontAlbSg,
        ec2.Port.tcp(8080),
        'Ingress from CloudFront LB security group to Keycloak ECS task',
      );
    }

    // ------------------------------------------------------------------
    // Task Definition
    // ------------------------------------------------------------------

    const taskDef = new ecs.FargateTaskDefinition(this, 'TaskDef', {
      family: 'keycloak',
      cpu: 1024,
      memoryLimitMiB: 2048,
      executionRole: taskExecRole,
      taskRole,
    });

    // Determine container image
    const containerImage = config.images.keycloak
      ? ecs.ContainerImage.fromRegistry(config.images.keycloak)
      : ecs.ContainerImage.fromEcrRepository(this.ecrRepo, 'latest');

    const container = taskDef.addContainer('keycloak', {
      containerName: 'keycloak',
      image: containerImage,
      command: [
        'start-dev',
        '--spi-realm-default-ssl-required=NONE',
      ],
      essential: true,
      environment: {
        AWS_REGION: region,
        KC_DB: 'mysql',
        KC_PROXY_HEADERS: 'xforwarded',
        KC_HOSTNAME: this.keycloakUrl,
        KC_HOSTNAME_ADMIN: this.keycloakUrl,
        KC_HTTP_ENABLED: 'true',
        KC_HEALTH_ENABLED: 'true',
        KC_METRICS_ENABLED: 'true',
        KEYCLOAK_LOGLEVEL: config.keycloak.logLevel,
      },
      secrets: {
        KEYCLOAK_ADMIN: ecs.Secret.fromSsmParameter(
          ssm.StringParameter.fromSecureStringParameterAttributes(this, 'SsmRefAdmin', {
            parameterName: '/keycloak/admin',
          }),
        ),
        KEYCLOAK_ADMIN_PASSWORD: ecs.Secret.fromSsmParameter(
          ssm.StringParameter.fromSecureStringParameterAttributes(this, 'SsmRefAdminPw', {
            parameterName: '/keycloak/admin_password',
          }),
        ),
        KC_DB_URL: ecs.Secret.fromSsmParameter(
          ssm.StringParameter.fromSecureStringParameterAttributes(this, 'SsmRefDbUrl', {
            parameterName: '/keycloak/database/url',
          }),
        ),
        KC_DB_USERNAME: ecs.Secret.fromSecretsManager(keycloakDbSecret, 'username'),
        KC_DB_PASSWORD: ecs.Secret.fromSecretsManager(keycloakDbSecret, 'password'),
      },
      logging: ecs.LogDrivers.awsLogs({
        logGroup,
        streamPrefix: 'ecs',
      }),
      healthCheck: {
        command: ['CMD-SHELL', 'exit 0'],
        interval: cdk.Duration.seconds(30),
        timeout: cdk.Duration.seconds(5),
        retries: 3,
        startPeriod: cdk.Duration.seconds(60),
      },
      readonlyRootFilesystem: false,
    });

    container.addPortMappings(
      { containerPort: 8080, hostPort: 8080, protocol: ecs.Protocol.TCP, name: 'keycloak' },
      { containerPort: 9000, hostPort: 9000, protocol: ecs.Protocol.TCP, name: 'keycloak-management' },
    );

    // ------------------------------------------------------------------
    // ALB + Target Group + Listeners
    // ------------------------------------------------------------------

    this.alb = new elbv2.ApplicationLoadBalancer(this, 'Alb', {
      loadBalancerName: 'keycloak-alb',
      vpc,
      internetFacing: true,
      vpcSubnets: { subnets: publicSubnets },
      securityGroup: this.albSg,
      dropInvalidHeaderFields: true,
      deletionProtection: false,
    });

    if (cloudfrontAlbSg) {
      this.alb.addSecurityGroup(cloudfrontAlbSg);
    }

    const targetGroup = new elbv2.ApplicationTargetGroup(this, 'TargetGroup', {
      targetGroupName: 'keycloak-tg',
      port: 8080,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targetType: elbv2.TargetType.IP,
      vpc,
      deregistrationDelay: cdk.Duration.seconds(30),
      healthCheck: {
        enabled: true,
        healthyThresholdCount: 2,
        unhealthyThresholdCount: 3,
        timeout: cdk.Duration.seconds(5),
        interval: cdk.Duration.seconds(30),
        path: '/',
        port: '8080',
        healthyHttpCodes: '200-399',
        protocol: elbv2.Protocol.HTTP,
      },
      stickinessCookieDuration: cdk.Duration.seconds(86400),
    });

    // DNS and HTTPS resources (conditional on enable_route53_dns)
    if (config.enableRoute53Dns) {
      const hostedZone = route53.HostedZone.fromLookup(this, 'HostedZone', {
        domainName: hostedZoneDomain,
      });

      const certificate = new acm.Certificate(this, 'Certificate', {
        domainName: this.keycloakDomain,
        validation: acm.CertificateValidation.fromDns(hostedZone),
      });
      cdk.Tags.of(certificate).add('Name', 'keycloak-cert');

      // HTTPS listener (port 443)
      this.alb.addListener('HttpsListener', {
        port: 443,
        protocol: elbv2.ApplicationProtocol.HTTPS,
        sslPolicy: elbv2.SslPolicy.RECOMMENDED_TLS,
        certificates: [certificate],
        defaultTargetGroups: [targetGroup],
      });

      // HTTP listener behavior depends on whether CloudFront is also enabled
      if (!config.cloudfront.enabled) {
        // Mode 2: Route53 without CloudFront - redirect HTTP to HTTPS
        this.alb.addListener('HttpListener', {
          port: 80,
          protocol: elbv2.ApplicationProtocol.HTTP,
          defaultAction: elbv2.ListenerAction.redirect({
            port: '443',
            protocol: 'HTTPS',
            permanent: true,
          }),
        });
      } else {
        // Mode 3: Route53 with CloudFront - forward HTTP (CloudFront handles TLS)
        this.alb.addListener('HttpListener', {
          port: 80,
          protocol: elbv2.ApplicationProtocol.HTTP,
          defaultTargetGroups: [targetGroup],
        });
      }

      // Route53 A record aliased to ALB (or CloudFront when both enabled)
      // When CloudFront is also enabled the alias target would be CloudFront,
      // but since CloudFront is managed in a separate stack, we point to ALB here.
      // The CloudFront stack can override this record if needed.
      new route53.ARecord(this, 'AliasRecord', {
        zone: hostedZone,
        recordName: this.keycloakDomain,
        target: route53.RecordTarget.fromAlias(
          new route53targets.LoadBalancerTarget(this.alb),
        ),
      });
    } else {
      // No Route53: HTTP listener forwards directly (CloudFront or plain HTTP)
      this.alb.addListener('HttpListener', {
        port: 80,
        protocol: elbv2.ApplicationProtocol.HTTP,
        defaultTargetGroups: [targetGroup],
      });

      // Without Route53, the keycloakDomain is unreachable via DNS.
      // Fall back to the ALB DNS name so downstream services can connect.
      this.keycloakDomain = this.alb.loadBalancerDnsName;
    }

    // Now that the ALB exists, resolve the Keycloak URL.
    if (config.enableRoute53Dns) {
      resolvedKeycloakUrl = `https://${this.keycloakDomain}`;
    } else {
      resolvedKeycloakUrl = `http://${this.alb.loadBalancerDnsName}`;
    }

    // ------------------------------------------------------------------
    // ECS Fargate Service
    // ------------------------------------------------------------------

    const fargateService = new ecs.FargateService(this, 'FargateService', {
      serviceName: 'keycloak',
      cluster: this.ecsCluster,
      taskDefinition: taskDef,
      desiredCount: 1,
      assignPublicIp: false,
      vpcSubnets: { subnets: privateSubnets },
      securityGroups: [this.ecsSg],
      enableExecuteCommand: true,
      circuitBreaker: { enable: true, rollback: true },
    });

    // Register service with ALB target group
    fargateService.attachToApplicationTargetGroup(targetGroup);

    // ------------------------------------------------------------------
    // Auto Scaling
    // ------------------------------------------------------------------

    const scaling = fargateService.autoScaleTaskCount({
      minCapacity: 1,
      maxCapacity: 4,
    });

    scaling.scaleOnCpuUtilization('CpuScaling', {
      targetUtilizationPercent: 70,
    });

    scaling.scaleOnMemoryUtilization('MemoryScaling', {
      targetUtilizationPercent: 80,
    });

    // ------------------------------------------------------------------
    // Tags
    // ------------------------------------------------------------------

    cdk.Tags.of(this).add('Project', 'mcp-gateway-registry');
    cdk.Tags.of(this).add('Component', 'auth');
    cdk.Tags.of(this).add('Environment', 'production');
    cdk.Tags.of(this).add('ManagedBy', 'cdk');
  }
}
