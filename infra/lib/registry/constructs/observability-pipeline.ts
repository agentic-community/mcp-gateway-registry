/**
 * ObservabilityPipeline - L3 construct that creates the full observability stack:
 *
 *   1. Amazon Managed Prometheus (AMP) workspace
 *   2. Metrics-service ECS Fargate service with ADOT sidecar
 *   3. Grafana OSS ECS Fargate service with ALB path-based routing
 *
 * The construct is a no-op when `config.enableObservability` is false.
 *
 * Translates: terraform/aws-ecs/modules/mcp-gateway/observability.tf
 */

import * as cdk from 'aws-cdk-lib';
import * as aps from 'aws-cdk-lib/aws-aps';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';

import { RegistryConfig } from '../registry-config';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ObservabilityPipelineProps {
  /** Fully resolved registry configuration */
  readonly config: RegistryConfig;
  /** VPC in which all resources are placed */
  readonly vpc: ec2.IVpc;
  /** Private subnets for Fargate task placement */
  readonly privateSubnets: ec2.ISubnet[];
  /** ECS cluster shared by all services */
  readonly ecsCluster: ecs.ICluster;
  /** Cloud Map namespace ARN for Service Connect */
  readonly serviceConnectNamespaceArn: string;
  /** Application Load Balancer for Grafana listener rules */
  readonly alb: elbv2.IApplicationLoadBalancer;
  /** HTTP listener on the ALB (port 80) */
  readonly httpListener: elbv2.IApplicationListener;
  /** HTTPS listener on the ALB (port 443) - undefined when no certificate */
  readonly httpsListener?: elbv2.IApplicationListener;
  /** KMS key used for encrypting application secrets */
  readonly appSecretsKmsKey: kms.IKey;
  /** Secrets Manager secret holding the metrics API key */
  readonly metricsApiKeySecret?: secretsmanager.ISecret;
  /** Secrets Manager secret holding the metrics API-key HMAC pepper */
  readonly metricsKeyPepperSecret?: secretsmanager.ISecret;
  /** Secrets Manager secret holding the OTLP exporter headers */
  readonly otlpExporterHeadersSecret?: secretsmanager.ISecret;
  /** Secrets Manager secret holding the Grafana admin password (issue #1325) */
  readonly grafanaAdminPasswordSecret?: secretsmanager.ISecret;
  /** Additional IAM policy statements for Secrets Manager + KMS access */
  readonly secretsAccessStatements: iam.PolicyStatement[];
  /** Security group of the registry ECS service (for metrics ingress) */
  readonly registryServiceSg: ec2.ISecurityGroup;
  /** Security group of the auth-server ECS service (for metrics ingress) */
  readonly authServiceSg: ec2.ISecurityGroup;
  /** Security group attached to the ALB (for Grafana ingress) */
  readonly albSg: ec2.ISecurityGroup;
  /** Deployment name prefix for resource naming */
  readonly namePrefix: string;
}

// ---------------------------------------------------------------------------
// Construct
// ---------------------------------------------------------------------------

export class ObservabilityPipeline extends Construct {
  /** AMP workspace ID (undefined when observability is disabled) */
  public readonly ampWorkspaceId?: string;

  /** AMP Prometheus endpoint URL (undefined when observability is disabled) */
  public readonly ampEndpoint?: string;

  /** Grafana dashboard URL path (undefined when observability is disabled) */
  public readonly grafanaUrl?: string;

  /** Security group for the metrics-service ECS tasks (undefined when disabled) */
  public readonly metricsServiceSg?: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: ObservabilityPipelineProps) {
    super(scope, id);

    const { config } = props;

    // No-op when observability is disabled
    if (!config.enableObservability) {
      return;
    }

    const region = cdk.Stack.of(this).region;
    const namePrefix = props.namePrefix;

    // ================================================================
    // Section 1: Amazon Managed Prometheus (AMP)
    // ================================================================

    const ampWorkspace = new aps.CfnWorkspace(this, 'AmpWorkspace', {
      alias: `${namePrefix}-amp`,
      tags: [
        { key: 'Project', value: 'mcp-gateway-registry' },
        { key: 'Component', value: 'observability' },
      ],
    });

    this.ampWorkspaceId = ampWorkspace.attrWorkspaceId;
    this.ampEndpoint = ampWorkspace.attrPrometheusEndpoint;

    const ampRemoteWriteEndpoint = `${ampWorkspace.attrPrometheusEndpoint}api/v1/remote_write`;

    // ================================================================
    // Section 2: ADOT collector YAML configuration
    // ================================================================

    const adotConfig = JSON.stringify({
      receivers: {
        prometheus: {
          config: {
            global: {
              scrape_interval: '15s',
            },
            scrape_configs: [
              {
                job_name: 'mcp-metrics-service',
                scrape_interval: '15s',
                metrics_path: '/metrics',
                static_configs: [
                  {
                    targets: ['localhost:9465'],
                  },
                ],
              },
            ],
          },
        },
      },
      exporters: {
        prometheusremotewrite: {
          endpoint: ampRemoteWriteEndpoint,
          auth: {
            authenticator: 'sigv4auth',
          },
        },
      },
      extensions: {
        sigv4auth: {
          region,
        },
        health_check: {
          endpoint: '0.0.0.0:13133',
        },
      },
      service: {
        extensions: ['sigv4auth', 'health_check'],
        pipelines: {
          metrics: {
            receivers: ['prometheus'],
            exporters: ['prometheusremotewrite'],
          },
        },
      },
    });

    // ================================================================
    // Section 3: Metrics-Service ECS (with ADOT sidecar)
    // ================================================================

    // --- IAM policy: AMP remote write ---

    const ampRemoteWritePolicy = new iam.ManagedPolicy(this, 'AmpRemoteWritePolicy', {
      statements: [
        new iam.PolicyStatement({
          sid: 'AmpRemoteWrite',
          effect: iam.Effect.ALLOW,
          actions: [
            'aps:RemoteWrite',
            'aps:GetSeries',
            'aps:GetLabels',
            'aps:GetMetricMetadata',
          ],
          resources: [ampWorkspace.attrArn],
        }),
      ],
    });

    // --- Log groups ---

    const metricsLogGroup = new logs.LogGroup(this, 'MetricsLogGroup', {
      logGroupName: `/ecs/${namePrefix}-metrics-service`,
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const adotLogGroup = new logs.LogGroup(this, 'AdotLogGroup', {
      logGroupName: `/ecs/${namePrefix}-adot-collector`,
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // --- IAM roles ---

    const metricsExecRole = _createTaskExecRole(
      this,
      'MetricsExecRole',
      `${namePrefix}-metrics-service-task-exec-${region}`,
      metricsLogGroup,
      props.secretsAccessStatements,
    );

    const metricsTaskRole = _createTaskRole(
      this,
      'MetricsTaskRole',
      `${namePrefix}-metrics-service-task-${region}`,
      [ampRemoteWritePolicy],
    );

    // --- Security group ---

    this.metricsServiceSg = new ec2.SecurityGroup(this, 'MetricsSg', {
      vpc: props.vpc,
      securityGroupName: `${namePrefix}-metrics-service-ecs`,
      description: 'Security group for metrics-service ECS tasks',
      allowAllOutbound: true,
    });

    cdk.Tags.of(this.metricsServiceSg).add('Name', `${namePrefix}-metrics-service-ecs`);

    this.metricsServiceSg.addIngressRule(
      props.registryServiceSg,
      ec2.Port.tcp(8890),
      'Metrics API from registry',
    );
    this.metricsServiceSg.addIngressRule(
      props.authServiceSg,
      ec2.Port.tcp(8890),
      'Metrics API from auth-server',
    );

    // --- Task definition ---

    const metricsTaskDef = new ecs.FargateTaskDefinition(this, 'MetricsTaskDef', {
      family: `${namePrefix}-metrics-service`,
      cpu: 512,
      memoryLimitMiB: 1024,
      executionRole: metricsExecRole,
      taskRole: metricsTaskRole,
    });

    // --- Metrics-service container ---

    const metricsEnv: Record<string, string> = {
      METRICS_SERVICE_HOST: '0.0.0.0',
      METRICS_SERVICE_PORT: '8890',
      OTEL_SERVICE_NAME: 'mcp-metrics-service',
      OTEL_PROMETHEUS_ENABLED: 'true',
      OTEL_PROMETHEUS_PORT: '9465',
      METRICS_RATE_LIMIT: '1000',
      HISTOGRAM_BUCKET_BOUNDARIES:
        '0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0,30.0,60.0,120.0,300.0',
      SQLITE_DB_PATH: '/tmp/metrics.db',
      METRICS_RETENTION_DAYS: '7',
      OTEL_OTLP_ENDPOINT: config.otel.otlpEndpoint,
      OTEL_OTLP_EXPORT_INTERVAL_MS: String(config.otel.otlpExportIntervalMs),
      OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE:
        config.otel.exporterOtlpMetricsTemporalityPreference,
    };

    const metricsSecrets: Record<string, ecs.Secret> = {};
    if (props.metricsApiKeySecret) {
      metricsSecrets['METRICS_API_KEY_REGISTRY'] = ecs.Secret.fromSecretsManager(
        props.metricsApiKeySecret,
      );
      metricsSecrets['METRICS_API_KEY_AUTH'] = ecs.Secret.fromSecretsManager(
        props.metricsApiKeySecret,
      );
      metricsSecrets['METRICS_API_KEY_MCPGW'] = ecs.Secret.fromSecretsManager(
        props.metricsApiKeySecret,
      );
    }
    if (props.metricsKeyPepperSecret) {
      // Required: metrics-service refuses to start without a strong pepper.
      metricsSecrets['METRICS_KEY_PEPPER'] = ecs.Secret.fromSecretsManager(
        props.metricsKeyPepperSecret,
      );
    }
    if (props.otlpExporterHeadersSecret && config.otel.otlpEndpoint !== '') {
      metricsSecrets['OTEL_EXPORTER_OTLP_HEADERS'] = ecs.Secret.fromSecretsManager(
        props.otlpExporterHeadersSecret,
      );
    }

    const metricsContainer = metricsTaskDef.addContainer('metrics-service', {
      containerName: 'metrics-service',
      image: ecs.ContainerImage.fromRegistry(config.images.metricsService),
      essential: true,
      cpu: 256,
      memoryLimitMiB: 512,
      environment: metricsEnv,
      secrets: metricsSecrets,
      logging: ecs.LogDrivers.awsLogs({
        logGroup: metricsLogGroup,
        streamPrefix: 'ecs',
      }),
      healthCheck: {
        command: ['CMD-SHELL', 'curl -f http://localhost:8890/health || exit 1'],
        interval: cdk.Duration.seconds(30),
        timeout: cdk.Duration.seconds(5),
        retries: 3,
        startPeriod: cdk.Duration.seconds(30),
      },
      readonlyRootFilesystem: false,
    });

    metricsContainer.addPortMappings(
      {
        containerPort: 8890,
        hostPort: 8890,
        protocol: ecs.Protocol.TCP,
        name: 'metrics-api',
      },
      {
        containerPort: 9465,
        hostPort: 9465,
        protocol: ecs.Protocol.TCP,
        name: 'prometheus-exporter',
      },
    );

    // --- ADOT sidecar container ---

    const adotContainer = metricsTaskDef.addContainer('adot-collector', {
      containerName: 'adot-collector',
      image: ecs.ContainerImage.fromRegistry(
        'public.ecr.aws/aws-observability/aws-otel-collector:latest',
      ),
      essential: false,
      cpu: 256,
      memoryLimitMiB: 512,
      command: ['--config=env:AOT_CONFIG_CONTENT'],
      environment: {
        AOT_CONFIG_CONTENT: adotConfig,
        AWS_REGION: region,
      },
      logging: ecs.LogDrivers.awsLogs({
        logGroup: adotLogGroup,
        streamPrefix: 'ecs',
      }),
      readonlyRootFilesystem: false,
    });

    adotContainer.addContainerDependencies({
      container: metricsContainer,
      condition: ecs.ContainerDependencyCondition.HEALTHY,
    });

    // --- Fargate service ---

    const metricsService = new ecs.FargateService(this, 'MetricsService', {
      serviceName: `${namePrefix}-metrics-service`,
      cluster: props.ecsCluster,
      taskDefinition: metricsTaskDef,
      desiredCount: 1,
      assignPublicIp: false,
      vpcSubnets: { subnets: props.privateSubnets },
      securityGroups: [this.metricsServiceSg],
      enableExecuteCommand: true,
      serviceConnectConfiguration: {
        namespace: props.serviceConnectNamespaceArn,
        services: [
          {
            portMappingName: 'metrics-api',
            dnsName: 'metrics-service',
            discoveryName: 'metrics-service',
            port: 8890,
          },
        ],
      },
    });

    // ================================================================
    // Section 4: Grafana OSS ECS Service
    // ================================================================

    // --- IAM policy: AMP query ---

    const ampQueryPolicy = new iam.ManagedPolicy(this, 'AmpQueryPolicy', {
      statements: [
        new iam.PolicyStatement({
          sid: 'AmpQuery',
          effect: iam.Effect.ALLOW,
          actions: [
            'aps:QueryMetrics',
            'aps:GetSeries',
            'aps:GetLabels',
            'aps:GetMetricMetadata',
          ],
          resources: [ampWorkspace.attrArn],
        }),
      ],
    });

    // --- Log group ---

    const grafanaLogGroup = new logs.LogGroup(this, 'GrafanaLogGroup', {
      logGroupName: `/ecs/${namePrefix}-grafana`,
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // --- IAM roles ---

    const grafanaExecRole = _createTaskExecRole(
      this,
      'GrafanaExecRole',
      `${namePrefix}-grafana-task-exec-${region}`,
      grafanaLogGroup,
      // Issue #1325: the execution role must read the Grafana admin password
      // secret (and KMS-decrypt it) for the container `secrets` valueFrom.
      props.secretsAccessStatements,
    );

    const grafanaTaskRole = _createTaskRole(
      this,
      'GrafanaTaskRole',
      `${namePrefix}-grafana-task-${region}`,
      [ampQueryPolicy],
    );

    // --- Security group ---

    const grafanaSg = new ec2.SecurityGroup(this, 'GrafanaSg', {
      vpc: props.vpc,
      securityGroupName: `${namePrefix}-grafana-ecs`,
      description: 'Security group for Grafana ECS tasks',
      allowAllOutbound: true,
    });

    cdk.Tags.of(grafanaSg).add('Name', `${namePrefix}-grafana-ecs`);

    grafanaSg.addIngressRule(
      props.albSg,
      ec2.Port.tcp(3000),
      'Grafana HTTP from ALB',
    );

    // --- Task definition ---

    const grafanaTaskDef = new ecs.FargateTaskDefinition(this, 'GrafanaTaskDef', {
      family: `${namePrefix}-grafana`,
      cpu: 512,
      memoryLimitMiB: 1024,
      executionRole: grafanaExecRole,
      taskRole: grafanaTaskRole,
    });

    // --- Grafana container ---

    const grafanaEnv: Record<string, string> = {
      AWS_REGION: region,
      GF_AUTH_SIGV4_AUTH_ENABLED: 'true',
      GF_AWS_ALLOWED_AUTH_PROVIDERS: 'default,ec2_iam_role',
      AMP_ENDPOINT: ampWorkspace.attrPrometheusEndpoint,
      GF_SERVER_ROOT_URL: '%(protocol)s://%(domain)s/grafana/',
      GF_SERVER_SERVE_FROM_SUB_PATH: 'true',
      GF_AUTH_ANONYMOUS_ENABLED: 'false',
      GF_AUTH_ANONYMOUS_ORG_ROLE: 'Viewer',
      GF_AUTH_DISABLE_LOGIN_FORM: 'false',
      GF_LOG_MODE: 'console',
      GF_LOG_LEVEL: 'info',
      GF_DASHBOARDS_MIN_REFRESH_INTERVAL: '10s',
    };

    // Issue #1325: source GF_SECURITY_ADMIN_PASSWORD from Secrets Manager via
    // the container `secrets` map instead of a plaintext env value, so it does
    // not appear in the rendered task definition. Falls back to the plaintext
    // env only if the secret was not provided (observability disabled paths).
    const grafanaSecrets: Record<string, ecs.Secret> = {};
    if (props.grafanaAdminPasswordSecret) {
      grafanaSecrets['GF_SECURITY_ADMIN_PASSWORD'] = ecs.Secret.fromSecretsManager(
        props.grafanaAdminPasswordSecret,
      );
    } else {
      grafanaEnv['GF_SECURITY_ADMIN_PASSWORD'] = config.grafanaAdminPassword;
    }

    const grafanaContainer = grafanaTaskDef.addContainer('grafana', {
      containerName: 'grafana',
      image: ecs.ContainerImage.fromRegistry(config.images.grafana),
      essential: true,
      cpu: 512,
      memoryLimitMiB: 1024,
      secrets: grafanaSecrets,
      environment: grafanaEnv,
      logging: ecs.LogDrivers.awsLogs({
        logGroup: grafanaLogGroup,
        streamPrefix: 'ecs',
      }),
      healthCheck: {
        command: ['CMD-SHELL', 'wget -q --spider http://localhost:3000/api/health || exit 1'],
        interval: cdk.Duration.seconds(30),
        timeout: cdk.Duration.seconds(5),
        retries: 3,
        startPeriod: cdk.Duration.seconds(30),
      },
      readonlyRootFilesystem: false,
    });

    grafanaContainer.addPortMappings({
      containerPort: 3000,
      hostPort: 3000,
      protocol: ecs.Protocol.TCP,
      name: 'grafana-http',
    });

    // --- ALB target group ---

    const grafanaTg = new elbv2.ApplicationTargetGroup(this, 'GrafanaTg', {
      targetGroupName: `${namePrefix}-grafana-tg`.substring(0, 32),
      port: 3000,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targetType: elbv2.TargetType.IP,
      vpc: props.vpc,
      deregistrationDelay: cdk.Duration.seconds(5),
      healthCheck: {
        enabled: true,
        healthyThresholdCount: 2,
        unhealthyThresholdCount: 2,
        timeout: cdk.Duration.seconds(5),
        interval: cdk.Duration.seconds(30),
        path: '/api/health',
        healthyHttpCodes: '200',
        protocol: elbv2.Protocol.HTTP,
      },
    });

    // --- ALB listener rules: /grafana and /grafana/* ---

    new elbv2.ApplicationListenerRule(this, 'GrafanaHttpRule', {
      listener: props.httpListener,
      priority: 15,
      conditions: [
        elbv2.ListenerCondition.pathPatterns(['/grafana', '/grafana/*']),
      ],
      targetGroups: [grafanaTg],
    });

    if (props.httpsListener) {
      new elbv2.ApplicationListenerRule(this, 'GrafanaHttpsRule', {
        listener: props.httpsListener,
        priority: 15,
        conditions: [
          elbv2.ListenerCondition.pathPatterns(['/grafana', '/grafana/*']),
        ],
        targetGroups: [grafanaTg],
      });
    }

    // --- Fargate service ---

    const grafanaService = new ecs.FargateService(this, 'GrafanaService', {
      serviceName: `${namePrefix}-grafana`,
      cluster: props.ecsCluster,
      taskDefinition: grafanaTaskDef,
      desiredCount: 1,
      assignPublicIp: false,
      vpcSubnets: { subnets: props.privateSubnets },
      securityGroups: [grafanaSg],
      enableExecuteCommand: true,
      serviceConnectConfiguration: {
        namespace: props.serviceConnectNamespaceArn,
        services: [
          {
            portMappingName: 'grafana-http',
            dnsName: 'grafana',
            discoveryName: 'grafana',
            port: 3000,
          },
        ],
      },
    });

    grafanaService.attachToApplicationTargetGroup(grafanaTg);

    this.grafanaUrl = '/grafana/';

    // ================================================================
    // Section 5: Tags
    // ================================================================

    cdk.Tags.of(this).add('Project', 'mcp-gateway-registry');
    cdk.Tags.of(this).add('Component', 'observability');
  }
}


// ===========================================================================
// Private helper functions
// ===========================================================================


/**
 * Create a standard ECS task execution role with CloudWatch Logs and
 * SSM Messages permissions plus optional additional statements.
 */
function _createTaskExecRole(
  scope: Construct,
  id: string,
  roleName: string,
  logGroup: logs.ILogGroup,
  additionalStatements: iam.PolicyStatement[],
): iam.Role {
  const role = new iam.Role(scope, id, {
    roleName,
    assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
    managedPolicies: [
      iam.ManagedPolicy.fromAwsManagedPolicyName(
        'service-role/AmazonECSTaskExecutionRolePolicy',
      ),
    ],
  });

  role.addToPolicy(
    new iam.PolicyStatement({
      sid: 'CloudWatchLogs',
      effect: iam.Effect.ALLOW,
      actions: ['logs:CreateLogStream', 'logs:PutLogEvents'],
      resources: [`${logGroup.logGroupArn}:*`],
    }),
  );

  role.addToPolicy(
    new iam.PolicyStatement({
      sid: 'SSMMessages',
      effect: iam.Effect.ALLOW,
      actions: [
        'ssmmessages:CreateControlChannel',
        'ssmmessages:CreateDataChannel',
        'ssmmessages:OpenControlChannel',
        'ssmmessages:OpenDataChannel',
      ],
      resources: ['*'],
    }),
  );

  for (const stmt of additionalStatements) {
    role.addToPolicy(stmt);
  }

  return role;
}


/**
 * Create a standard ECS task role with SSM Session Manager permissions
 * plus optional managed policies.
 */
function _createTaskRole(
  scope: Construct,
  id: string,
  roleName: string,
  additionalPolicies: iam.IManagedPolicy[],
): iam.Role {
  const role = new iam.Role(scope, id, {
    roleName,
    assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
  });

  role.addToPolicy(
    new iam.PolicyStatement({
      sid: 'SSMSessionManager',
      effect: iam.Effect.ALLOW,
      actions: [
        'ssmmessages:CreateControlChannel',
        'ssmmessages:CreateDataChannel',
        'ssmmessages:OpenControlChannel',
        'ssmmessages:OpenDataChannel',
      ],
      resources: ['*'],
    }),
  );

  for (const policy of additionalPolicies) {
    role.addManagedPolicy(policy);
  }

  return role;
}
