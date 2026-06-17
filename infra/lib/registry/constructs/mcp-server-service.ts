/**
 * McpServerService - L3 construct for optional MCP server / A2A agent ECS
 * services. A service is only created when its container image URI is non-empty.
 *
 * Handles: currenttime, mcpgw, realserverfaketools, flight-booking-agent,
 * travel-assistant-agent.
 *
 * Each service gets its own Fargate task definition, security group, Cloud Map
 * Service Connect entry, and optional EFS volume mounts.
 */

import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import { Construct } from 'constructs';

// ---------------------------------------------------------------------------
// EFS volume mount (simplified - reused from registry-ecs-service)
// ---------------------------------------------------------------------------

export interface McpEfsVolumeConfig {
  readonly volumeName: string;
  readonly fileSystemId: string;
  readonly accessPointId: string;
  readonly containerPath: string;
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface McpServerServiceProps {
  /** Short service name (e.g. 'currenttime', 'mcpgw') */
  readonly serviceName: string;
  /** Container image URI - service is only created if non-empty */
  readonly imageUri: string;
  /** Container port (e.g. 8000, 8003) */
  readonly containerPort: number;
  /** VPC in which the service runs */
  readonly vpc: ec2.IVpc;
  /** Private subnets for task placement */
  readonly subnets: ec2.ISubnet[];
  /** ECS cluster */
  readonly cluster: ecs.ICluster;
  /** Cloud Map namespace ARN for Service Connect */
  readonly serviceConnectNamespaceArn: string;
  /** Service Connect DNS name (e.g. 'currenttime-server') */
  readonly serviceConnectDnsName: string;
  /** Service Connect port name (e.g. 'currenttime') */
  readonly serviceConnectPortName: string;
  /** Environment variables for the container */
  readonly environment: Record<string, string>;
  /** EFS volumes (optional, e.g. mcpgw uses EFS) */
  readonly efsVolumes?: McpEfsVolumeConfig[];
  /** Security group to allow ingress from (typically registry ECS SG) */
  readonly ingressSecurityGroup?: ec2.ISecurityGroup;
  /** VPC CIDR for ingress (used by agents instead of SG ref) */
  readonly ingressCidr?: string;
  /** Health check command override */
  readonly healthCheckCommand?: string;
  /** Desired task count (defaults to 1) */
  readonly desiredCount?: number;
  /** Deployment name prefix */
  readonly namePrefix: string;
  /** Additional exec role policy statements (e.g. Secrets Manager) */
  readonly additionalExecRoleStatements?: iam.PolicyStatement[];
  /** Additional task role policy statements */
  readonly additionalTaskRoleStatements?: iam.PolicyStatement[];
}

// ---------------------------------------------------------------------------
// Construct
// ---------------------------------------------------------------------------

export class McpServerService extends Construct {
  /** The ECS Fargate service (undefined if imageUri was empty) */
  public readonly service?: ecs.FargateService;

  /** Security group for the ECS tasks (undefined if not created) */
  public readonly securityGroup?: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: McpServerServiceProps) {
    super(scope, id);

    // Skip creation if image URI is empty
    if (!props.imageUri || props.imageUri === '') {
      return;
    }

    const region = cdk.Stack.of(this).region;

    // ------------------------------------------------------------------
    // CloudWatch Log Group
    // ------------------------------------------------------------------

    const logGroup = new logs.LogGroup(this, 'LogGroup', {
      logGroupName: `/ecs/${props.namePrefix}-${props.serviceName}`,
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // ------------------------------------------------------------------
    // IAM - Task Execution Role
    // ------------------------------------------------------------------

    const taskExecRole = new iam.Role(this, 'TaskExecRole', {
      roleName: `${props.namePrefix}-${props.serviceName}-task-exec-${region}`,
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          'service-role/AmazonECSTaskExecutionRolePolicy',
        ),
      ],
    });

    taskExecRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'CloudWatchLogs',
        effect: iam.Effect.ALLOW,
        actions: ['logs:CreateLogStream', 'logs:PutLogEvents'],
        resources: [`${logGroup.logGroupArn}:*`],
      }),
    );

    taskExecRole.addToPolicy(
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

    if (props.additionalExecRoleStatements) {
      for (const stmt of props.additionalExecRoleStatements) {
        taskExecRole.addToPolicy(stmt);
      }
    }

    // ------------------------------------------------------------------
    // IAM - Task Role
    // ------------------------------------------------------------------

    const taskRole = new iam.Role(this, 'TaskRole', {
      roleName: `${props.namePrefix}-${props.serviceName}-task-${region}`,
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
    });

    taskRole.addToPolicy(
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

    if (props.additionalTaskRoleStatements) {
      for (const stmt of props.additionalTaskRoleStatements) {
        taskRole.addToPolicy(stmt);
      }
    }

    // ------------------------------------------------------------------
    // Security Group
    // ------------------------------------------------------------------

    const sg = new ec2.SecurityGroup(this, 'Sg', {
      vpc: props.vpc,
      securityGroupName: `${props.namePrefix}-${props.serviceName}-ecs`,
      description: `Security group for ${props.serviceName} ECS tasks`,
      allowAllOutbound: true,
    });

    cdk.Tags.of(sg).add('Name', `${props.namePrefix}-${props.serviceName}-ecs`);

    // Add ingress from registry SG or VPC CIDR
    if (props.ingressSecurityGroup) {
      sg.addIngressRule(
        props.ingressSecurityGroup,
        ec2.Port.tcp(props.containerPort),
        `Service Connect from registry to ${props.serviceName}`,
      );
    } else if (props.ingressCidr) {
      sg.addIngressRule(
        ec2.Peer.ipv4(props.ingressCidr),
        ec2.Port.tcp(props.containerPort),
        `Service Connect from VPC to ${props.serviceName}`,
      );
    }

    this.securityGroup = sg;

    // ------------------------------------------------------------------
    // Task Definition
    // ------------------------------------------------------------------

    const taskDef = new ecs.FargateTaskDefinition(this, 'TaskDef', {
      family: `${props.namePrefix}-${props.serviceName}`,
      cpu: 512,
      memoryLimitMiB: 1024,
      executionRole: taskExecRole,
      taskRole,
    });

    // Add EFS volumes
    if (props.efsVolumes) {
      for (const vol of props.efsVolumes) {
        taskDef.addVolume({
          name: vol.volumeName,
          efsVolumeConfiguration: {
            fileSystemId: vol.fileSystemId,
            transitEncryption: 'ENABLED',
            authorizationConfig: {
              accessPointId: vol.accessPointId,
            },
          },
        });
      }
    }

    // ------------------------------------------------------------------
    // Container
    // ------------------------------------------------------------------

    const healthCmd = props.healthCheckCommand
      ?? `nc -z localhost ${props.containerPort} || exit 1`;

    const container = taskDef.addContainer(props.serviceName, {
      containerName: props.serviceName,
      image: ecs.ContainerImage.fromRegistry(props.imageUri),
      essential: true,
      cpu: 512,
      memoryLimitMiB: 1024,
      environment: props.environment,
      logging: ecs.LogDrivers.awsLogs({
        logGroup,
        streamPrefix: 'ecs',
      }),
      healthCheck: {
        command: ['CMD-SHELL', healthCmd],
        interval: cdk.Duration.seconds(30),
        timeout: cdk.Duration.seconds(5),
        retries: 3,
        startPeriod: cdk.Duration.seconds(30),
      },
      readonlyRootFilesystem: false,
    });

    container.addPortMappings({
      containerPort: props.containerPort,
      hostPort: props.containerPort,
      protocol: ecs.Protocol.TCP,
      name: props.serviceConnectPortName,
    });

    // EFS mount points
    if (props.efsVolumes) {
      for (const vol of props.efsVolumes) {
        container.addMountPoints({
          sourceVolume: vol.volumeName,
          containerPath: vol.containerPath,
          readOnly: false,
        });
      }
    }

    // ------------------------------------------------------------------
    // Fargate Service
    // ------------------------------------------------------------------

    this.service = new ecs.FargateService(this, 'Service', {
      serviceName: `${props.namePrefix}-${props.serviceName}`,
      cluster: props.cluster,
      taskDefinition: taskDef,
      desiredCount: props.desiredCount ?? 1,
      assignPublicIp: false,
      vpcSubnets: { subnets: props.subnets },
      securityGroups: [sg],
      enableExecuteCommand: true,
      serviceConnectConfiguration: {
        namespace: props.serviceConnectNamespaceArn,
        services: [
          {
            portMappingName: props.serviceConnectPortName,
            dnsName: props.serviceConnectDnsName,
            discoveryName: props.serviceConnectDnsName,
            port: props.containerPort,
          },
        ],
      },
    });

    // ------------------------------------------------------------------
    // Auto Scaling (min 1, max 4, CPU 70%)
    // ------------------------------------------------------------------

    const scaling = this.service.autoScaleTaskCount({
      minCapacity: 1,
      maxCapacity: 4,
    });

    scaling.scaleOnCpuUtilization('CpuScaling', {
      targetUtilizationPercent: 70,
    });

    // ------------------------------------------------------------------
    // Tags
    // ------------------------------------------------------------------

    cdk.Tags.of(this).add('Service', props.serviceName);
  }
}
