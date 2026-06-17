/**
 * RegistryEcsService - Reusable L3 construct for creating a single ECS Fargate
 * service with standard configuration, IAM roles, logging, and autoscaling.
 *
 * Used by RegistryServiceStack for the registry and auth-server core services.
 */

import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import { Construct } from 'constructs';

// ---------------------------------------------------------------------------
// EFS volume mount configuration
// ---------------------------------------------------------------------------

export interface EfsVolumeConfig {
  /** Logical volume name used in task definition and mountPoints */
  readonly volumeName: string;
  /** EFS file system ID */
  readonly fileSystemId: string;
  /** EFS access point ID */
  readonly accessPointId: string;
  /** Container-side mount path (e.g. /app/logs) */
  readonly containerPath: string;
  /** Whether the mount is read-only */
  readonly readOnly?: boolean;
}

// ---------------------------------------------------------------------------
// Service Connect configuration
// ---------------------------------------------------------------------------

export interface ServiceConnectEntry {
  /** Port exposed by Service Connect */
  readonly port: number;
  /** DNS name other services use to reach this service */
  readonly dnsName: string;
  /** Named port used in portMappings */
  readonly portName: string;
  /** Discovery name in Cloud Map */
  readonly discoveryName: string;
}

// ---------------------------------------------------------------------------
// Construct props
// ---------------------------------------------------------------------------

export interface RegistryEcsServiceProps {
  /** Human-readable service name (e.g. 'registry', 'auth-server') */
  readonly serviceName: string;
  /** Container image URI (or ECR repo:tag) */
  readonly image: string;
  /** Task CPU units (e.g. 1024) */
  readonly cpu: number;
  /** Task memory in MiB (e.g. 2048) */
  readonly memory: number;
  /** Primary container port */
  readonly containerPort: number;
  /** Additional container ports (optional) */
  readonly additionalPorts?: Array<{ port: number; name: string }>;
  /** VPC in which the service runs */
  readonly vpc: ec2.IVpc;
  /** Private subnets for task placement */
  readonly subnets: ec2.ISubnet[];
  /** ECS cluster */
  readonly cluster: ecs.ICluster;
  /** Cloud Map namespace ARN for Service Connect */
  readonly serviceConnectNamespaceArn: string;
  /** Service Connect configuration */
  readonly serviceConnect: ServiceConnectEntry;
  /** Environment variables for the container */
  readonly environment: Record<string, string>;
  /** Secrets (ECS secret references) */
  readonly secrets?: Record<string, ecs.Secret>;
  /** EFS volume mounts */
  readonly efsVolumes?: EfsVolumeConfig[];
  /** ALB target groups to attach (optional) */
  readonly targetGroups?: Array<{
    targetGroup: elbv2.IApplicationTargetGroup;
    containerPort: number;
  }>;
  /** Additional managed policy ARNs for the task role */
  readonly additionalTaskRolePolicies?: iam.IManagedPolicy[];
  /** Additional inline policy statements for the task execution role */
  readonly additionalExecRoleStatements?: iam.PolicyStatement[];
  /** Security groups to allow ingress from (port = containerPort) */
  readonly ingressSources?: Array<{
    peer: ec2.ISecurityGroup;
    port: number;
    description: string;
  }>;
  /** Desired task count (defaults to 1) */
  readonly desiredCount?: number;
  /** Health check command override */
  readonly healthCheckCommand?: string;
  /** Health check start period in seconds (default 60) */
  readonly healthCheckStartPeriod?: number;
  /** Deployment name prefix for resource naming */
  readonly namePrefix: string;
}

// ---------------------------------------------------------------------------
// Construct
// ---------------------------------------------------------------------------

export class RegistryEcsService extends Construct {
  /** The ECS Fargate service */
  public readonly service: ecs.FargateService;

  /** The Fargate task definition */
  public readonly taskDefinition: ecs.FargateTaskDefinition;

  /** Security group attached to the ECS tasks */
  public readonly securityGroup: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: RegistryEcsServiceProps) {
    super(scope, id);

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

    // CloudWatch logs permission
    taskExecRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'CloudWatchLogs',
        effect: iam.Effect.ALLOW,
        actions: ['logs:CreateLogStream', 'logs:PutLogEvents'],
        resources: [`${logGroup.logGroupArn}:*`],
      }),
    );

    // SSM messages for ECS Exec
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

    // Additional exec role statements (e.g. Secrets Manager, KMS)
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

    // SSM Session Manager (ECS Exec)
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

    // Additional task role managed policies (e.g. Bedrock AgentCore)
    if (props.additionalTaskRolePolicies) {
      for (const policy of props.additionalTaskRolePolicies) {
        taskRole.addManagedPolicy(policy);
      }
    }

    // ------------------------------------------------------------------
    // Security Group
    // ------------------------------------------------------------------

    this.securityGroup = new ec2.SecurityGroup(this, 'Sg', {
      vpc: props.vpc,
      securityGroupName: `${props.namePrefix}-${props.serviceName}-ecs`,
      description: `Security group for ${props.serviceName} ECS tasks`,
      allowAllOutbound: true,
    });

    cdk.Tags.of(this.securityGroup).add('Name', `${props.namePrefix}-${props.serviceName}-ecs`);

    // Add ingress rules from specified sources
    if (props.ingressSources) {
      for (const source of props.ingressSources) {
        this.securityGroup.addIngressRule(
          source.peer,
          ec2.Port.tcp(source.port),
          source.description,
        );
      }
    }

    // ------------------------------------------------------------------
    // Task Definition
    // ------------------------------------------------------------------

    this.taskDefinition = new ecs.FargateTaskDefinition(this, 'TaskDef', {
      family: `${props.namePrefix}-${props.serviceName}`,
      cpu: props.cpu,
      memoryLimitMiB: props.memory,
      executionRole: taskExecRole,
      taskRole,
    });

    // Add EFS volumes
    if (props.efsVolumes) {
      for (const vol of props.efsVolumes) {
        this.taskDefinition.addVolume({
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
      ?? `curl -f http://localhost:${props.containerPort}/health || exit 1`;

    const container = this.taskDefinition.addContainer(props.serviceName, {
      containerName: props.serviceName,
      image: ecs.ContainerImage.fromRegistry(props.image),
      essential: true,
      environment: props.environment,
      secrets: props.secrets ?? {},
      logging: ecs.LogDrivers.awsLogs({
        logGroup,
        streamPrefix: 'ecs',
      }),
      healthCheck: {
        command: ['CMD-SHELL', healthCmd],
        interval: cdk.Duration.seconds(30),
        timeout: cdk.Duration.seconds(5),
        retries: 3,
        startPeriod: cdk.Duration.seconds(props.healthCheckStartPeriod ?? 60),
      },
      readonlyRootFilesystem: false,
    });

    // Primary port mapping
    container.addPortMappings({
      containerPort: props.containerPort,
      hostPort: props.containerPort,
      protocol: ecs.Protocol.TCP,
      name: props.serviceConnect.portName,
    });

    // Additional port mappings
    if (props.additionalPorts) {
      for (const p of props.additionalPorts) {
        container.addPortMappings({
          containerPort: p.port,
          hostPort: p.port,
          protocol: ecs.Protocol.TCP,
          name: p.name,
        });
      }
    }

    // EFS mount points
    if (props.efsVolumes) {
      for (const vol of props.efsVolumes) {
        container.addMountPoints({
          sourceVolume: vol.volumeName,
          containerPath: vol.containerPath,
          readOnly: vol.readOnly ?? false,
        });
      }
    }

    // ------------------------------------------------------------------
    // Fargate Service
    // ------------------------------------------------------------------

    this.service = new ecs.FargateService(this, 'Service', {
      serviceName: `${props.namePrefix}-${props.serviceName}`,
      cluster: props.cluster,
      taskDefinition: this.taskDefinition,
      desiredCount: props.desiredCount ?? 1,
      assignPublicIp: false,
      vpcSubnets: { subnets: props.subnets },
      securityGroups: [this.securityGroup],
      enableExecuteCommand: true,
      circuitBreaker: { enable: true, rollback: true },
      serviceConnectConfiguration: {
        namespace: props.serviceConnectNamespaceArn,
        services: [
          {
            portMappingName: props.serviceConnect.portName,
            dnsName: props.serviceConnect.dnsName,
            discoveryName: props.serviceConnect.discoveryName,
            port: props.serviceConnect.port,
          },
        ],
      },
    });

    // Attach to ALB target groups
    if (props.targetGroups) {
      for (const tg of props.targetGroups) {
        this.service.attachToApplicationTargetGroup(tg.targetGroup);
      }
    }

    // ------------------------------------------------------------------
    // Auto Scaling (min 1, max 4, CPU 70%, memory 80%)
    // ------------------------------------------------------------------

    const scaling = this.service.autoScaleTaskCount({
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

    cdk.Tags.of(this).add('Service', props.serviceName);
  }
}
