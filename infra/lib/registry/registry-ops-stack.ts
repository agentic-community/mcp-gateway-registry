/**
 * RegistryOpsStack - Secret rotation: custom Lambda for DocumentDB,
 * AWS-hosted rotator (mysqlSingleUser) for Keycloak Aurora MySQL.
 */

import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as path from 'path';
import { Construct } from 'constructs';
import { RegistryConfig } from './registry-config';
import { RegistryNetworkStack } from './registry-network-stack';
import { RegistryDataStack } from './registry-data-stack';
import { SecretRotation } from './constructs/secret-rotation';

// ---------------------------------------------------------------------------
// Stack props
// ---------------------------------------------------------------------------

export interface RegistryOpsStackProps extends cdk.StackProps {
  readonly config: RegistryConfig;
  readonly networkStack: RegistryNetworkStack;
  readonly dataStack: RegistryDataStack;
}

// ---------------------------------------------------------------------------
// Stack
// ---------------------------------------------------------------------------

export class RegistryOpsStack extends cdk.Stack {
  /** Security group for the rotation Lambda functions */
  public readonly rotationLambdaSg: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: RegistryOpsStackProps) {
    super(scope, id, props);

    const { config, networkStack, dataStack } = props;
    const { vpc, privateSubnets } = networkStack;
    const region = this.region;

    // Required for CfnRotationSchedule.hostedRotationLambda below.
    this.templateOptions.transforms = ['AWS::SecretsManager-2020-07-23'];

    // ==================================================================
    // IAM role for rotation Lambda functions
    // ==================================================================

    const rotationRole = _createRotationRole(this, config, dataStack, region);

    // ==================================================================
    // Security group for rotation Lambdas
    // ==================================================================

    this.rotationLambdaSg = new ec2.SecurityGroup(this, 'RotationLambdaSg', {
      vpc,
      securityGroupName: `${config.name}-rotation-lambda-sg`,
      description: 'Security group for secret rotation Lambda functions',
      allowAllOutbound: false,
    });

    cdk.Tags.of(this.rotationLambdaSg).add('Name', `${config.name}-rotation-lambda-sg`);
    cdk.Tags.of(this.rotationLambdaSg).add('Component', 'secrets-rotation');

    // ------------------------------------------------------------------
    // Egress rules
    // ------------------------------------------------------------------

    this.rotationLambdaSg.addEgressRule(
      ec2.Peer.securityGroupId(dataStack.documentDbSg.securityGroupId),
      ec2.Port.tcp(27017),
      'Lambda to DocumentDB',
    );
    this.rotationLambdaSg.addEgressRule(
      ec2.Peer.securityGroupId(dataStack.keycloakDbSg.securityGroupId),
      ec2.Port.tcp(3306),
      'Lambda to RDS',
    );
    this.rotationLambdaSg.addEgressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(443),
      'Lambda to AWS APIs (Secrets Manager, KMS)',
    );

    // ------------------------------------------------------------------
    // Ingress rules on database security groups
    // Uses CfnSecurityGroupIngress to avoid cross-stack cyclic dependency
    // (Ops depends on Data, so Data SG objects cannot reference Ops SG)
    // ------------------------------------------------------------------

    new ec2.CfnSecurityGroupIngress(this, 'DocDbFromLambda', {
      groupId: dataStack.documentDbSg.securityGroupId,
      ipProtocol: 'tcp', fromPort: 27017, toPort: 27017,
      sourceSecurityGroupId: this.rotationLambdaSg.securityGroupId,
      description: 'Lambda rotation to DocumentDB',
    });
    new ec2.CfnSecurityGroupIngress(this, 'RdsFromLambda', {
      groupId: dataStack.keycloakDbSg.securityGroupId,
      ipProtocol: 'tcp', fromPort: 3306, toPort: 3306,
      sourceSecurityGroupId: this.rotationLambdaSg.securityGroupId,
      description: 'Hosted rotator to RDS',
    });

    // ==================================================================
    // Lambda environment variables (shared by both functions)
    // ==================================================================

    const lambdaEnv: Record<string, string> = {
      SECRETS_MANAGER_ENDPOINT: `https://secretsmanager.${region}.amazonaws.com`,
      EXCLUDE_CHARACTERS: '/@"\'\\',
    };

    // ==================================================================
    // Lambda source code paths (reuse Terraform Lambda source)
    // ==================================================================

    const lambdaBasePath = path.join(
      __dirname, '..', '..', '..', 'terraform', 'aws-ecs', 'lambda',
    );

    // ==================================================================
    // DocumentDB rotation Lambda
    // ==================================================================

    const docdbRotation = new SecretRotation(this, 'DocumentDbRotation', {
      rotationName: `${config.name}-rotate-documentdb`,
      vpc,
      privateSubnets,
      lambdaSg: this.rotationLambdaSg,
      secretArn: dataStack.documentDbSecretArn,
      lambdaRole: rotationRole,
      lambdaCodePath: path.join(lambdaBasePath, 'rotate-documentdb'),
      environmentVariables: lambdaEnv,
    });

    // RDS Aurora MySQL: AWS-hosted rotator. Use the L1 directly in this stack
    // (addRotationSchedule attaches to the secret's stack and creates a Data
    // -> Ops cycle via the rotationLambdaSg reference).
    new secretsmanager.CfnRotationSchedule(this, 'RdsRotationSchedule', {
      secretId: dataStack.keycloakDbSecret.secretArn,
      hostedRotationLambda: {
        rotationType: 'MySQLSingleUser',
        vpcSecurityGroupIds: this.rotationLambdaSg.securityGroupId,
        vpcSubnetIds: privateSubnets.map((s) => s.subnetId).join(','),
      },
      rotationRules: { automaticallyAfterDays: 30 },
    });

    const docdbRotationSchedule = new secretsmanager.CfnRotationSchedule(
      this, 'DocumentDbRotationSchedule', {
        secretId: dataStack.documentDbSecretArn,
        rotationLambdaArn: docdbRotation.lambdaFunction.functionArn,
        rotationRules: { automaticallyAfterDays: 30 },
      },
    );
    docdbRotationSchedule.node.addDependency(docdbRotation.lambdaFunction);

    // ------------------------------------------------------------------
    // Common tags
    // ------------------------------------------------------------------

    cdk.Tags.of(this).add('Project', 'mcp-gateway-registry');
    cdk.Tags.of(this).add('Component', 'ops');
    cdk.Tags.of(this).add('Environment', 'production');
    cdk.Tags.of(this).add('ManagedBy', 'cdk');
  }
}

// ---------------------------------------------------------------------------
// Private helper: create the IAM role and policies for rotation Lambdas
// ---------------------------------------------------------------------------

function _createRotationRole(
  scope: Construct,
  config: RegistryConfig,
  dataStack: RegistryDataStack,
  region: string,
): iam.Role {
  const role = new iam.Role(scope, 'RotationLambdaRole', {
    roleName: `${config.name}-secret-rotation-lambda`,
    assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
  });

  // ---- Secrets Manager access (both secrets) ----
  role.addToPolicy(
    new iam.PolicyStatement({
      sid: 'SecretsManagerAccess',
      effect: iam.Effect.ALLOW,
      actions: [
        'secretsmanager:DescribeSecret',
        'secretsmanager:GetSecretValue',
        'secretsmanager:PutSecretValue',
        'secretsmanager:UpdateSecretVersionStage',
      ],
      resources: [dataStack.documentDbSecretArn],
    }),
  );

  // ---- Generate random password (requires wildcard resource) ----
  role.addToPolicy(
    new iam.PolicyStatement({
      sid: 'GenerateRandomPassword',
      effect: iam.Effect.ALLOW,
      actions: ['secretsmanager:GetRandomPassword'],
      resources: ['*'],
    }),
  );

  // ---- KMS access (DocumentDB key only — Aurora rotation is hosted) ----
  role.addToPolicy(
    new iam.PolicyStatement({
      sid: 'KMSAccess',
      effect: iam.Effect.ALLOW,
      actions: [
        'kms:Decrypt',
        'kms:DescribeKey',
        'kms:GenerateDataKey',
      ],
      resources: [dataStack.documentDbKmsKey.keyArn],
    }),
  );

  // ---- DocumentDB access ----
  // Build the ARN from the CfnDBCluster ref (cluster identifier)
  const documentDbClusterArn = cdk.Fn.sub(
    'arn:aws:rds:${AWS::Region}:${AWS::AccountId}:cluster:${ClusterId}',
    { ClusterId: dataStack.documentDbCluster.ref },
  );

  role.addToPolicy(
    new iam.PolicyStatement({
      sid: 'DocumentDBAccess',
      effect: iam.Effect.ALLOW,
      actions: [
        'docdb:DescribeDBClusters',
        'docdb:ModifyDBCluster',
      ],
      resources: [documentDbClusterArn],
    }),
  );

  // ---- VPC network interface management (requires wildcard resource) ----
  role.addToPolicy(
    new iam.PolicyStatement({
      sid: 'VPCNetworkInterface',
      effect: iam.Effect.ALLOW,
      actions: [
        'ec2:CreateNetworkInterface',
        'ec2:DescribeNetworkInterfaces',
        'ec2:DeleteNetworkInterface',
        'ec2:AssignPrivateIpAddresses',
        'ec2:UnassignPrivateIpAddresses',
      ],
      resources: ['*'],
    }),
  );

  // ---- Attach managed policy for Lambda VPC execution ----
  role.addManagedPolicy(
    iam.ManagedPolicy.fromAwsManagedPolicyName(
      'service-role/AWSLambdaVPCAccessExecutionRole',
    ),
  );

  return role;
}
