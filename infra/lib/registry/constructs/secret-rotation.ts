/**
 * SecretRotation - Reusable L3 construct for a single secret rotation Lambda.
 *
 * Creates a Lambda function deployed in VPC private subnets that implements
 * the AWS Secrets Manager 4-step rotation process. Used for both DocumentDB
 * and RDS (Aurora MySQL) credential rotation.
 *
 * Translated from: terraform/aws-ecs/secret-rotation.tf (Lambda + CloudWatch sections)
 */

import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';
import { Construct } from 'constructs';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface SecretRotationProps {
  /** Descriptive name used for naming Lambda, log group, etc. */
  readonly rotationName: string;
  /** VPC in which the Lambda function runs */
  readonly vpc: ec2.IVpc;
  /** Private subnets for Lambda VPC placement */
  readonly privateSubnets: ec2.ISubnet[];
  /** Security group attached to the Lambda function */
  readonly lambdaSg: ec2.ISecurityGroup;
  /** ARN of the Secrets Manager secret to rotate */
  readonly secretArn: string;
  /** IAM role for the Lambda function */
  readonly lambdaRole: iam.IRole;
  /** Path to the Lambda source code directory (bundled via Code.fromAsset) */
  readonly lambdaCodePath: string;
  /** Additional environment variables for the Lambda function */
  readonly environmentVariables: Record<string, string>;
}

// ---------------------------------------------------------------------------
// Construct
// ---------------------------------------------------------------------------

export class SecretRotation extends Construct {
  /** The Lambda function performing secret rotation */
  public readonly lambdaFunction: lambda.Function;

  /** CloudWatch log group for the Lambda function */
  public readonly logGroup: logs.LogGroup;

  constructor(scope: Construct, id: string, props: SecretRotationProps) {
    super(scope, id);

    const {
      rotationName,
      vpc,
      privateSubnets,
      lambdaSg,
      lambdaRole,
      lambdaCodePath,
      environmentVariables,
    } = props;

    // ------------------------------------------------------------------
    // CloudWatch log group (30-day retention)
    // ------------------------------------------------------------------

    this.logGroup = new logs.LogGroup(this, 'LogGroup', {
      logGroupName: `/aws/lambda/${rotationName}`,
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // ------------------------------------------------------------------
    // Lambda function (Python 3.11, 256 MB, 300 s timeout, VPC-deployed)
    // ------------------------------------------------------------------

    this.lambdaFunction = new lambda.Function(this, 'Function', {
      functionName: rotationName,
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'index.lambda_handler',
      code: lambda.Code.fromAsset(lambdaCodePath),
      memorySize: 256,
      timeout: cdk.Duration.seconds(300),
      role: lambdaRole,
      vpc,
      vpcSubnets: { subnets: privateSubnets },
      securityGroups: [lambdaSg],
      environment: environmentVariables,
      tracing: lambda.Tracing.ACTIVE,
    });

    // Ensure the log group is created before the Lambda (CDK auto-creates
    // one otherwise, but we want explicit control over retention).
    this.lambdaFunction.node.addDependency(this.logGroup);

    // ------------------------------------------------------------------
    // Lambda permission: allow Secrets Manager to invoke
    // ------------------------------------------------------------------

    this.lambdaFunction.addPermission('AllowSecretsManager', {
      principal: new iam.ServicePrincipal('secretsmanager.amazonaws.com'),
      action: 'lambda:InvokeFunction',
    });
  }
}
