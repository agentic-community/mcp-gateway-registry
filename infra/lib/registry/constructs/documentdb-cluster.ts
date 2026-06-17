/**
 * DocumentDbCluster - L3 construct that creates a DocumentDB (instance-based) cluster
 * with KMS encryption, Secrets Manager credentials, and SSM parameter exports.
 *
 * Translated from: terraform/aws-ecs/documentdb.tf
 *
 * Security groups are created with egress-only. Ingress rules from ECS service
 * SGs and rotation Lambda SGs are added by RegistryServiceStack and RegistryOpsStack.
 */

import * as cdk from 'aws-cdk-lib';
import * as docdb from 'aws-cdk-lib/aws-docdb';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';
import { RegistryConfig } from '../registry-config';
import { putSecureSsmParam } from './_lib';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface DocumentDbClusterProps {
  readonly vpc: ec2.IVpc;
  readonly privateSubnets: ec2.ISubnet[];
  readonly config: RegistryConfig;
}

// ---------------------------------------------------------------------------
// Construct
// ---------------------------------------------------------------------------

export class DocumentDbCluster extends Construct {
  /** The CfnDBCluster resource */
  public readonly cluster: docdb.CfnDBCluster;

  /** Security group for DocumentDB (egress-only; ingress added by other stacks) */
  public readonly sg: ec2.SecurityGroup;

  /** KMS key used for DocumentDB and secrets encryption */
  public readonly kmsKey: kms.Key;

  /** ARN of the Secrets Manager secret holding DocumentDB credentials */
  public readonly secretArn: string;

  /** ARN of the DocumentDB cluster */
  public readonly clusterArn: string;

  /** Primary read-write endpoint of the DocumentDB cluster */
  public readonly endpoint: string;

  /** Reader endpoint of the DocumentDB cluster */
  public readonly readerEndpoint: string;

  constructor(scope: Construct, id: string, props: DocumentDbClusterProps) {
    super(scope, id);

    const { vpc, privateSubnets, config } = props;
    const region = cdk.Stack.of(this).region;
    const accountId = cdk.Stack.of(this).account;

    // ------------------------------------------------------------------
    // KMS key for DocumentDB encryption
    // ------------------------------------------------------------------

    this.kmsKey = new kms.Key(this, 'Key', {
      description: 'KMS key for DocumentDB Cluster and secrets encryption',
      enableKeyRotation: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      pendingWindow: cdk.Duration.days(7),
    });

    // Apply the same key policy from documentdb.tf
    this.kmsKey.addToResourcePolicy(
      new iam.PolicyStatement({
        sid: 'AllowEcsTaskExecDecrypt',
        effect: iam.Effect.ALLOW,
        principals: [new iam.AnyPrincipal()],
        actions: ['kms:Decrypt', 'kms:DescribeKey'],
        resources: ['*'],
        conditions: {
          StringEquals: {
            'aws:PrincipalAccount': accountId,
          },
          StringLike: {
            'aws:PrincipalArn': [
              `arn:aws:iam::${accountId}:role/*task-exec*`,
              `arn:aws:iam::${accountId}:role/mcp-gateway-v2-*`,
            ],
          },
        },
      }),
    );

    this.kmsKey.addToResourcePolicy(
      new iam.PolicyStatement({
        sid: 'AllowCloudWatchLogs',
        effect: iam.Effect.ALLOW,
        principals: [new iam.ServicePrincipal(`logs.${region}.amazonaws.com`)],
        actions: [
          'kms:Encrypt',
          'kms:Decrypt',
          'kms:ReEncrypt*',
          'kms:GenerateDataKey*',
          'kms:CreateGrant',
          'kms:DescribeKey',
        ],
        resources: ['*'],
        conditions: {
          ArnLike: {
            'kms:EncryptionContext:aws:logs:arn': `arn:aws:logs:${region}:${accountId}:log-group:*`,
          },
        },
      }),
    );

    new kms.Alias(this, 'KeyAlias', {
      aliasName: `alias/${config.name}-documentdb`,
      targetKey: this.kmsKey,
    });

    // ------------------------------------------------------------------
    // Secrets Manager secret for DocumentDB credentials
    // ------------------------------------------------------------------

    const secret = new secretsmanager.Secret(this, 'Secret', {
      secretName: `${config.name}/documentdb/credentials`,
      description: 'DocumentDB Cluster admin credentials',
      encryptionKey: this.kmsKey,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const cfnSecret = secret.node.defaultChild as secretsmanager.CfnSecret;

    // Set the secret value with username, password, and engine
    cfnSecret.addPropertyOverride('SecretString', JSON.stringify({
      username: config.documentdb.adminUsername,
      password: config.documentdb.adminPassword,
      engine: 'docdb',
    }));

    // Remove any GenerateSecretString that the L2 might add
    cfnSecret.addPropertyDeletionOverride('GenerateSecretString');

    this.secretArn = secret.secretArn;

    // ------------------------------------------------------------------
    // Security group (egress-only)
    // ------------------------------------------------------------------

    this.sg = new ec2.SecurityGroup(this, 'Sg', {
      vpc,
      securityGroupName: `${config.name}-v2-documentdb-sg`,
      description: 'Security group for DocumentDB Elastic Cluster',
      allowAllOutbound: true,
    });

    cdk.Tags.of(this.sg).add('Name', `${config.name}-v2-documentdb-sg`);
    cdk.Tags.of(this.sg).add('Component', 'documentdb');

    // ------------------------------------------------------------------
    // DB subnet group
    // ------------------------------------------------------------------

    const subnetGroup = new docdb.CfnDBSubnetGroup(this, 'SubnetGroup', {
      dbSubnetGroupDescription: `Subnet group for ${config.name} DocumentDB cluster`,
      dbSubnetGroupName: `${config.name}-registry-subnet-group`,
      subnetIds: privateSubnets.map((s) => s.subnetId),
      tags: [
        { key: 'Name', value: `${config.name}-registry-subnet-group` },
        { key: 'Component', value: 'documentdb' },
      ],
    });

    // ------------------------------------------------------------------
    // Cluster parameter group
    // ------------------------------------------------------------------

    const parameterGroup = new docdb.CfnDBClusterParameterGroup(this, 'ParamGroup', {
      family: 'docdb5.0',
      name: `${config.name}-registry-params`,
      description: 'DocumentDB cluster parameter group for MCP Gateway Registry',
      parameters: {
        tls: 'enabled',
        audit_logs: 'enabled',
        ttl_monitor: 'enabled',
      },
      tags: [
        { key: 'Name', value: `${config.name}-registry-params` },
        { key: 'Component', value: 'documentdb' },
      ],
    });

    // ------------------------------------------------------------------
    // DocumentDB cluster
    // ------------------------------------------------------------------

    this.cluster = new docdb.CfnDBCluster(this, 'Cluster', {
      dbClusterIdentifier: `${config.name}-registry`,
      engineVersion: '5.0.0',
      masterUsername: config.documentdb.adminUsername,
      masterUserPassword: config.documentdb.adminPassword,
      dbSubnetGroupName: subnetGroup.dbSubnetGroupName,
      vpcSecurityGroupIds: [this.sg.securityGroupId],
      port: 27017,
      backupRetentionPeriod: 7,
      preferredBackupWindow: '02:00-04:00',
      preferredMaintenanceWindow: 'sun:04:00-sun:05:00',
      storageEncrypted: true,
      kmsKeyId: this.kmsKey.keyArn,
      dbClusterParameterGroupName: parameterGroup.name,
      deletionProtection: false,
      enableCloudwatchLogsExports: ['audit', 'profiler'],
      tags: [
        { key: 'Name', value: `${config.name}-registry-docdb` },
        { key: 'Component', value: 'documentdb' },
        { key: 'Environment', value: 'production' },
        { key: 'Service', value: 'mcp-gateway-registry' },
      ],
    });

    this.cluster.addDependency(subnetGroup);
    this.cluster.addDependency(parameterGroup);

    // ------------------------------------------------------------------
    // Primary instance
    // ------------------------------------------------------------------

    const primaryInstance = new docdb.CfnDBInstance(this, 'PrimaryInstance', {
      dbClusterIdentifier: this.cluster.ref,
      dbInstanceIdentifier: `${config.name}-registry-primary`,
      dbInstanceClass: config.documentdb.instanceClass,
      autoMinorVersionUpgrade: true,
      preferredMaintenanceWindow: 'sun:04:00-sun:05:00',
      tags: [
        { key: 'Name', value: `${config.name}-registry-primary` },
        { key: 'Component', value: 'documentdb' },
        { key: 'Role', value: 'primary' },
      ],
    });

    primaryInstance.addDependency(this.cluster);

    // ------------------------------------------------------------------
    // Resolve endpoint attributes from the CfnDBCluster
    // ------------------------------------------------------------------

    this.endpoint = this.cluster.attrEndpoint;
    this.readerEndpoint = this.cluster.attrReadEndpoint;
    this.clusterArn = cdk.Fn.sub(
      'arn:aws:rds:${AWS::Region}:${AWS::AccountId}:cluster:${ClusterId}',
      { ClusterId: this.cluster.ref },
    );

    // ------------------------------------------------------------------
    // SSM parameters
    // ------------------------------------------------------------------

    new ssm.StringParameter(this, 'EndpointParam', {
      parameterName: `/${config.name}/documentdb/endpoint`,
      description: 'DocumentDB Cluster endpoint',
      stringValue: this.endpoint,
    });

    new ssm.StringParameter(this, 'ReaderEndpointParam', {
      parameterName: `/${config.name}/documentdb/reader_endpoint`,
      description: 'DocumentDB Cluster reader endpoint',
      stringValue: this.readerEndpoint,
    });

    const connectionString = cdk.Fn.join('', [
      'mongodb://',
      config.documentdb.adminUsername,
      ':',
      config.documentdb.adminPassword,
      '@',
      this.endpoint,
      ':27017/?authMechanism=SCRAM-SHA-1&authSource=admin&tls=true',
      '&tlsCAFile=global-bundle.pem&replicaSet=rs0',
      '&readPreference=secondaryPreferred&retryWrites=false',
    ]);
    putSecureSsmParam(
      this, 'ConnectionStringParam',
      `/${config.name}/documentdb/connection_string`,
      connectionString,
      this.kmsKey,
    );
  }
}
