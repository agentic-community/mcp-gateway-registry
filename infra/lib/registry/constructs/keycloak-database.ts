/**
 * KeycloakDatabase - L3 construct for Aurora MySQL Serverless v2 + RDS Proxy.
 *
 * Translated from: terraform/aws-ecs/keycloak-database.tf
 */

import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as rds from 'aws-cdk-lib/aws-rds';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';
import { RegistryConfig } from '../registry-config';
import { putSecureSsmParam } from './_lib';

export interface KeycloakDatabaseProps {
  readonly vpc: ec2.IVpc;
  readonly privateSubnets: ec2.ISubnet[];
  readonly config: RegistryConfig;
}

export class KeycloakDatabase extends Construct {
  public readonly cluster: rds.CfnDBCluster;
  public readonly proxy: rds.CfnDBProxy;
  public readonly secret: secretsmanager.Secret;
  public readonly sg: ec2.SecurityGroup;
  public readonly kmsKey: kms.Key;

  constructor(scope: Construct, id: string, props: KeycloakDatabaseProps) {
    super(scope, id);

    const { vpc, privateSubnets, config } = props;
    const stack = cdk.Stack.of(this);
    const region = stack.region;
    const accountId = stack.account;

    // KMS key for RDS + secrets encryption
    this.kmsKey = new kms.Key(this, 'KmsKey', {
      description: 'KMS key for RDS and secrets encryption',
      enableKeyRotation: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      pendingWindow: cdk.Duration.days(7),
    });

    this.kmsKey.addToResourcePolicy(new iam.PolicyStatement({
      sid: 'AllowEcsTaskExecDecrypt',
      effect: iam.Effect.ALLOW,
      principals: [new iam.AnyPrincipal()],
      actions: ['kms:Decrypt', 'kms:DescribeKey'],
      resources: ['*'],
      conditions: {
        StringEquals: { 'aws:PrincipalAccount': accountId },
        StringLike: { 'aws:PrincipalArn': `arn:aws:iam::${accountId}:role/*task-exec*` },
      },
    }));

    this.kmsKey.addToResourcePolicy(new iam.PolicyStatement({
      sid: 'AllowRdsService',
      effect: iam.Effect.ALLOW,
      principals: [new iam.ServicePrincipal('rds.amazonaws.com')],
      actions: ['kms:Decrypt', 'kms:DescribeKey', 'kms:CreateGrant'],
      resources: ['*'],
      conditions: { StringEquals: { 'kms:ViaService': `rds.${region}.amazonaws.com` } },
    }));

    this.kmsKey.addToResourcePolicy(new iam.PolicyStatement({
      sid: 'AllowCloudWatchLogs',
      effect: iam.Effect.ALLOW,
      principals: [new iam.ServicePrincipal(`logs.${region}.amazonaws.com`)],
      actions: ['kms:Encrypt', 'kms:Decrypt', 'kms:ReEncrypt*', 'kms:GenerateDataKey*', 'kms:CreateGrant', 'kms:DescribeKey'],
      resources: ['*'],
      conditions: {
        ArnLike: { 'kms:EncryptionContext:aws:logs:arn': `arn:aws:logs:${region}:${accountId}:log-group:*` },
      },
    }));

    new kms.Alias(this, 'KmsAlias', { aliasName: 'alias/keycloak-rds', targetKey: this.kmsKey });

    // Secret with static username/password (no rotation generation)
    this.secret = new secretsmanager.Secret(this, 'Secret', {
      secretName: 'keycloak/database',
      description: 'Keycloak database credentials',
      encryptionKey: this.kmsKey,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    const cfnSecret = this.secret.node.defaultChild as secretsmanager.CfnSecret;
    cfnSecret.addPropertyOverride('SecretString', JSON.stringify({
      username: config.keycloak.databaseUsername,
      password: config.keycloak.databasePassword,
    }));
    cfnSecret.addPropertyDeletionOverride('GenerateSecretString');

    // SG (egress-only; ingress added by callers)
    this.sg = new ec2.SecurityGroup(this, 'Sg', {
      vpc,
      securityGroupName: 'keycloak-db',
      description: 'Security group for Keycloak database',
      allowAllOutbound: true,
    });
    cdk.Tags.of(this.sg).add('Name', 'keycloak-db');

    // Subnet + parameter group + cluster + instance
    const subnetGroup = new rds.CfnDBSubnetGroup(this, 'SubnetGroup', {
      dbSubnetGroupDescription: 'Subnet group for Keycloak Aurora MySQL cluster',
      dbSubnetGroupName: 'keycloak-subnet-group',
      subnetIds: privateSubnets.map((s) => s.subnetId),
      tags: [{ key: 'Name', value: 'keycloak-subnet-group' }],
    });

    const paramGroup = new rds.CfnDBClusterParameterGroup(this, 'ParamGroup', {
      family: 'aurora-mysql8.0',
      description: 'Keycloak Aurora MySQL parameter group',
      dbClusterParameterGroupName: 'keycloak-params',
      parameters: { character_set_server: 'utf8mb4', collation_server: 'utf8mb4_unicode_ci' },
    });

    this.cluster = new rds.CfnDBCluster(this, 'Cluster', {
      dbClusterIdentifier: 'keycloak',
      engine: 'aurora-mysql',
      engineVersion: '8.0.mysql_aurora.3.10.3',
      databaseName: 'keycloak',
      masterUsername: config.keycloak.databaseUsername,
      masterUserPassword: config.keycloak.databasePassword,
      dbSubnetGroupName: subnetGroup.dbSubnetGroupName,
      dbClusterParameterGroupName: paramGroup.dbClusterParameterGroupName,
      vpcSecurityGroupIds: [this.sg.securityGroupId],
      backupRetentionPeriod: 7,
      preferredBackupWindow: '02:00-04:00',
      preferredMaintenanceWindow: 'sun:04:00-sun:05:00',
      copyTagsToSnapshot: true,
      storageEncrypted: true,
      kmsKeyId: this.kmsKey.keyArn,
      deletionProtection: false,
      serverlessV2ScalingConfiguration: {
        maxCapacity: config.keycloak.databaseMaxAcu,
        minCapacity: config.keycloak.databaseMinAcu,
      },
    });
    this.cluster.addDependency(subnetGroup);
    this.cluster.addDependency(paramGroup);

    const instance = new rds.CfnDBInstance(this, 'Instance', {
      dbClusterIdentifier: this.cluster.ref,
      dbInstanceClass: 'db.serverless',
      engine: 'aurora-mysql',
      engineVersion: '8.0.mysql_aurora.3.10.3',
      autoMinorVersionUpgrade: true,
    });
    instance.addDependency(this.cluster);

    // RDS Proxy
    const proxyRole = new iam.Role(this, 'ProxyRole', {
      roleName: `keycloak-rds-proxy-role-${config.awsRegion}`,
      assumedBy: new iam.ServicePrincipal('rds.amazonaws.com'),
    });
    proxyRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['secretsmanager:GetSecretValue'],
      resources: [this.secret.secretArn],
    }));
    proxyRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['kms:Decrypt'],
      resources: [this.kmsKey.keyArn],
    }));

    this.sg.addIngressRule(this.sg, ec2.Port.tcp(3306), 'RDS Proxy self-referencing rule for Aurora MySQL');

    this.proxy = new rds.CfnDBProxy(this, 'Proxy', {
      dbProxyName: 'keycloak-proxy',
      engineFamily: 'MYSQL',
      auth: [{
        authScheme: 'SECRETS',
        secretArn: this.secret.secretArn,
        clientPasswordAuthType: 'MYSQL_CACHING_SHA2_PASSWORD',
        iamAuth: 'DISABLED',
      }],
      roleArn: proxyRole.roleArn,
      vpcSubnetIds: privateSubnets.map((s) => s.subnetId),
      vpcSecurityGroupIds: [this.sg.securityGroupId],
      requireTls: false,
    });
    this.proxy.addDependency(instance);

    const proxyTargetGroup = new rds.CfnDBProxyTargetGroup(this, 'ProxyTargetGroup', {
      dbProxyName: this.proxy.dbProxyName!,
      targetGroupName: 'default',
      dbClusterIdentifiers: [this.cluster.ref],
    });
    proxyTargetGroup.addDependency(this.proxy);
    proxyTargetGroup.addDependency(instance);

    // SSM SecureString for JDBC URL (via Proxy so creds rotate without app restart)
    const dbUrl = cdk.Fn.join('', [
      'jdbc:mysql://', this.proxy.attrEndpoint,
      ':3306/keycloak?allowPublicKeyRetrieval=true&useSSL=false',
    ]);
    const ssmCr = putSecureSsmParam(this, 'DbUrlParam', '/keycloak/database/url', dbUrl, this.kmsKey);
    ssmCr.node.addDependency(instance);
  }
}
