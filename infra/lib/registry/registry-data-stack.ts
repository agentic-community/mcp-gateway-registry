/**
 * RegistryDataStack - thin wrapper around DocumentDbCluster + KeycloakDatabase L3 constructs.
 */

import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as rds from 'aws-cdk-lib/aws-rds';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';
import { RegistryConfig } from './registry-config';
import { RegistryNetworkStack } from './registry-network-stack';
import { DocumentDbCluster } from './constructs/documentdb-cluster';
import { KeycloakDatabase } from './constructs/keycloak-database';

export interface RegistryDataStackProps extends cdk.StackProps {
  readonly config: RegistryConfig;
  readonly networkStack: RegistryNetworkStack;
}

export class RegistryDataStack extends cdk.Stack {
  // DocumentDB
  public readonly documentDbCluster: import('aws-cdk-lib/aws-docdb').CfnDBCluster;
  public readonly documentDbSg: ec2.SecurityGroup;
  public readonly documentDbKmsKey: kms.Key;
  public readonly documentDbSecretArn: string;

  // Keycloak DB
  public readonly keycloakDbCluster: rds.CfnDBCluster;
  public readonly keycloakDbProxy: rds.CfnDBProxy;
  public readonly keycloakDbSecret: secretsmanager.Secret;
  public readonly keycloakDbSg: ec2.SecurityGroup;
  public readonly rdsKmsKey: kms.Key;

  constructor(scope: Construct, id: string, props: RegistryDataStackProps) {
    super(scope, id, props);

    const { config, networkStack } = props;
    const { vpc, privateSubnets } = networkStack;

    const docdb = new DocumentDbCluster(this, 'DocumentDb', { vpc, privateSubnets, config });
    this.documentDbCluster = docdb.cluster;
    this.documentDbSg = docdb.sg;
    this.documentDbKmsKey = docdb.kmsKey;
    this.documentDbSecretArn = docdb.secretArn;

    const keycloakDb = new KeycloakDatabase(this, 'KeycloakDb', { vpc, privateSubnets, config });
    this.keycloakDbCluster = keycloakDb.cluster;
    this.keycloakDbProxy = keycloakDb.proxy;
    this.keycloakDbSecret = keycloakDb.secret;
    this.keycloakDbSg = keycloakDb.sg;
    this.rdsKmsKey = keycloakDb.kmsKey;

    cdk.Tags.of(this).add('Project', 'mcp-gateway-registry');
    cdk.Tags.of(this).add('Component', 'data');
    cdk.Tags.of(this).add('Environment', 'production');
    cdk.Tags.of(this).add('ManagedBy', 'cdk');
  }
}
