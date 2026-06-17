/**
 * RegistryAuthStack - Keycloak authentication infrastructure.
 *
 * Instantiates the KeycloakService L3 construct which creates:
 *   - ECR repository with lifecycle policies
 *   - ECS Cluster (Fargate + Fargate Spot)
 *   - ECS Task Definition, Service, and Auto Scaling
 *   - Application Load Balancer with HTTP/HTTPS listeners
 *   - Security groups for ALB, ECS, and database connectivity
 *   - Route53 DNS record and ACM certificate (when enabled)
 *   - SSM parameters for Keycloak admin credentials
 *
 * This stack depends on both RegistryNetworkStack and RegistryDataStack.
 */

import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as kms from 'aws-cdk-lib/aws-kms';
import { Construct } from 'constructs';
import { RegistryConfig } from './registry-config';
import { KeycloakService } from './constructs/keycloak-service';

// ---------------------------------------------------------------------------
// Stack props
// ---------------------------------------------------------------------------

export interface RegistryAuthStackProps extends cdk.StackProps {
  readonly config: RegistryConfig;
  /** VPC from the network stack */
  readonly vpc: ec2.IVpc;
  /** Private subnets from the network stack */
  readonly privateSubnets: ec2.ISubnet[];
  /** Public subnets from the network stack */
  readonly publicSubnets: ec2.ISubnet[];
  /** Keycloak database security group from the data stack */
  readonly keycloakDbSg: ec2.ISecurityGroup;
  /** KMS key used for RDS / SSM encryption from the data stack */
  readonly rdsKmsKey: kms.IKey;
  /** ARN of the Secrets Manager secret containing Keycloak DB credentials */
  readonly keycloakDbSecretArn: string;
}

// ---------------------------------------------------------------------------
// Stack
// ---------------------------------------------------------------------------

export class RegistryAuthStack extends cdk.Stack {
  /** Full HTTPS URL for Keycloak (e.g. https://kc.us-west-2.mycorp.click) */
  public readonly keycloakUrl: string;

  /** ALB DNS name for Keycloak */
  public readonly keycloakAlbDns: string;

  /** ARN of the Keycloak ALB */
  public readonly keycloakAlbArn: string;

  /** Security group attached to the Keycloak ALB */
  public readonly keycloakAlbSg: ec2.SecurityGroup;

  /** Security group attached to Keycloak ECS tasks */
  public readonly keycloakEcsSg: ec2.SecurityGroup;

  /** ECS cluster running Keycloak */
  public readonly keycloakEcsCluster: ecs.Cluster;

  /** ECR repository for Keycloak container images */
  public readonly keycloakEcrRepo: ecr.Repository;

  /** Resolved Keycloak domain name */
  public readonly keycloakDomain: string;

  constructor(scope: Construct, id: string, props: RegistryAuthStackProps) {
    super(scope, id, props);

    // ------------------------------------------------------------------
    // Keycloak Service construct
    // ------------------------------------------------------------------

    const keycloak = new KeycloakService(this, 'KeycloakService', {
      config: props.config,
      vpc: props.vpc,
      privateSubnets: props.privateSubnets,
      publicSubnets: props.publicSubnets,
      keycloakDbSg: props.keycloakDbSg,
      rdsKmsKey: props.rdsKmsKey,
      keycloakDbSecretArn: props.keycloakDbSecretArn,
    });

    // ------------------------------------------------------------------
    // Expose cross-stack outputs
    // ------------------------------------------------------------------

    this.keycloakUrl = keycloak.keycloakUrl;
    this.keycloakAlbDns = keycloak.alb.loadBalancerDnsName;
    this.keycloakAlbArn = keycloak.alb.loadBalancerArn;
    this.keycloakAlbSg = keycloak.albSg;
    this.keycloakEcsSg = keycloak.ecsSg;
    this.keycloakEcsCluster = keycloak.ecsCluster;
    this.keycloakEcrRepo = keycloak.ecrRepo;
    this.keycloakDomain = keycloak.keycloakDomain;

    // ------------------------------------------------------------------
    // Common tags
    // ------------------------------------------------------------------

    cdk.Tags.of(this).add('Project', 'mcp-gateway-registry');
    cdk.Tags.of(this).add('Component', 'auth');
    cdk.Tags.of(this).add('Environment', 'production');
    cdk.Tags.of(this).add('ManagedBy', 'cdk');

    // ------------------------------------------------------------------
    // Stack Outputs
    // ------------------------------------------------------------------

    new cdk.CfnOutput(this, 'KeycloakUrl', {
      value: this.keycloakUrl,
      description: 'Keycloak URL',
    });

    new cdk.CfnOutput(this, 'KeycloakAlbDnsName', {
      value: this.keycloakAlbDns,
      description: 'Keycloak ALB DNS name',
    });

    new cdk.CfnOutput(this, 'KeycloakAdminConsole', {
      value: `${this.keycloakUrl}/admin`,
      description: 'Keycloak admin console URL',
    });
  }
}
