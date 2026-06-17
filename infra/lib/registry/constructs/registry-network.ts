/**
 * RegistryNetwork - L3 construct for VPC, subnets, NAT, and VPC endpoints.
 *
 * Translated from: terraform/aws-ecs/vpc.tf
 */

import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as cr from 'aws-cdk-lib/custom-resources';
import { Construct } from 'constructs';
import { RegistryConfig } from '../registry-config';

export interface RegistryNetworkProps {
  readonly config: RegistryConfig;
}

export class RegistryNetwork extends Construct {
  public readonly vpc: ec2.Vpc;
  public readonly privateSubnets: ec2.ISubnet[];
  public readonly publicSubnets: ec2.ISubnet[];
  public readonly vpcEndpointsSg: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: RegistryNetworkProps) {
    super(scope, id);
    const { config } = props;

    // Service-linked roles (idempotent — InvalidInput on "already exists").
    for (const service of ['ecs.amazonaws.com', 'elasticloadbalancing.amazonaws.com']) {
      const short = service.split('.')[0];
      new cr.AwsCustomResource(this, `${short}Slr`, {
        onCreate: {
          service: 'IAM',
          action: 'createServiceLinkedRole',
          parameters: { AWSServiceName: service },
          physicalResourceId: cr.PhysicalResourceId.of(`slr-${short}`),
          ignoreErrorCodesMatching: 'InvalidInput',
        },
        policy: cr.AwsCustomResourcePolicy.fromStatements([
          new iam.PolicyStatement({
            actions: ['iam:CreateServiceLinkedRole'],
            resources: [`arn:aws:iam::*:role/aws-service-role/${service}/*`],
          }),
        ]),
      });
    }

    // VPC: /20 private + /24 public per AZ, NAT-per-AZ
    this.vpc = new ec2.Vpc(this, 'Vpc', {
      vpcName: `${config.name}-vpc`,
      ipAddresses: ec2.IpAddresses.cidr(config.vpcCidr),
      maxAzs: 3,
      natGateways: 3,
      enableDnsHostnames: true,
      enableDnsSupport: true,
      subnetConfiguration: [
        { cidrMask: 20, name: 'private', subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
        { cidrMask: 24, name: 'public', subnetType: ec2.SubnetType.PUBLIC },
      ],
    });

    this.privateSubnets = this.vpc.selectSubnets({
      subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
    }).subnets;
    this.publicSubnets = this.vpc.selectSubnets({ subnetType: ec2.SubnetType.PUBLIC }).subnets;

    for (const subnet of this.privateSubnets) cdk.Tags.of(subnet).add('subnet-type', 'private');
    for (const subnet of this.publicSubnets) cdk.Tags.of(subnet).add('subnet-type', 'public');

    // VPC endpoints SG
    this.vpcEndpointsSg = new ec2.SecurityGroup(this, 'VpcEndpointsSg', {
      vpc: this.vpc,
      securityGroupName: `${config.name}-vpc-endpoints`,
      description: 'Security group for VPC interface endpoints allowing HTTPS from within VPC',
      allowAllOutbound: true,
    });

    this.vpcEndpointsSg.addIngressRule(
      ec2.Peer.ipv4(config.vpcCidr),
      ec2.Port.tcp(443),
      'Allow HTTPS from VPC CIDR for AWS service endpoints',
    );

    this.vpc.addInterfaceEndpoint('StsEndpoint', {
      service: ec2.InterfaceVpcEndpointAwsService.STS,
      subnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [this.vpcEndpointsSg],
      privateDnsEnabled: true,
    });

    this.vpc.addGatewayEndpoint('S3Endpoint', {
      service: ec2.GatewayVpcEndpointAwsService.S3,
      subnets: [{ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }],
    });
  }
}
