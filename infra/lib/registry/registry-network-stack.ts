/**
 * RegistryNetworkStack - thin wrapper around the RegistryNetwork L3 construct.
 */

import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Construct } from 'constructs';
import { RegistryConfig } from './registry-config';
import { RegistryNetwork } from './constructs/registry-network';

export interface RegistryNetworkStackProps extends cdk.StackProps {
  readonly config: RegistryConfig;
}

export class RegistryNetworkStack extends cdk.Stack {
  public readonly vpc: ec2.Vpc;
  public readonly privateSubnets: ec2.ISubnet[];
  public readonly publicSubnets: ec2.ISubnet[];
  public readonly vpcEndpointsSg: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: RegistryNetworkStackProps) {
    super(scope, id, props);

    const network = new RegistryNetwork(this, 'Network', { config: props.config });

    this.vpc = network.vpc;
    this.privateSubnets = network.privateSubnets;
    this.publicSubnets = network.publicSubnets;
    this.vpcEndpointsSg = network.vpcEndpointsSg;

    cdk.Tags.of(this).add('Project', 'mcp-gateway-registry');
    cdk.Tags.of(this).add('Component', 'network');
    cdk.Tags.of(this).add('Environment', 'production');
    cdk.Tags.of(this).add('ManagedBy', 'cdk');
  }
}
