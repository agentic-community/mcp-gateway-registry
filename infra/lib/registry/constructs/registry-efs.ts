/**
 * RegistryEfs - L3 construct for the EFS file system + access points.
 *
 * Translated from: terraform/aws-ecs/modules/mcp-gateway/storage.tf
 */

import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as efs from 'aws-cdk-lib/aws-efs';
import { Construct } from 'constructs';
import { RegistryConfig } from '../registry-config';

const ACCESS_POINTS = [
  { key: 'servers', path: '/servers' },
  { key: 'models', path: '/models' },
  { key: 'logs', path: '/logs' },
  { key: 'agents', path: '/agents' },
  { key: 'authConfig', path: '/auth_config' },
  { key: 'mcpgwData', path: '/mcpgw_data' },
] as const;

export interface RegistryEfsProps {
  readonly config: RegistryConfig;
  readonly vpc: ec2.IVpc;
  readonly privateSubnets: ec2.ISubnet[];
}

export class RegistryEfs extends Construct {
  public readonly fileSystem: efs.FileSystem;
  public readonly accessPoints: Record<string, efs.AccessPoint>;

  constructor(scope: Construct, id: string, props: RegistryEfsProps) {
    super(scope, id);
    const { config, vpc, privateSubnets } = props;
    const { name: namePrefix } = config;

    this.fileSystem = new efs.FileSystem(this, 'Fs', {
      fileSystemName: `${namePrefix}-efs`,
      vpc,
      vpcSubnets: { subnets: privateSubnets },
      encrypted: true,
      performanceMode: efs.PerformanceMode.GENERAL_PURPOSE,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Allow ECS tasks (any IP in the VPC) to mount via NFS.
    this.fileSystem.connections.allowFrom(
      ec2.Peer.ipv4(config.vpcCidr),
      ec2.Port.tcp(2049),
      'NFS from VPC CIDR (ECS tasks)',
    );

    this.accessPoints = {};
    for (const ap of ACCESS_POINTS) {
      this.accessPoints[ap.key] = new efs.AccessPoint(this, `Ap${_capitalize(ap.key)}`, {
        fileSystem: this.fileSystem,
        path: ap.path,
        posixUser: { uid: '1000', gid: '1000' },
        createAcl: { ownerUid: '1000', ownerGid: '1000', permissions: '755' },
      });
      cdk.Tags.of(this.accessPoints[ap.key]).add('Name', `${namePrefix} ${ap.key}`);
    }
  }
}

function _capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
