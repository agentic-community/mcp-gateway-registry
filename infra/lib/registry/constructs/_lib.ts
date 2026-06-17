/**
 * Shared helpers for registry L3 constructs.
 */

import * as iam from 'aws-cdk-lib/aws-iam';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as cdk from 'aws-cdk-lib';
import * as cr from 'aws-cdk-lib/custom-resources';
import { Construct } from 'constructs';

/**
 * Put an SSM SecureString parameter via AwsCustomResource.
 * CFN does not support SecureString natively, hence this dance.
 */
export function putSecureSsmParam(
  scope: Construct,
  id: string,
  name: string,
  value: string,
  kmsKey: kms.IKey,
): cr.AwsCustomResource {
  const params = {
    Name: name,
    Type: 'SecureString',
    KeyId: kmsKey.keyId,
    Value: value,
    Overwrite: true,
  };
  return new cr.AwsCustomResource(scope, id, {
    onCreate: { service: 'SSM', action: 'putParameter', parameters: params, physicalResourceId: cr.PhysicalResourceId.of(name) },
    onUpdate: { service: 'SSM', action: 'putParameter', parameters: params, physicalResourceId: cr.PhysicalResourceId.of(name) },
    onDelete: { service: 'SSM', action: 'deleteParameter', parameters: { Name: name } },
    policy: cr.AwsCustomResourcePolicy.fromStatements([
      new iam.PolicyStatement({
        actions: ['ssm:PutParameter', 'ssm:DeleteParameter', 'ssm:AddTagsToResource'],
        resources: [cdk.Stack.of(scope).formatArn({
          service: 'ssm',
          resource: 'parameter',
          resourceName: name.replace(/^\//, ''),
        })],
      }),
      new iam.PolicyStatement({
        actions: ['kms:Encrypt', 'kms:GenerateDataKey'],
        resources: [kmsKey.keyArn],
      }),
    ]),
  });
}
