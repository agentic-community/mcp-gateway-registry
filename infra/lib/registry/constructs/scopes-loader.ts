/**
 * ScopesLoader - upserts UI-scope group docs into DocumentDB and copies
 * scopes.yml to EFS on every deploy. Bridges a Terraform-parity gap.
 */

import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as efs from 'aws-cdk-lib/aws-efs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as cr from 'aws-cdk-lib/custom-resources';
import { ILocalBundling } from 'aws-cdk-lib/core';
import * as child_process from 'child_process';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { Construct } from 'constructs';

const CA_BUNDLE_URL = 'https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem';

// Cache the Amazon RDS/DocumentDB CA bundle once per synth run, outside the
// repo (so it isn't committed). Re-downloaded after `cdk.out` is wiped.
function ensureCaBundle(): string {
  const cached = path.join(__dirname, '..', '..', '..', 'cdk.out', '.global-bundle.pem');
  fs.mkdirSync(path.dirname(cached), { recursive: true });
  if (!fs.existsSync(cached) || fs.statSync(cached).size === 0) {
    child_process.execSync(`curl -sfo "${cached}" "${CA_BUNDLE_URL}"`, { stdio: 'inherit' });
  }
  return cached;
}

// ponytail: local-only Lambda bundling (no Docker fallback). Requires pip3 on
// the synth host. Drop and use Docker if a CI runner without pip3 ever shows up.
function buildLocalBundling(scopesYmlPath: string, caBundlePath: string): ILocalBundling {
  return {
    tryBundle(outputDir: string, _opts): boolean {
      try {
        child_process.execSync('pip3 --version', { stdio: 'ignore' });
      } catch {
        return false;
      }
      const inputDir = path.join(__dirname, '..', '..', '..', 'lambda', 'scopes-loader');
      child_process.execSync(
        `pip3 install -r ${path.join(inputDir, 'requirements.txt')} -t ${outputDir} --quiet --no-cache-dir`,
        { stdio: 'inherit' },
      );
      fs.copyFileSync(path.join(inputDir, 'index.py'), path.join(outputDir, 'index.py'));
      fs.copyFileSync(caBundlePath, path.join(outputDir, 'global-bundle.pem'));
      fs.copyFileSync(scopesYmlPath, path.join(outputDir, 'scopes.yml'));
      return true;
    },
  };
}

export interface ScopesLoaderProps {
  readonly vpc: ec2.IVpc;
  readonly privateSubnets: ec2.ISubnet[];
  /** SG that already has ingress rights into DocumentDB */
  readonly ingressSg: ec2.ISecurityGroup;
  readonly documentDbHost: string;
  readonly documentDbPort: number;
  readonly documentDbDatabase: string;
  readonly documentDbNamespace: string;
  readonly documentDbSecretArn: string;
  readonly documentDbSecretKmsKeyArn: string;
  /** Absolute path to scopes.yml */
  readonly scopesYmlPath: string;
  /** EFS access point for the auth-config volume (the Lambda copies scopes.yml here) */
  readonly authConfigAccessPoint: efs.IAccessPoint;
  readonly namePrefix: string;
}

export class ScopesLoader extends Construct {
  constructor(scope: Construct, id: string, props: ScopesLoaderProps) {
    super(scope, id);

    const scopesYml = fs.readFileSync(props.scopesYmlPath, 'utf8');
    const caBundle = ensureCaBundle();

    // Stage Lambda source + scopes.yml + CA bundle into one dir so
    // Code.fromAsset picks up the YAML alongside index.py (4KB env-var
    // limit forbids inlining the YAML at runtime).
    const lambdaSrc = path.join(__dirname, '..', '..', '..', 'lambda', 'scopes-loader');
    const stagingDir = path.join(__dirname, '..', '..', '..', 'cdk.out', '.scopes-loader');
    fs.rmSync(stagingDir, { recursive: true, force: true });
    fs.cpSync(lambdaSrc, stagingDir, { recursive: true });
    fs.copyFileSync(props.scopesYmlPath, path.join(stagingDir, 'scopes.yml'));
    fs.copyFileSync(caBundle, path.join(stagingDir, 'global-bundle.pem'));

    // Lambda role: VPC + Secrets Manager read + KMS decrypt
    const role = new iam.Role(this, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaVPCAccessExecutionRole'),
      ],
    });
    role.addToPolicy(new iam.PolicyStatement({
      actions: ['secretsmanager:GetSecretValue'],
      resources: [props.documentDbSecretArn],
    }));
    role.addToPolicy(new iam.PolicyStatement({
      actions: ['kms:Decrypt'],
      resources: [props.documentDbSecretKmsKeyArn],
    }));

    const fn = new lambda.Function(this, 'Fn', {
      functionName: `${props.namePrefix}-scopes-loader`,
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'index.handler',
      logRetention: logs.RetentionDays.ONE_MONTH,
      code: lambda.Code.fromAsset(stagingDir, {
        bundling: {
          image: lambda.Runtime.PYTHON_3_11.bundlingImage,
          local: buildLocalBundling(props.scopesYmlPath, caBundle),
          user: 'root',
          command: [
            'bash', '-c',
            'mkdir -p /asset-output && pip install -r /asset-input/requirements.txt -t /asset-output --no-cache-dir && cp /asset-input/index.py /asset-input/global-bundle.pem /asset-input/scopes.yml /asset-output/',
          ],
        },
      }),
      memorySize: 256,
      timeout: cdk.Duration.minutes(2),
      role,
      vpc: props.vpc,
      vpcSubnets: { subnets: props.privateSubnets },
      securityGroups: [props.ingressSg],
      filesystem: lambda.FileSystem.fromEfsAccessPoint(props.authConfigAccessPoint, '/mnt/auth_config'),
      environment: {
        DOCUMENTDB_HOST: props.documentDbHost,
        DOCUMENTDB_PORT: String(props.documentDbPort),
        DOCUMENTDB_DATABASE: props.documentDbDatabase,
        DOCUMENTDB_NAMESPACE: props.documentDbNamespace,
        DOCUMENTDB_SECRET_ARN: props.documentDbSecretArn,
      },
    });

    // Re-invoke on every YAML change — physicalResourceId derived from a
    // synth-time hash of the file content (not a CFN token).
    const scopesHash = crypto.createHash('sha256').update(scopesYml).digest('hex').slice(0, 16);
    new cr.AwsCustomResource(this, 'Trigger', {
      onUpdate: {
        service: 'Lambda',
        action: 'invoke',
        parameters: {
          FunctionName: fn.functionName,
          Payload: JSON.stringify({ hash: scopesHash }),
        },
        physicalResourceId: cr.PhysicalResourceId.of(`scopes-loader-${scopesHash}`),
      },
      policy: cr.AwsCustomResourcePolicy.fromStatements([
        new iam.PolicyStatement({
          actions: ['lambda:InvokeFunction'],
          resources: [fn.functionArn],
        }),
      ]),
    });
  }
}
