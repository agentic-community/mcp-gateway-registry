/**
 * RegistryBuildStack - ECR repositories and CodeBuild pipeline.
 *
 * Instantiates the CodeBuildPipeline L3 construct which creates
 * ECR repos for all service images, an S3 artifacts bucket, and
 * a CodeBuild project for building container images.
 *
 * This stack has no dependencies on other registry stacks.
 */

import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { RegistryConfig } from './registry-config';
import { CodeBuildPipeline } from './constructs/codebuild-pipeline';

// ---------------------------------------------------------------------------
// Stack props
// ---------------------------------------------------------------------------

export interface RegistryBuildStackProps extends cdk.StackProps {
  readonly config: RegistryConfig;
}

// ---------------------------------------------------------------------------
// Stack
// ---------------------------------------------------------------------------

export class RegistryBuildStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: RegistryBuildStackProps) {
    super(scope, id, props);

    const { config } = props;

    // ------------------------------------------------------------------
    // CodeBuild pipeline (no-op when config.createCodebuild is false)
    // ------------------------------------------------------------------

    new CodeBuildPipeline(this, 'CodeBuild', {
      config,
    });

    // ------------------------------------------------------------------
    // Common tags
    // ------------------------------------------------------------------

    cdk.Tags.of(this).add('Project', 'mcp-gateway-registry');
    cdk.Tags.of(this).add('Component', 'build');
    cdk.Tags.of(this).add('Environment', 'production');
    cdk.Tags.of(this).add('ManagedBy', 'cdk');
  }
}
