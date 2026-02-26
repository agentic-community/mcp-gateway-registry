#
# CodeBuild Project for Building Container Images
# Set create_codebuild = true in terraform.tfvars to enable
#
# This creates:
# - ECR repositories for all service images
# - S3 bucket for buildspec storage
# - CodeBuild project that builds all containers in parallel
# - IAM role with ECR push and CloudWatch Logs permissions
#

variable "create_codebuild" {
  description = "Whether to create CodeBuild resources (ECR repos, build project) for building container images"
  type        = bool
  default     = false
}

# =============================================================================
# ECR REPOSITORIES
# =============================================================================

locals {
  # All service images that CodeBuild will build and push.
  # Keycloak is excluded — it has its own resource in keycloak-ecr.tf.
  ecr_repositories = toset([
    "mcp-gateway-registry",
    "mcp-gateway-auth-server",
    "mcp-gateway-currenttime",
    "mcp-gateway-mcpgw",
    "mcp-gateway-realserverfaketools",
    "mcp-gateway-flight-booking-agent",
    "mcp-gateway-travel-assistant-agent",
    "mcp-gateway-scopes-init",
    "mcp-gateway-metrics-service",
    "mcp-gateway-grafana",
  ])
}

resource "aws_ecr_repository" "services" {
  for_each = var.create_codebuild ? local.ecr_repositories : toset([])

  name                 = each.key
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = merge(
    local.common_tags,
    {
      Name = each.key
    }
  )
}

resource "aws_ecr_lifecycle_policy" "services" {
  for_each   = aws_ecr_repository.services
  repository = each.value.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 10
        description  = "Keep last 10 tagged images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["sha-"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 20
        description  = "Expire untagged images older than 7 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# =============================================================================
# S3 BUCKET FOR CODEBUILD ARTIFACTS
# =============================================================================

resource "aws_s3_bucket" "codebuild" {
  count  = var.create_codebuild ? 1 : 0
  bucket = "mcp-gateway-terraform-${data.aws_caller_identity.current.account_id}"

  tags = merge(
    local.common_tags,
    {
      Name = "mcp-gateway-codebuild"
    }
  )
}

resource "aws_s3_bucket_versioning" "codebuild" {
  count  = var.create_codebuild ? 1 : 0
  bucket = aws_s3_bucket.codebuild[0].id
  versioning_configuration {
    status = "Enabled"
  }
}

# =============================================================================
# BUILDSPEC (inline, uploaded to S3)
# =============================================================================

resource "aws_s3_object" "upstream_buildspec" {
  count   = var.create_codebuild ? 1 : 0
  bucket  = aws_s3_bucket.codebuild[0].id
  key     = "buildspecs/upstream-buildspec.yaml"
  content = <<-EOF
version: 0.2

env:
  variables:
    DOCKER_BUILDKIT: "1"

phases:
  pre_build:
    commands:
      - echo "=== Building MCP Gateway container images ==="
      - echo "Source version - $CODEBUILD_RESOLVED_SOURCE_VERSION"
      - export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
      - export ECR_REGISTRY="$${AWS_ACCOUNT_ID}.dkr.ecr.$${AWS_DEFAULT_REGION}.amazonaws.com"
      - export IMAGE_TAG="sha-$${CODEBUILD_RESOLVED_SOURCE_VERSION:0:7}"
      - echo "ECR Registry - $ECR_REGISTRY"
      - echo "Image tag - $IMAGE_TAG"
      - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY
      - echo "Pre-pulling base images for layer caching..."
      - docker pull public.ecr.aws/docker/library/python:3.12-slim || true
      - docker tag public.ecr.aws/docker/library/python:3.12-slim python:3.12-slim
      - docker pull quay.io/keycloak/keycloak:23.0 || true
      - docker pull grafana/grafana:12.3.1 || true
      - echo "Pulling existing images for cache..."
      - for repo in mcp-gateway-registry mcp-gateway-auth-server keycloak mcp-gateway-currenttime mcp-gateway-mcpgw mcp-gateway-realserverfaketools mcp-gateway-flight-booking-agent mcp-gateway-travel-assistant-agent mcp-gateway-scopes-init mcp-gateway-metrics-service mcp-gateway-grafana; do docker pull $ECR_REGISTRY/$repo:latest 2>/dev/null || true; done
      - echo "Setting up A2A agent dependencies..."
      - mkdir -p agents/a2a/src/flight-booking-agent/.tmp agents/a2a/src/travel-assistant-agent/.tmp
      - cp agents/a2a/pyproject.toml agents/a2a/uv.lock agents/a2a/src/flight-booking-agent/.tmp/ 2>/dev/null || true
      - cp agents/a2a/pyproject.toml agents/a2a/uv.lock agents/a2a/src/travel-assistant-agent/.tmp/ 2>/dev/null || true

  build:
    commands:
      - echo "=== Building all container images in parallel ==="
      - |
        build_and_push() {
          local name=$1
          local dockerfile=$2
          local context=$3
          echo "Starting build: $name"
          if docker build --cache-from $ECR_REGISTRY/$name:latest \
               -t $ECR_REGISTRY/$name:$IMAGE_TAG \
               --build-arg BUILD_VERSION=$IMAGE_TAG \
               -f $dockerfile $context && \
             docker tag $ECR_REGISTRY/$name:$IMAGE_TAG $ECR_REGISTRY/$name:latest && \
             docker push $ECR_REGISTRY/$name:$IMAGE_TAG && \
             docker push $ECR_REGISTRY/$name:latest; then
            echo "Completed: $name"
          else
            echo "FAILED: $name"
            return 1
          fi
        }

        # Core services
        build_and_push mcp-gateway-registry docker/Dockerfile.registry-cpu . &
        build_and_push mcp-gateway-auth-server docker/Dockerfile.auth . &
        build_and_push keycloak docker/keycloak/Dockerfile docker/keycloak &

        # MCP servers
        build_and_push mcp-gateway-currenttime docker/Dockerfile.mcp-server servers/currenttime &
        (docker build --cache-from $ECR_REGISTRY/mcp-gateway-mcpgw:latest \
          -t $ECR_REGISTRY/mcp-gateway-mcpgw:$IMAGE_TAG \
          --build-arg SERVER_DIR=servers/mcpgw --build-arg BUILD_VERSION=$IMAGE_TAG \
          -f docker/Dockerfile.mcp-server-cpu . && \
          docker tag $ECR_REGISTRY/mcp-gateway-mcpgw:$IMAGE_TAG $ECR_REGISTRY/mcp-gateway-mcpgw:latest && \
          docker push $ECR_REGISTRY/mcp-gateway-mcpgw:$IMAGE_TAG && \
          docker push $ECR_REGISTRY/mcp-gateway-mcpgw:latest && \
          echo "Completed: mcp-gateway-mcpgw" || { echo "FAILED: mcp-gateway-mcpgw"; exit 1; }) &
        build_and_push mcp-gateway-realserverfaketools docker/Dockerfile.mcp-server servers/realserverfaketools &

        # A2A agents
        build_and_push mcp-gateway-flight-booking-agent agents/a2a/src/flight-booking-agent/Dockerfile agents/a2a/src/flight-booking-agent &
        build_and_push mcp-gateway-travel-assistant-agent agents/a2a/src/travel-assistant-agent/Dockerfile agents/a2a/src/travel-assistant-agent &

        # Utilities
        build_and_push mcp-gateway-scopes-init docker/Dockerfile.scopes-init . &

        # Observability pipeline
        build_and_push mcp-gateway-metrics-service metrics-service/Dockerfile metrics-service &
        build_and_push mcp-gateway-grafana terraform/aws-ecs/grafana/Dockerfile terraform/aws-ecs/grafana &

        # Wait for all background jobs
        FAILED=0
        for job in $(jobs -p); do
          wait $job || FAILED=$((FAILED+1))
        done

        if [ $FAILED -gt 0 ]; then
          echo "$FAILED build(s) failed"
          exit 1
        fi
        echo "All builds completed successfully"

  post_build:
    commands:
      - echo "Build completed on $(date)"
      - echo "All images pushed to $ECR_REGISTRY with tags $IMAGE_TAG and latest"
EOF

  tags = local.common_tags
}

# =============================================================================
# IAM ROLE FOR CODEBUILD
# =============================================================================

resource "aws_iam_role" "codebuild" {
  count = var.create_codebuild ? 1 : 0
  name  = "mcp-gateway-tf-codebuild-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "codebuild.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "codebuild" {
  count = var.create_codebuild ? 1 : 0
  name  = "mcp-gateway-tf-codebuild-policy"
  role  = aws_iam_role.codebuild[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
        Resource = "arn:aws:ecr:${var.aws_region}:${data.aws_caller_identity.current.account_id}:repository/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion"
        ]
        Resource = "${aws_s3_bucket.codebuild[0].arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "sts:GetCallerIdentity"
        ]
        Resource = "*"
      }
    ]
  })
}

# =============================================================================
# CODEBUILD PROJECT
# =============================================================================

resource "aws_codebuild_project" "upstream" {
  count         = var.create_codebuild ? 1 : 0
  name          = "mcp-gateway-upstream-build-tf"
  description   = "Build MCP Gateway container images (all services + observability pipeline)"
  build_timeout = 60
  service_role  = aws_iam_role.codebuild[0].arn

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type                = "BUILD_GENERAL1_LARGE"
    image                       = "aws/codebuild/amazonlinux2-x86_64-standard:5.0"
    type                        = "LINUX_CONTAINER"
    privileged_mode             = true
    image_pull_credentials_type = "CODEBUILD"
  }

  source {
    type            = "GITHUB"
    location        = "https://github.com/agentic-community/mcp-gateway-registry.git"
    buildspec       = aws_s3_object.upstream_buildspec[0].content
    git_clone_depth = 1

    git_submodules_config {
      fetch_submodules = false
    }
  }

  source_version = "main"

  cache {
    type  = "LOCAL"
    modes = ["LOCAL_DOCKER_LAYER_CACHE", "LOCAL_SOURCE_CACHE"]
  }

  tags = local.common_tags
}

# =============================================================================
# OUTPUTS
# =============================================================================

output "codebuild_project_upstream" {
  description = "CodeBuild project for building from upstream"
  value       = var.create_codebuild ? aws_codebuild_project.upstream[0].name : null
}

output "codebuild_s3_bucket" {
  description = "S3 bucket for CodeBuild artifacts"
  value       = var.create_codebuild ? aws_s3_bucket.codebuild[0].id : null
}

output "ecr_repository_urls" {
  description = "ECR repository URLs for all service images (use as *_image_uri variable values)"
  value       = var.create_codebuild ? { for k, v in aws_ecr_repository.services : k => "${v.repository_url}:latest" } : {}
}
