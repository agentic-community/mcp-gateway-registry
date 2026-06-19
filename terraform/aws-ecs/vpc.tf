data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  azs = slice(data.aws_availability_zones.available.names, 0, 3)

  # VPC endpoint service name prefix varies by partition and endpoint type
  # Gateway endpoints (S3, DynamoDB): com.amazonaws.{region}.{service} (same in all regions)
  # Interface endpoints (STS, etc):
  #   - Standard AWS: com.amazonaws.{region}.{service}
  #   - China regions: cn.com.amazonaws.{region}.{service}
  interface_endpoint_prefix = data.aws_partition.current.partition == "aws-cn" ? "cn.com.amazonaws" : "com.amazonaws"
  gateway_endpoint_prefix   = "com.amazonaws"
}

data "aws_vpc" "existing" {
  count = var.use_existing_vpc && trimspace(var.existing_vpc_id) != "" ? 1 : 0

  id = var.existing_vpc_id
}

#checkov:skip=CKV_TF_1:Module version is pinned via version constraint
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 6.0"

  create_vpc = !var.use_existing_vpc

  name = "${var.name}-vpc"
  cidr = var.vpc_cidr

  azs             = local.azs
  private_subnets = [for k, v in local.azs : cidrsubnet(var.vpc_cidr, 4, k)]
  public_subnets  = [for k, v in local.azs : cidrsubnet(var.vpc_cidr, 8, k + 48)]

  enable_nat_gateway     = true
  single_nat_gateway     = false
  one_nat_gateway_per_az = true

  enable_dns_hostnames = true
  enable_dns_support   = true

  # VPC Flow Logs
  enable_flow_log = false

  # Tags for ECS and ALB usage
  private_subnet_tags = {
    "subnet-type" = "private"
  }

  public_subnet_tags = {
    "subnet-type" = "public"
  }
}

locals {
  selected_vpc_id                  = var.use_existing_vpc ? var.existing_vpc_id : module.vpc.vpc_id
  selected_vpc_cidr_block          = var.use_existing_vpc ? try(data.aws_vpc.existing[0].cidr_block, "") : module.vpc.vpc_cidr_block
  selected_private_subnet_ids      = var.use_existing_vpc ? var.existing_private_subnet_ids : module.vpc.private_subnets
  selected_public_subnet_ids       = var.use_existing_vpc ? var.existing_public_subnet_ids : module.vpc.public_subnets
  selected_private_route_table_ids = var.use_existing_vpc ? var.existing_private_route_table_ids : module.vpc.private_route_table_ids
  selected_nat_public_ips          = var.use_existing_vpc ? var.existing_nat_public_ips : module.vpc.nat_public_ips
}

resource "terraform_data" "existing_vpc_configuration" {
  input = var.use_existing_vpc ? var.existing_vpc_id : "managed-vpc"

  lifecycle {
    precondition {
      condition     = !var.use_existing_vpc || trimspace(var.existing_vpc_id) != ""
      error_message = "existing_vpc_id must be set when use_existing_vpc is true."
    }

    precondition {
      condition     = !var.use_existing_vpc || length(var.existing_private_subnet_ids) >= 2
      error_message = "existing_private_subnet_ids must contain at least two subnet IDs when use_existing_vpc is true."
    }

    precondition {
      condition     = !var.use_existing_vpc || length(var.existing_public_subnet_ids) >= 2
      error_message = "existing_public_subnet_ids must contain at least two subnet IDs when use_existing_vpc is true."
    }

    precondition {
      condition     = !var.use_existing_vpc || !var.create_vpc_endpoints || length(var.existing_private_route_table_ids) > 0
      error_message = "existing_private_route_table_ids must be set when use_existing_vpc and create_vpc_endpoints are both true."
    }
  }
}

# VPC Endpoints for AWS services
resource "aws_vpc_endpoint" "sts" {
  count = var.create_vpc_endpoints ? 1 : 0

  vpc_id             = local.selected_vpc_id
  service_name       = "${local.interface_endpoint_prefix}.${data.aws_region.current.region}.sts"
  vpc_endpoint_type  = "Interface"
  subnet_ids         = local.selected_private_subnet_ids
  security_group_ids = [aws_security_group.vpc_endpoints[0].id]

  private_dns_enabled = true
}

resource "aws_vpc_endpoint" "s3" {
  count = var.create_vpc_endpoints ? 1 : 0

  vpc_id            = local.selected_vpc_id
  service_name      = "${local.gateway_endpoint_prefix}.${data.aws_region.current.region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = local.selected_private_route_table_ids
}

# Security group for VPC endpoints
resource "aws_security_group" "vpc_endpoints" {
  count = var.create_vpc_endpoints ? 1 : 0

  name        = "${var.name}-vpc-endpoints"
  description = "Security group for VPC interface endpoints allowing HTTPS from within VPC"
  vpc_id      = local.selected_vpc_id

  ingress {
    description = "Allow HTTPS from VPC CIDR for AWS service endpoints"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [local.selected_vpc_cidr_block]
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${var.name}-vpc-endpoints"
    }
  )
}
