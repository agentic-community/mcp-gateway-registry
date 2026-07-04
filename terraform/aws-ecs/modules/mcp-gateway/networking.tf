# Networking resources for MCP Gateway Registry

# Service Discovery Namespace
resource "aws_service_discovery_private_dns_namespace" "mcp" {
  name        = "${local.name_prefix}.local"
  description = "Service discovery namespace for MCP Gateway Registry"
  vpc         = var.vpc_id
  tags        = local.common_tags
}

# CloudFront managed prefix list (for allowing CloudFront or other CDN IPs)
# Default prefix list is AWS CloudFront origin-facing IPs (com.amazonaws.global.cloudfront.origin-facing)
data "aws_ec2_managed_prefix_list" "cloudfront" {
  count = var.cloudfront_prefix_list_name != "" ? 1 : 0
  name  = var.cloudfront_prefix_list_name
}

# Separate security group for CloudFront prefix list ingress
# This avoids hitting the 60 rules per security group limit since the CloudFront
# prefix list has ~55 reserved entries that count against the quota
resource "aws_security_group" "alb_cloudfront" {
  #checkov:skip=CKV2_AWS_5:Security group is attached to ALB via security_groups parameter
  count       = var.cloudfront_prefix_list_name != "" ? 1 : 0
  name        = "${local.name_prefix}-alb-cloudfront"
  description = "Security group for CloudFront access to MCP Gateway ALB"
  vpc_id      = var.vpc_id

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-alb-cloudfront"
    }
  )
}

resource "aws_security_group_rule" "alb_cloudfront_ingress_http" {
  count             = var.cloudfront_prefix_list_name != "" ? 1 : 0
  description       = "Ingress from CloudFront prefix list to ALB (HTTP)"
  type              = "ingress"
  from_port         = 80
  to_port           = 80
  protocol          = "tcp"
  prefix_list_ids   = [data.aws_ec2_managed_prefix_list.cloudfront[0].id]
  security_group_id = aws_security_group.alb_cloudfront[0].id
}

resource "aws_security_group_rule" "alb_cloudfront_egress" {
  #checkov:skip=CKV_AWS_382:ALB security group requires unrestricted egress to reach ECS tasks and health checks
  count             = var.cloudfront_prefix_list_name != "" ? 1 : 0
  description       = "Egress to all"
  type              = "egress"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.alb_cloudfront[0].id
}

# Main Application Load Balancer (for registry, auth, gradio)
module "alb" {
  #checkov:skip=CKV_TF_1:Module version is pinned via version constraint
  source  = "terraform-aws-modules/alb/aws"
  version = "~> 9.0"

  name                       = "${local.name_prefix}-alb"
  load_balancer_type         = "application"
  internal                   = var.alb_scheme == "internal"
  enable_deletion_protection = false

  vpc_id  = var.vpc_id
  subnets = var.alb_scheme == "internal" ? var.private_subnet_ids : var.public_subnet_ids

  # Attach additional security groups (CloudFront SG when enabled)
  # This keeps CloudFront prefix list rules in a separate SG to avoid the 60 rules/SG limit
  security_groups = var.cloudfront_prefix_list_name != "" ? [aws_security_group.alb_cloudfront[0].id] : []

  # Enable access logs
  access_logs = {
    bucket  = var.alb_logs_bucket
    enabled = true
  }

  # Security Groups
  # Create dynamic ingress rules for each CIDR block and port combination
  # Note: CloudFront prefix list is in a separate SG (alb_cloudfront) to avoid rules limit
  security_group_ingress_rules = merge(
    merge([
      for idx, cidr in var.ingress_cidr_blocks : {
        "http_${idx}" = {
          from_port   = 80
          to_port     = 80
          ip_protocol = "tcp"
          cidr_ipv4   = cidr
        }
        "https_${idx}" = {
          from_port   = 443
          to_port     = 443
          ip_protocol = "tcp"
          cidr_ipv4   = cidr
        }
        # NOTE: the public :8888 (auth) and :7860 (gradio) raw-app ALB exposures
        # were removed (registry internal-auth hardening). The auth-server is
        # reached internally via Service Connect (auth-server:8888 -> container
        # 18888), and external OAuth URLs resolve to the nginx-fronted :443/:80 --
        # neither needs a public raw-app listener. The registry app (:7860) is
        # nginx-fronted and loopback-bound. Only http/https remain.
      }
    ]...),
    {
    }
  )
  security_group_egress_rules = {
    all = {
      ip_protocol = "-1"
      cidr_ipv4   = "0.0.0.0/0"
    }
  }

  listeners = merge(
    {
      http = {
        port     = 80
        protocol = "HTTP"
        forward = {
          target_group_key = "registry"
        }
      }
      # The public auth (:8888) and gradio (:7860) raw-app listeners were removed
      # (registry internal-auth hardening). Internal callers use Service Connect;
      # external traffic uses the nginx-fronted http/https listeners.
    },
    var.enable_https ? {
      https = {
        port            = 443
        protocol        = "HTTPS"
        certificate_arn = var.certificate_arn
        ssl_policy      = "ELBSecurityPolicy-TLS13-1-2-2021-06"
        forward = {
          target_group_key = "registry"
        }
      }
    } : {}
  )

  target_groups = {
    registry = {
      backend_protocol                  = "HTTP"
      backend_port                      = 8080
      target_type                       = "ip"
      deregistration_delay              = 5
      load_balancing_cross_zone_enabled = true

      health_check = {
        enabled             = true
        healthy_threshold   = 2
        interval            = 30
        matcher             = "200"
        path                = "/health"
        port                = 8080
        protocol            = "HTTP"
        timeout             = 5
        unhealthy_threshold = 2
      }

      create_attachment = false
    }
    # The "auth" (:18888) and "gradio" (:7860) target groups were removed with
    # their public listeners (registry internal-auth hardening). The registry
    # target group (nginx :8080) is the only ALB attachment; auth-server is
    # reached via Service Connect, not the ALB.
  }

  tags = local.common_tags
}
