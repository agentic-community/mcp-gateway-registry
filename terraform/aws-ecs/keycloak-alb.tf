#
# Keycloak Application Load Balancer
#

resource "aws_lb" "keycloak" {
  count = local.keycloak_selected ? 1 : 0

  #checkov:skip=CKV_AWS_150:Deletion protection managed at deployment level
  #checkov:skip=CKV2_AWS_76:Target group is attached to ALB listener and ECS service
  name               = "keycloak-alb"
  internal           = false
  load_balancer_type = "application"

  drop_invalid_header_fields = true
  enable_deletion_protection = false

  security_groups = concat(
    [aws_security_group.keycloak_lb[0].id],
    local.cloudfront_prefix_list_name != "" ? [aws_security_group.keycloak_lb_cloudfront[0].id] : []
  )
  subnets = local.selected_public_subnet_ids

  access_logs {
    bucket  = aws_s3_bucket.alb_logs.id
    enabled = true
    prefix  = "keycloak-alb"
  }

  tags = merge(
    local.common_tags,
    {
      Name = "keycloak-alb"
    }
  )

  # Wait for S3 bucket policy to propagate (30s delay)
  # This prevents "Access Denied" errors when ALB tests write permissions
  depends_on = [time_sleep.wait_for_bucket_policy]
}

# Random suffix for target group name (required by AWS)
resource "random_string" "alb_tg_suffix" {
  length  = 3
  special = false
  upper   = false
}

# Target Group
resource "aws_lb_target_group" "keycloak" {
  count = local.keycloak_selected ? 1 : 0

  #checkov:skip=CKV_AWS_378:HTTP backend protocol is intentional - TLS terminates at ALB
  name                 = "keycloak-tg-${random_string.alb_tg_suffix.result}"
  port                 = 8080
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = local.selected_vpc_id
  deregistration_delay = 30

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    path                = "/"
    matcher             = "200-399"
    protocol            = "HTTP"
  }

  stickiness {
    type            = "lb_cookie"
    enabled         = true
    cookie_duration = 86400
  }

  tags = merge(
    local.common_tags,
    {
      Name = "keycloak-tg"
    }
  )

  lifecycle {
    create_before_destroy = true
    ignore_changes = [
      stickiness[0].cookie_name
    ]
  }
}

# Path routing on the listeners that reach Keycloak.
#
# Only these paths are forwarded; every other path gets a fixed 404:
#   - the gateway realm (/realms/<realm>, /realms/<realm>/*) and /resources/*
#     (login pages, OIDC endpoints, DCR, static assets) from any source the
#     security groups admit;
#   - the exact root path "/", which Keycloak answers with a redirect to
#     /admin/; readiness probes such as infra/scripts/post-deploy.sh use it;
#   - the admin paths (/admin, /admin/*, /realms/master, /realms/master/*) only
#     from keycloak_admin_cidr_blocks plus the NAT gateway egress IPs. The
#     registry calls the admin API through the public Keycloak URL, so its NAT
#     egress IPs must always be allowed.
#
# With CloudFront in front (Mode 1 and 3) the ALB sees CloudFront edge IPs, not
# the client, so a source-ip condition would block the registry's own admin
# calls. In those modes the admin paths are forwarded without a source-ip
# condition, and a non-default keycloak_admin_cidr_blocks fails the plan.
#
# ALB limits: at most 3 values per condition and 5 per rule. Each admin rule
# has a 2-value path-pattern and up to 3 source CIDRs, so the CIDR lists are
# split into chunks of 3 with one rule pair per chunk. A listener holds at most
# 100 rules by default, which caps the admin list at roughly 140 CIDRs.
locals {
  keycloak_gateway_realm = "mcp-gateway"

  keycloak_public_paths = [
    "/realms/${local.keycloak_gateway_realm}",
    "/realms/${local.keycloak_gateway_realm}/*",
    "/resources/*",
  ]

  keycloak_admin_path_groups = [
    ["/admin", "/admin/*"],
    ["/realms/master", "/realms/master/*"],
  ]

  keycloak_admin_cidr_blocks = (
    var.keycloak_admin_cidr_blocks == null ? var.ingress_cidr_blocks : var.keycloak_admin_cidr_blocks
  )

  # The NAT IPs are chunked separately from the operator's list: their values
  # are unknown until the EIPs exist, and keeping them apart avoids a dedup step
  # that would make the rule count unknown at plan time.
  keycloak_admin_source_cidr_chunks = concat(
    chunklist(distinct(local.keycloak_admin_cidr_blocks), 3),
    chunklist([for ip in local.selected_nat_public_ips : "${ip}/32"], 3),
  )

  # One entry per admin rule. An empty cidrs list means "no source-ip condition"
  # and is produced only in CloudFront modes. In direct modes an empty CIDR list
  # yields no admin rules, so the admin paths fall through to the 404 default.
  keycloak_admin_cidr_chunks = var.enable_cloudfront ? [[]] : local.keycloak_admin_source_cidr_chunks

  keycloak_admin_rules = merge([
    for group_index, paths in local.keycloak_admin_path_groups : {
      for chunk_index, cidrs in local.keycloak_admin_cidr_chunks :
      "admin-${group_index}-${chunk_index}" => {
        paths    = paths
        cidrs    = cidrs
        priority = 100 + group_index * 50 + chunk_index
      }
    }
  ]...)

  keycloak_routing_rules = merge(
    {
      public = {
        paths    = local.keycloak_public_paths
        cidrs    = []
        priority = 10
      }
      root = {
        paths    = ["/"]
        cidrs    = []
        priority = 20
      }
    },
    local.keycloak_admin_rules,
  )

  # Listeners that forward to Keycloak. In Mode 2 the HTTP listener only
  # redirects to HTTPS, so it gets no rules.
  keycloak_routed_listeners = local.keycloak_selected ? merge(
    var.enable_route53_dns ? { https = aws_lb_listener.keycloak_https[0].arn } : {},
    var.enable_route53_dns && !var.enable_cloudfront ? {} : { http = aws_lb_listener.keycloak_http[0].arn },
  ) : {}

  keycloak_listener_rules = merge([
    for listener_key, listener_arn in local.keycloak_routed_listeners : {
      for rule_key, rule in local.keycloak_routing_rules :
      "${listener_key}-${rule_key}" => merge(rule, { listener_arn = listener_arn })
    }
  ]...)
}

# Plan-time checks for keycloak_admin_cidr_blocks. Behind CloudFront the ALB
# cannot see the client IP, so a narrowed admin CIDR list would not be
# enforced: refuse it instead of silently ignoring it.
resource "terraform_data" "keycloak_admin_cidr_settings" {
  count = local.keycloak_selected ? 1 : 0

  input = local.keycloak_admin_cidr_blocks

  lifecycle {
    precondition {
      condition     = alltrue([for cidr in local.keycloak_admin_cidr_blocks : can(cidrhost(cidr, 0))])
      error_message = "keycloak_admin_cidr_blocks entries must be CIDR blocks such as 203.0.113.10/32."
    }

    precondition {
      condition = !var.enable_cloudfront || var.keycloak_admin_cidr_blocks == null || (
        sort(local.keycloak_admin_cidr_blocks) == sort(var.ingress_cidr_blocks)
      )
      error_message = join(" ", [
        "keycloak_admin_cidr_blocks cannot be enforced when enable_cloudfront = true:",
        "the Keycloak ALB sees CloudFront edge IPs, not the client IP.",
        "Remove keycloak_admin_cidr_blocks (or set it to ingress_cidr_blocks).",
      ])
    }

    precondition {
      condition     = 2 + length(local.keycloak_admin_path_groups) * length(local.keycloak_admin_cidr_chunks) <= 100
      error_message = "keycloak_admin_cidr_blocks needs more than the 100 rules an ALB listener holds by default. Use fewer, wider CIDR blocks."
    }
  }
}

# HTTPS Listener (only when Route53 DNS is enabled with ACM certificate)
resource "aws_lb_listener" "keycloak_https" {
  count             = local.keycloak_selected && var.enable_route53_dns ? 1 : 0
  load_balancer_arn = aws_lb.keycloak[0].arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = aws_acm_certificate.keycloak[0].arn

  # Paths not matched by a keycloak_routing rule get a 404.
  default_action {
    type = "fixed-response"

    fixed_response {
      content_type = "text/plain"
      message_body = "Not Found"
      status_code  = "404"
    }
  }

  tags = local.common_tags
}

# HTTP Listener - behavior depends on deployment mode
# Mode 2 (Custom Domain → ALB): redirect to HTTPS
# Mode 1 & 3 (CloudFront enabled) and plain HTTP: path rules forward to the
# target group (CloudFront handles HTTPS); unmatched paths get a 404
resource "aws_lb_listener" "keycloak_http" {
  count = local.keycloak_selected ? 1 : 0

  load_balancer_arn = aws_lb.keycloak[0].arn
  port              = "80"
  protocol          = "HTTP"

  # Redirect to HTTPS only when Route53 is enabled WITHOUT CloudFront (Mode 2)
  default_action {
    type = var.enable_route53_dns && !var.enable_cloudfront ? "redirect" : "fixed-response"

    dynamic "redirect" {
      for_each = var.enable_route53_dns && !var.enable_cloudfront ? [1] : []
      content {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }

    dynamic "fixed_response" {
      for_each = var.enable_route53_dns && !var.enable_cloudfront ? [] : [1]
      content {
        content_type = "text/plain"
        message_body = "Not Found"
        status_code  = "404"
      }
    }
  }

  tags = local.common_tags
}

resource "aws_lb_listener_rule" "keycloak_routing" {
  for_each = local.keycloak_listener_rules

  listener_arn = each.value.listener_arn
  priority     = each.value.priority

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.keycloak[0].arn
  }

  condition {
    path_pattern {
      values = each.value.paths
    }
  }

  dynamic "condition" {
    for_each = length(each.value.cidrs) > 0 ? [each.value.cidrs] : []
    content {
      source_ip {
        values = condition.value
      }
    }
  }

  tags = local.common_tags

  depends_on = [terraform_data.keycloak_admin_cidr_settings]
}

# Preserve resource identity when upgrading an active Keycloak deployment.
moved {
  from = aws_lb.keycloak
  to   = aws_lb.keycloak[0]
}

moved {
  from = aws_lb_target_group.keycloak
  to   = aws_lb_target_group.keycloak[0]
}

moved {
  from = aws_lb_listener.keycloak_http
  to   = aws_lb_listener.keycloak_http[0]
}

moved {
  from = terraform_data.keycloak_admin_cidr_settings
  to   = terraform_data.keycloak_admin_cidr_settings[0]
}
