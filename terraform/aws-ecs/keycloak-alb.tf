#
# Keycloak Application Load Balancer
#

resource "aws_lb" "keycloak" {
  #checkov:skip=CKV_AWS_150:Deletion protection managed at deployment level
  #checkov:skip=CKV2_AWS_76:Target group is attached to ALB listener and ECS service
  name               = "keycloak-alb"
  internal           = false
  load_balancer_type = "application"

  drop_invalid_header_fields = true
  enable_deletion_protection = false

  security_groups = concat(
    [aws_security_group.keycloak_lb.id],
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

# HTTPS Listener (only when Route53 DNS is enabled with ACM certificate)
resource "aws_lb_listener" "keycloak_https" {
  count             = var.enable_route53_dns ? 1 : 0
  load_balancer_arn = aws_lb.keycloak.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = aws_acm_certificate.keycloak[0].arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.keycloak.arn
  }

  tags = local.common_tags
}

# HTTP Listener - behavior depends on deployment mode
# Mode 2 (Custom Domain → ALB): redirect to HTTPS
# Mode 1 & 3 (CloudFront enabled): forward to target (CloudFront handles HTTPS)
resource "aws_lb_listener" "keycloak_http" {
  load_balancer_arn = aws_lb.keycloak.arn
  port              = "80"
  protocol          = "HTTP"

  # Redirect to HTTPS only when Route53 is enabled WITHOUT CloudFront (Mode 2)
  # When CloudFront is enabled (Mode 1 or 3), forward to target group
  default_action {
    type             = var.enable_route53_dns && !var.enable_cloudfront ? "redirect" : "forward"
    target_group_arn = var.enable_route53_dns && !var.enable_cloudfront ? null : aws_lb_target_group.keycloak.arn

    dynamic "redirect" {
      for_each = var.enable_route53_dns && !var.enable_cloudfront ? [1] : []
      content {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }
  }

  tags = local.common_tags
}
