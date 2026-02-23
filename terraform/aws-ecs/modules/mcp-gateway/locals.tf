# Local values for MCP Gateway Registry Module

locals {
  name_prefix = var.name

  # Determine protocol based on whether TLS is configured (certificate_arn or domain_name set)
  https_enabled = var.certificate_arn != "" || var.domain_name != ""
  protocol      = local.https_enabled ? "https" : "http"

  # Full Keycloak URL with correct protocol
  keycloak_url = var.keycloak_domain != "" ? "${local.protocol}://${var.keycloak_domain}" : ""

  common_tags = merge(
    {
      stack     = var.name
      component = "mcp-gateway-registry"
    },
    var.additional_tags
  )
}