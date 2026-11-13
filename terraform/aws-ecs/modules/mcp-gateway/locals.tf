# Local values for MCP Gateway Registry Module

locals {
  name_prefix = var.name

  common_tags = merge(
    {
      stack     = var.name
      component = "mcp-gateway-registry"
    },
    var.additional_tags
  )

  # Flatten the operator's embeddings header map into the fixed
  # EMBEDDINGS_EXTRA_HEADER_<N> slots the registry reads (issue #1819). Keys are
  # sorted so slot numbers are stable: an unordered walk could renumber slots on
  # an unrelated apply and show a spurious task-definition diff.
  embeddings_extra_header_env = [
    for index, header_name in sort(keys(var.embeddings_extra_headers)) : {
      name  = "EMBEDDINGS_EXTRA_HEADER_${index + 1}"
      value = "${header_name}: ${var.embeddings_extra_headers[header_name]}"
    }
  ]

  # OpenAPI MCP servers (issue #2022). The registry (and the nginx inside its
  # container) reach the openapi-mcp service through this Service Connect name.
  openapi_mcp_port         = 9876
  openapi_mcp_internal_url = "http://openapi-mcp:${local.openapi_mcp_port}"

  # The forward-proxy variables are standard ones that operators pass through
  # registry_extra_env. Copy the same entries into the openapi-mcp task so its
  # SSRF guard routes egress exactly as the registry's does.
  openapi_mcp_forward_proxy_env = [
    for entry in var.registry_extra_env : entry
    if contains(["HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"], upper(trimspace(entry.name)))
  ]
}
