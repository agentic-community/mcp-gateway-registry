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
}
