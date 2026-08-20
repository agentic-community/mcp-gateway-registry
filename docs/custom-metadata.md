# Custom Metadata for Servers & Agents

Enrich your MCP servers and agents with custom metadata for organization, compliance tracking, and integration purposes. All metadata is fully searchable via semantic search.

## Use Cases

### Organization & Team Management

```json
{
  "team": "data-platform",
  "owner": "alice@example.com",
  "department": "engineering"
}
```
*Search by: "team:data-platform servers", "alice@example.com owned services"*

### Compliance & Governance

```json
{
  "compliance_level": "PCI-DSS",
  "data_classification": "confidential",
  "regulatory_requirements": ["GDPR", "HIPAA"],
  "audit_logging": true
}
```
*Search by: "PCI-DSS compliant servers", "HIPAA regulated services"*

### Cost & Project Tracking

```json
{
  "cost_center": "analytics-dept",
  "project_code": "AI-2024-Q1",
  "budget_allocation": "R&D"
}
```
*Search by: "cost center analytics", "project AI-2024-Q1"*

### Deployment & Integration

```json
{
  "deployment_region": "us-east-1",
  "environment": "production",
  "jira_ticket": "MCPGW-123",
  "version": "2.1.0"
}
```
*Search by: "us-east-1 deployed services", "JIRA MCPGW-123", "version 2.1.0"*

## API Usage

### Register MCP Server with Metadata

```bash
curl -X POST https://registry.example.com/api/services/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "payment-processor",
    "description": "Payment processing service",
    "path": "/payment-processor",
    "proxy_pass_url": "http://payment:8080",
    "metadata": {
      "team": "finance-platform",
      "owner": "alice@example.com",
      "compliance_level": "PCI-DSS",
      "cost_center": "finance-ops",
      "deployment_region": "us-east-1"
    }
  }'
```

### Register A2A Agent with Metadata

```bash
curl -X POST https://registry.example.com/api/agents/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "analytics-agent",
    "description": "Data analytics agent",
    "metadata": {
      "team": "data-science",
      "owner": "bob@example.com",
      "version": "3.2.1",
      "cost_center": "analytics-dept"
    }
  }'
```

### Search by Metadata

```bash
# Find servers by team
curl "https://registry.example.com/api/search?q=team:finance-platform"

# Find PCI-DSS compliant services
curl "https://registry.example.com/api/search?q=PCI-DSS compliant services"

# Find services by owner
curl "https://registry.example.com/api/search?q=alice@example.com owned"

# Find services in specific region
curl "https://registry.example.com/api/search?q=us-east-1 deployed"
```

## Key Features

- **Flexible Schema:** Store any JSON-serializable data (strings, numbers, booleans, nested objects, arrays)
- **Fully Searchable:** All metadata included in semantic search embeddings
- **Backward Compatible:** Optional field - existing registrations work without modification
- **Type-Safe:** Pydantic validation ensures data integrity
- **REST API:** Full CRUD support via standard API endpoints

## Related Documentation

- [Service Management Guide](service-management.md)
- [A2A Agent Guide](a2a.md)
- [Semantic Search](design/hybrid-search-architecture.md)
- [Metadata Field Projection](metadata-field-projection.md) — return only selected metadata fields on list and search responses
- [FAQ: return only the metadata fields I need when listing or searching](faq/metadata-field-projection-search-listing.md)
