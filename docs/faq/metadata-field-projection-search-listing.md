# How do I return only the metadata fields I need when listing or searching?

When you list or search servers, agents, or skills, the response includes each asset's full `metadata` subdocument. If that metadata holds large or deeply nested objects you don't need, add the `metadata_fields` parameter to project only the paths you care about. This shrinks the payload without changing any other field in the response.

Omit the parameter and responses are unchanged, so this is a safe, opt-in addition to any existing integration.

## The quick version

Pass a comma-separated list of dot-notation paths:

```bash
# List: only the "owner" and the nested config.region metadata fields
curl "http://localhost/api/servers?metadata_fields=owner,config.region" \
  -H "Authorization: Bearer $TOKEN"
```

```json
"metadata": { "owner": "team-platform", "config": { "region": "us-east-1" } }
```

Everything outside `metadata` (name, description, tags, endpoint, etc.) is always returned in full.

## Which endpoints support it?

Listing and single-GET for all three asset types, plus semantic search:

| Where | How to pass it |
|-------|----------------|
| `GET /api/servers`, `GET /api/servers/{path}` | query param |
| `GET /api/agents`, `GET /api/agents/{path}` | query param |
| `GET /api/skills`, `GET /api/skills/{path}` | query param |
| `POST /api/search/semantic` | `metadata_fields` field in the JSON body |

## Search example

```bash
curl -X POST "http://localhost/api/search/semantic" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"query": "deployment tools", "metadata_fields": "owner,config.region"}'
```

Each returned asset's `metadata` is projected to the requested paths. An asset whose metadata does not contain a requested path simply returns `{}` (or `null` if it has no metadata at all) for that field — never an error.

## Two ways to pass multiple fields

Comma-separated and repeated query params are equivalent, and you can mix them:

```
?metadata_fields=owner,config.region
?metadata_fields=owner&metadata_fields=config.region
?metadata_fields=owner,config.region&metadata_fields=limits.rps
```

## From the CLI

```bash
registry_management.py list        --metadata-fields "owner,config.region" --json
registry_management.py agent-list  --metadata-fields "team" --json
registry_management.py skill-list  --metadata-fields "owner" --json
registry_management.py server-search --query "tools" --metadata-fields "owner"
```

## A few behaviors worth knowing

- **Nested paths** use dots: `config.region` returns `{"config": {"region": ...}}`.
- **Ancestor wins:** if you request both `config` and `config.region`, you get the full `config` subtree (the ancestor already includes the descendant).
- **Missing paths are dropped silently**, not errored — so you can request a superset of fields across a heterogeneous result set.
- **Backward compatible:** leaving `metadata_fields` off returns full metadata exactly as before.

## What gets rejected (HTTP 422)

Input is validated at the API boundary and fails closed with a descriptive message:

| Rule | Example that is rejected |
|------|--------------------------|
| At most 20 paths | 21 comma-separated fields |
| At most 5 levels deep | `a.b.c.d.e.f` |
| Segment at most 64 characters | a single 65-char segment |
| No `$`-prefixed segment | `$set.injection` |
| No empty segment | `config..region` |
| Allowed characters only (letters incl. accented/non-Latin, digits, `_`, `-`) | `owner name`, `re@gion`, `own$er` |

## See also

- [Metadata Field Projection](../metadata-field-projection.md) — full reference, including the database-level projection and performance notes.
- [Custom Metadata](../custom-metadata.md) — how to attach metadata to assets in the first place.
