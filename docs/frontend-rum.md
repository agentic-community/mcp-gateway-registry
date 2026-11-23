# Frontend Real User Monitoring (RUM)

Real User Monitoring (RUM) instruments the real end-user browser: it captures page-load timing, Core Web Vitals, JavaScript errors, and XHR/fetch timing from actual sessions. The MCP Gateway Registry ships a vendor-neutral RUM hook so you can instrument the React admin UI with your own RUM vendor (Splunk/SignalFx, Datadog, New Relic, Grafana Faro, and others) without forking the repository or rebuilding the image. No vendor-specific code lives in our source; you supply the vendor snippet at deploy time through a single configuration value.

> **Trust boundary: this is an admin/deploy-time control only.** `RUM_SNIPPET_B64` injects operator-supplied JavaScript into every user's browser, including before login. Whoever can set this value can run arbitrary JavaScript in every session. Set it only via deployment configuration or a secret store. It must never be exposed through a user-facing API, and it must never be derived from registry data or user input.

## How it works

The frontend `index.html` references a small script served by the registry:

```html
<script src="/rum.js" crossorigin="anonymous"></script>
```

At container startup, the entrypoint decodes `RUM_SNIPPET_B64` and writes the result to `/rum.js` inside the frontend build tree. When the variable is unset (the default), the entrypoint writes an empty stub (a JavaScript comment), so `/rum.js` always returns a valid `200` response with `Content-Type: application/javascript` and RUM stays disabled. FastAPI serves `/rum.js` with a short `Cache-Control: public, max-age=300` so a snippet change after a redeploy propagates quickly.

The RUM beacon traffic goes directly from the browser to your vendor's ingest endpoint. It never transits the registry.

## Base64 recipe

Put your full vendor snippet (the vendor `<script>` tags, including any inline init) into a file, then base64-encode it into a single line:

```bash
# Linux
base64 -w0 my-rum-snippet.html

# macOS
base64 my-rum-snippet.html | tr -d '\n'
```

Use the resulting string as the value of `RUM_SNIPPET_B64` (or the per-surface equivalent below).

## Vendor-neutral example (Splunk RUM)

Any vendor's snippet works the same way: paste the vendor snippet into a file, base64-encode it, and set the variable. The example below uses Splunk (SignalFx) RUM. Replace the `<realm>` placeholder with your Splunk realm (region) and `RUM_ACCESS_TOKEN` with your own token. Do not commit a real token or realm.

```html
<script src="https://cdn.signalfx.com/o11y-gdi-rum/latest/splunk-otel-web.js" crossorigin="anonymous"></script>
<script>
  SplunkRum.init({
    beaconEndpoint: 'https://rum-ingest.<realm>.signalfx.com/v1/rum',
    rumAccessToken: 'RUM_ACCESS_TOKEN',
    applicationName: 'enter-your-application-name',
    deploymentEnvironment: 'enter-your-deployment-env',
    globalAttributes: {
      app_cmdb_id: 'enter-your-app-cmdb',
      app_name: 'enter-your-application-name',
      app_environment: 'enter-your-application-environment',
    },
  });
</script>
```

Datadog, New Relic, and Grafana Faro snippets work identically: they are the same shape of thing (one or two `<script>` tags that go in `<head>`), so the process is the same. Paste the vendor snippet, base64-encode it, and set the variable.

## Secret handling

Many RUM snippets embed a vendor access token (the Splunk example above has `rumAccessToken`). For any token-bearing snippet, use the secret path so the token is stored in a secret manager rather than in plaintext configuration:

- ECS/Terraform: supply the base64 value from AWS Secrets Manager via `registry_rum_snippet_secret_arn`. The task definition sources it with a `secrets`/`valueFrom` reference rather than a plaintext `environment` entry.
- Helm/EKS: use `extraEnv` with a `secretKeyRef` (or an `existingSecret`) so `RUM_SNIPPET_B64` is populated from a Kubernetes Secret.

The plaintext variable (`RUM_SNIPPET_B64` in `.env`, `registry_rum_snippet_b64` in Terraform, `registry.rumSnippetB64` in Helm) is intended for token-free snippets only. Do not place a token-bearing base64 value in plaintext tfvars or a plaintext Helm value.

## Per-surface configuration

The registry reads a single `RUM_SNIPPET_B64` environment variable. Each deployment surface supplies it as follows.

### Docker Compose

Set it in `.env` at the repository root:

```
RUM_SNIPPET_B64=<base64-of-your-snippet>
```

Or set it in `extra_env/registry.env` (the extra-env escape hatch):

```
RUM_SNIPPET_B64=<base64-of-your-snippet>
```

### ECS / Terraform

For a token-free snippet, use the plaintext variable in `terraform.tfvars`:

```hcl
registry_rum_snippet_b64 = "<base64-of-your-snippet>"
```

For a token-bearing snippet, use the Secrets Manager path:

```hcl
registry_rum_snippet_secret_arn = "arn:aws:secretsmanager:...:secret:my-rum-snippet"
```

### Helm / EKS

For a token-free snippet, set the value:

```yaml
registry:
  rumSnippetB64: "<base64-of-your-snippet>"
```

For a token-bearing snippet, source it from a Kubernetes Secret via `extraEnv`:

```yaml
registry:
  extraEnv:
    - name: RUM_SNIPPET_B64
      valueFrom:
        secretKeyRef:
          name: my-rum-secret
          key: rum-snippet-b64
```

## Content-Security-Policy allowances

The registry does not ship a Content-Security-Policy (CSP). If you run your own CSP in front of the UI, you must allow the vendor's script host in `script-src` and the vendor's beacon/ingest host in `connect-src`, or the browser will silently block RUM. For the Splunk example above:

```
script-src 'self' cdn.signalfx.com;
connect-src rum-ingest.<realm>.signalfx.com;
```

Adjust the hosts to match your vendor. This applies only if you add a CSP; it is your responsibility, not the registry's.

The UI itself needs three more allowances under a CSP, with or without RUM:

- The inline script at the top of `index.html` applies the saved light/dark mode and theme before the first paint. Allow it by hash in `script-src`. The hash changes whenever that script changes, so compute it from the built file at deploy time rather than copying a value, for example by extracting the first `<script>` element's contents from `frontend/build/index.html` and running it through `openssl dgst -sha256 -binary | base64`.
- The audit log grid (AG Grid) loads its icon font and one image from `data:` URLs inside its stylesheet, so add `data:` to `font-src` and `img-src`. Its styles come from static files, which `style-src 'self'` covers.
- Deployment branding (see [UI branding](ui-branding.md)) loads `theme.css`, logos and fonts from the registry's own origin, which `style-src 'self'`, `img-src 'self'` and `font-src 'self'` cover. Branding cannot point at other origins for logos or the favicon. A `url()` inside `theme.css` can, so if you allow other origins in `img-src` or `font-src`, those apply to it too.

This policy covers the UI without RUM; add your vendor's hosts as shown above:

```
default-src 'self'; script-src 'self' 'sha256-<hash of the inline script>'; style-src 'self'; font-src 'self' data:; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'
```

## Development-mode note

`RUM_SNIPPET_B64` is a deploy-time control and has no effect under `npm run dev`. The Vite dev server serves the committed stub `frontend/public/rum.js`, so RUM stays disabled in local development regardless of the variable.

The `<script src="/rum.js">` tag is render-blocking in `<head>` by design: RUM vendors require the agent to load early so it can capture page-load timing and errors before other scripts run. Do not add `defer` or `async` to it without first checking your vendor's guidance.
