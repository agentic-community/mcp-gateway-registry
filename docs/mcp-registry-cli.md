# MCP Registry CLI Guide

Interactive terminal interface for chatting with AI models and using MCP (Model Context Protocol) tools.



## Table of Contents
- [Quick Start](#quick-start)
- [Setup](#setup)
- [Available Commands](#available-commands)
- [Provider Selection](#provider-selection)
- [Available Models](#available-models)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Build
cd cli && npm install && npm run build

# 2. Configure AI provider (choose one):
export AWS_PROFILE=your-profile              # Bedrock via AWS profile
export ANTHROPIC_API_KEY=sk-ant-xxx          # Anthropic API

# 3. Run (OAuth tokens auto-generated on first start)
npm start
```

**Default model:** Claude Haiku 4.5 (fastest/cheapest)

**Change model:**
```bash
export BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0  # Bedrock
export ANTHROPIC_MODEL=claude-opus-4-20250514                          # Anthropic API
```

---

## Setup

### Prerequisites

1. **Build the CLI:** `cd cli && npm install && npm run build`
2. **Configure AI Provider:** Set AWS or Anthropic credentials (see below)
3. **OAuth Tokens:** Auto-generated on first run (stored in `.oauth-tokens/ingress.json`)

### AI Provider Configuration

**Option 1: Amazon Bedrock - EC2 Execution Role** (Recommended for EC2)
```bash
# No configuration needed - AWS SDK auto-detects credentials from IMDS
npm start
```
Requirements: EC2 instance with IAM role having `bedrock:InvokeModel` permission

**Option 2: Amazon Bedrock - AWS Profile**
```bash
export AWS_PROFILE=your-profile
export AWS_REGION=us-west-2  # Optional
npm start
```

**Option 3: Amazon Bedrock - Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
npm start
```

**Option 4: Anthropic API**
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key
npm start
```

### Status Footer

Shows real-time status at the bottom:
```
Token: Valid for 5m 23s | Source: ingress-json | Last refresh: 14:32:15 | Model: us.anthropic.claude-haiku-4-5-20251001-v1:0 | Tokens: In: 1,234 | Out: 567 | Cost: $0.01
```

- **Token:** Time remaining (green > 60s, yellow < 60s, red when expired) - auto-refreshes at < 10s
- **Source:** Token origin (`ingress-json`, `env`, `token-file`)
- **Model:** Current AI model
- **Tokens:** Input/output usage for session
- **Cost:** Estimated session cost

---

## Available Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/exit` | Exit CLI (or Ctrl+C) |
| `/ping` | Test gateway connectivity |
| `/list` | List MCP tools |
| `/servers` | List MCP servers |
| `/refresh` | Manually refresh OAuth tokens |
| `/retry` | Retry authentication |

**Tip:** Type `/` for autocomplete suggestions

---

## Provider Selection

**Priority (first match wins):**
1. **Amazon Bedrock** - If AWS credentials found (`AWS_PROFILE`, `AWS_ACCESS_KEY_ID`, or EC2 execution role)
2. **Anthropic API** - If `ANTHROPIC_API_KEY` set

**Force Anthropic API when both available:**
```bash
unset AWS_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
export ANTHROPIC_API_KEY=sk-ant-your-key
npm start
```

---

## Available Models

### Amazon Bedrock (use `BEDROCK_MODEL_ID`)

**Claude 4+ (Inference Profile IDs):**
| Model ID | Best For | Cost |
|----------|----------|------|
| `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Fast, efficient (default) | $$ |
| `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | Balanced | $$$ |
| `us.anthropic.claude-opus-4-1-20250805-v1:0` | Most capable | $$$$ |
| `global.anthropic.claude-sonnet-4-5-20250929-v1:0` | Multi-region routing | $$$ |

**Note:** Claude 4+ requires inference profile IDs (prefix: `us.anthropic.*` or `global.anthropic.*`)

List available: `aws bedrock list-inference-profiles --region us-west-2`

### Anthropic API (use `ANTHROPIC_MODEL`)

| Model ID | Best For |
|----------|----------|
| `claude-haiku-4-5-20251001` | Fast, efficient (default) |
| `claude-sonnet-4-20250514` | Balanced |
| `claude-opus-4-20250514` | Most capable |

Docs: https://docs.anthropic.com/en/docs/about-claude/models


---

## Troubleshooting

### OAuth Token Issues

**Error:** "Failed to load ingress tokens" or authentication errors

**Fix:**
1. **Auto-generate:** Run `npm start` - tokens auto-generate on first run
2. **Manual refresh:** Type `/refresh` in running CLI
3. **Manual generation:** `./credentials-provider/generate_creds.sh --ingress-only`

**Note:** Tokens stored in `.oauth-tokens/ingress.json` (project root). Auto-refresh at < 10s remaining.

### Build Errors

**Fix:**
```bash
cd cli && rm -rf dist/ node_modules/ && npm install && npm run build
```

### "Agent mode is disabled"

**Cause:** No AI credentials found

**Fix:**
```bash
# Bedrock - verify AWS credentials
aws sts get-caller-identity

# Bedrock - EC2 execution role (check IAM role attached)
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/

# Anthropic API
echo $ANTHROPIC_API_KEY  # Should show key
export ANTHROPIC_API_KEY=sk-ant-your-key
```

### Bedrock Access Denied

**Cause:** Missing `bedrock:InvokeModel` permission or model not enabled

**Fix:**
```bash
aws sts get-caller-identity  # Check identity
aws bedrock list-inference-profiles --region us-west-2  # List available models
```

Contact admin to grant Bedrock permissions or enable model access.

### Model Not Found

**Cause:** Wrong model ID format for Claude 4+

**Fix:**
```bash
# ❌ Wrong
export BEDROCK_MODEL_ID=anthropic.claude-sonnet-4-5-20250929-v1:0

# ✅ Correct (use inference profile ID)
export BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

### Anthropic API Errors

**Rate limit (429):** Wait and retry, or use Bedrock
**Auth failed (401):** Verify `ANTHROPIC_API_KEY` is valid (starts with `sk-ant-`)

---

## Environment Variables

### Amazon Bedrock
```bash
AWS_PROFILE=your-profile           # AWS profile (recommended)
AWS_ACCESS_KEY_ID=xxx              # Access key
AWS_SECRET_ACCESS_KEY=xxx          # Secret key
AWS_REGION=us-west-2               # Region (default: us-east-1)
BEDROCK_MODEL_ID=us.anthropic...   # Override model
```

### Anthropic API
```bash
ANTHROPIC_API_KEY=sk-ant-xxx       # API key (required)
ANTHROPIC_MODEL=claude-opus-4...   # Override model
```

---

## Resources

- **Bedrock:** https://docs.aws.amazon.com/bedrock/
- **Anthropic API:** https://docs.anthropic.com/
- **API Console:** https://console.anthropic.com/
