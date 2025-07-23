# Google OAuth Setup Guide for MCP Gateway Registry

This guide explains how to configure the registry to authenticate users and agents using **Google OAuth 2.0**. The same flow applies to other OpenID Connect providers.

## Overview

The MCP Gateway Registry supports two authentication patterns:

- **Agent Uses User Identity** – agents act on behalf of a human user using OAuth 2.0 PKCE.
- **Agent Uses Its Own Identity** – agents obtain their own access token using the client credentials flow.

Google provides OAuth credentials via the [Google Cloud Console](https://console.cloud.google.com/). The resulting client ID and secret are used by the auth server and agents.

## Google Cloud Configuration

1. **Create an OAuth Consent Screen** in your Google Cloud project.
2. **Create OAuth Credentials** → **Web Application**.
   - Add the following authorized redirect URIs:
     - `http://localhost:9090/callback`
     - `http://localhost/oauth2/callback/google`
     - `http://localhost:8888/oauth2/callback/google`
     - `https://your-domain.com/oauth2/callback/google`
   - Note the **Client ID** and **Client Secret**.
3. (Optional) **Service Accounts** can be used for non-interactive agents. Generate a service account key and exchange it for an OAuth token using Google APIs.

## Environment Configuration Examples

Set the following variables in your `.env` file:

```bash
COGNITO_CLIENT_ID=<your Google OAuth client ID>
COGNITO_CLIENT_SECRET=<your Google OAuth client secret>
COGNITO_USER_POOL_ID=<google-project-id>
AWS_REGION=<unused>
```

The variable names remain for compatibility with existing scripts.

## Testing and Troubleshooting

After updating the configuration, run `./build_and_run.sh` and authenticate via the "Login with Google" button. For CLI authentication see `agents/cli_user_auth.py`.
