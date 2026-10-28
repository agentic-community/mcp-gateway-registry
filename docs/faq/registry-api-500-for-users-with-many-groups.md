# Why do registry API calls return 500 for users in many IdP groups?

## Question

Some users get `500 Internal Server Error` from registry API calls such as `POST /api/search/semantic`, while other users with the same access succeed. The affected users belong to many IdP groups (for example, more than 150 Entra ID groups). The registry log shows no error for the failed requests. What is happening?

## Answer

nginx checks every API request with an internal `/validate` subrequest to the auth server and reads the response headers into a fixed buffer. Before issue #1939 was fixed, that buffer was 16 KiB and the response carried the user's groups twice: once in an `X-Groups` header and once inside the signed `X-Internal-Token-Registry` token. About 180 Entra group IDs filled the buffer. nginx then logged `upstream sent too big header`, looked for a `/502.html` page the registry image did not ship, and answered 500. The request never reached the registry, which is why its log stays clean.

The fix sends the groups once, in the signed token, raises the buffer to 64 KiB, and ships the `502.html` page. A browser session's token carries no groups at all, because the registry reads them from the session store. A bearer token's registry token still grows with the groups in the caller's IdP token, so very large group sets still have a limit.

## What you see after the fix

The auth server refuses an oversized response itself and logs why, so the cause is in its log rather than only in nginx's:

| Auth server log line | Client sees | Cause |
|---|---|---|
| `/validate: refusing an oversized internal token` | 500 | The caller's groups or scopes make a registry token over 40 KiB. |
| `/validate: response headers are ... bytes, over the ... budget` | 500 | All `/validate` response headers together exceed 60 KiB. |
| `Entra recovered group set exceeds the safe token size` | 503 | Microsoft Graph returned more than 8 KiB of groups for a token with a group overage marker. |
| `... present but nginx marker missing/mismatched` | 401 | `AUTH_SERVER_NGINX_MARKER_SECRET` differs between the registry and the auth server. |

To check from the registry container, search the nginx error log for `too big header`. A hit there after the fix points at a template that predates it.

## How to reduce the group count

At login and on Graph recovery, the auth server keeps only the groups that some scope maps, or the groups listed in `ALLOWED_IDP_GROUPS` when that is set. Mapping fewer groups shrinks browser sessions and Graph-recovered sets. It does not change a bearer token that carries its groups inline: the registry passes those through unfiltered, because group-restricted visibility (`allowed_groups`) matches against them.

For Entra ID bearer tokens, set the app registration's groups claim to **Groups assigned to the application** (see [Group Filtering](../entra-id-setup.md#group-filtering)), so the token carries only the groups the gateway needs. See [Entra ID setup](../entra-id-setup.md#group-membership-for-user-delegated-bearer-tokens) for how inline, overage and Graph-recovered groups differ, and [the JWT bearer session flow design](../design/session-flow-jwt-bearer.md) for the size limits.
