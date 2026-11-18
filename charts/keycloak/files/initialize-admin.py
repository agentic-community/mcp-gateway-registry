"""Reconcile the admin realm through the trusted, in-cluster proxy boundary."""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Keycloak initialization refused an unexpected redirect")


def main():
    base = os.environ["KEYCLOAK_ADMIN_URL"].rstrip("/")
    realm = urllib.parse.quote(os.environ["KEYCLOAK_ADMIN_REALM"], safe="")
    hostname = urllib.parse.urlsplit(os.environ["KEYCLOAK_HOSTNAME"]).netloc
    # This Job is a trusted reverse-proxy peer on the private service, just as
    # the gateway is. HTTPS metadata permits master authentication even when
    # the cluster's pod CIDR is not considered private by Keycloak (100.64/10).
    headers = {
        "X-Forwarded-Proto": "https",
        "X-Forwarded-Port": "443",
        "X-Forwarded-Host": hostname,
        "Forwarded": 'proto=https;host="' + hostname + '"',
    }
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(path, method="GET", data=None, content_type="application/json"):
        req = urllib.request.Request(
            base + path,
            data=data,
            headers={**headers, "Content-Type": content_type},
            method=method,
        )
        with opener.open(req, timeout=10) as response:
            body = response.read()
            return json.loads(body) if body else None

    form = urllib.parse.urlencode(
        {
            "grant_type": "password",
            "client_id": "admin-cli",
            "username": os.environ["KEYCLOAK_ADMIN_USERNAME"],
            "password": os.environ.pop("KEYCLOAK_ADMIN_PASSWORD"),
        }
    ).encode()
    deadline = time.monotonic() + 600
    while True:
        try:
            token = request(
                f"/realms/{realm}/protocol/openid-connect/token",
                "POST",
                form,
                "application/x-www-form-urlencoded",
            )
            break
        except urllib.error.HTTPError as exc:
            # Wrong credentials are not a startup delay; retrying them can lock
            # the administrator out. Never include provider response bodies.
            if exc.code not in (404, 502, 503, 504):
                raise RuntimeError(
                    f"Keycloak admin authentication failed (HTTP {exc.code})"
                ) from None
        except (urllib.error.URLError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("Keycloak admin initialization timed out")
        time.sleep(5)

    del form
    headers["Authorization"] = "Bearer " + token["access_token"]
    path = f"/admin/realms/{realm}"
    if str(request(path).get("sslRequired", "")).lower() != "none":
        request(path, "PUT", json.dumps({"sslRequired": "none"}).encode())
    if str(request(path).get("sslRequired", "")).lower() != "none":
        raise RuntimeError("Keycloak admin realm SSL policy readback did not match")
    print("Keycloak admin realm initialization verified")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Credentials, bearer tokens and provider bodies must not enter logs.
        if isinstance(exc, RuntimeError):
            print(str(exc), file=sys.stderr)
        elif isinstance(exc, urllib.error.HTTPError):
            print(f"Keycloak admin reconciliation failed (HTTP {exc.code})", file=sys.stderr)
        else:
            print(
                "Keycloak admin reconciliation failed; check service and configuration",
                file=sys.stderr,
            )
        sys.exit(1)
