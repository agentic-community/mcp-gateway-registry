from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

import boto3
import jwt
import requests
from botocore.exceptions import (
    ClientError,
)
from jwt.api_jwk import (
    PyJWK,
)

logger = logging.getLogger(__name__)


def _hash_username(
    username: str,
) -> str:
    if not username:
        return "anonymous"
    return f"user_{hashlib.sha256(username.encode()).hexdigest()[:8]}"


class SimplifiedCognitoValidator:
    """Cognito token validator that does not rely on environment variables."""

    def __init__(
        self,
        region: str = "us-east-1",
        *,
        secret_key: Optional[str] = None,
        jwt_issuer: Optional[str] = None,
        jwt_audience: Optional[str] = None,
    ) -> None:
        self.default_region = region
        self.secret_key = secret_key
        self.jwt_issuer = jwt_issuer
        self.jwt_audience = jwt_audience
        self._cognito_clients: dict[str, Any] = {}
        self._jwks_cache: dict[str, dict[str, Any]] = {}

    def _get_cognito_client(
        self,
        region: str,
    ):
        if region not in self._cognito_clients:
            self._cognito_clients[region] = boto3.client("cognito-idp", region_name=region)
        return self._cognito_clients[region]

    def _get_jwks(
        self,
        user_pool_id: str,
        region: str,
    ) -> dict[str, Any]:
        cache_key = f"{region}:{user_pool_id}"

        if cache_key not in self._jwks_cache:
            issuer = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}"
            jwks_url = f"{issuer}/.well-known/jwks.json"

            try:
                response = requests.get(jwks_url, timeout=10)
                response.raise_for_status()
                jwks = response.json()

                self._jwks_cache[cache_key] = jwks
                logger.debug(
                    "Retrieved JWKS for %s with %s keys",
                    cache_key,
                    len(jwks.get("keys", [])),
                )
            except Exception as exc:
                logger.error("Failed to retrieve JWKS from %s: %s", jwks_url, exc)
                raise ValueError(f"Cannot retrieve JWKS: {exc}") from exc

        return self._jwks_cache[cache_key]

    def validate_jwt_token(
        self,
        access_token: str,
        user_pool_id: str,
        client_id: str,
        region: Optional[str] = None,
    ) -> dict[str, Any]:
        if not region:
            region = self.default_region

        try:
            unverified_header = jwt.get_unverified_header(access_token)
            kid = unverified_header.get("kid")

            if not kid:
                raise ValueError("Token missing 'kid' in header")

            jwks = self._get_jwks(user_pool_id, region)
            signing_key = None

            for key in jwks.get("keys", []):
                if key.get("kid") != kid:
                    continue

                try:
                    from jwt.algorithms import (
                        RSAAlgorithm,
                    )

                    signing_key = RSAAlgorithm.from_jwk(key)
                except (ImportError, AttributeError):
                    try:
                        from jwt.algorithms import (
                            get_default_algorithms,
                        )

                        algorithms = get_default_algorithms()
                        signing_key = algorithms["RS256"].from_jwk(key)
                    except (ImportError, AttributeError):
                        signing_key = PyJWK.from_jwk(json.dumps(key)).key
                break

            if not signing_key:
                raise ValueError(f"No matching key found for kid: {kid}")

            issuer = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}"

            claims = jwt.decode(
                access_token,
                signing_key,
                algorithms=["RS256"],
                issuer=issuer,
                options={
                    "verify_aud": False,
                    "verify_exp": True,
                    "verify_iat": True,
                },
            )

            token_use = claims.get("token_use")
            if token_use not in ["access", "id"]:
                raise ValueError(f"Invalid token_use: {token_use}")

            token_client_id = claims.get("client_id")
            if token_client_id and token_client_id != client_id:
                logger.warning(
                    "Token issued for different client: %s vs expected %s",
                    token_client_id,
                    client_id,
                )

            logger.info("Successfully validated JWT token for client/user")
            return claims

        except jwt.ExpiredSignatureError as exc:
            raise ValueError("Token has expired") from exc
        except jwt.InvalidTokenError as exc:
            raise ValueError(f"Invalid token: {exc}") from exc
        except Exception as exc:
            logger.error("JWT validation error: %s", exc)
            raise ValueError(f"Token validation failed: {exc}") from exc

    def validate_with_boto3(
        self,
        access_token: str,
        region: Optional[str] = None,
    ) -> dict[str, Any]:
        if not region:
            region = self.default_region

        try:
            cognito_client = self._get_cognito_client(region)
            response = cognito_client.get_user(AccessToken=access_token)

            user_attributes: dict[str, str] = {}
            for attr in response.get("UserAttributes", []):
                user_attributes[attr["Name"]] = attr["Value"]

            result = {
                "username": response.get("Username"),
                "user_attributes": user_attributes,
                "user_status": response.get("UserStatus"),
                "token_use": "access",
                "auth_method": "boto3",
            }

            logger.info(
                "Successfully validated token via boto3 for user %s",
                _hash_username(result["username"] or ""),
            )
            return result

        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            error_message = exc.response["Error"]["Message"]

            if error_code == "NotAuthorizedException":
                logger.warning("Cognito error %s: %s", error_code, error_message)
                raise ValueError("Invalid or expired access token") from exc

            if error_code == "UserNotFoundException":
                logger.warning("Cognito error %s: %s", error_code, error_message)
                raise ValueError("User not found") from exc

            logger.error("Cognito error %s: %s", error_code, error_message)
            raise ValueError(f"Token validation failed: {error_message}") from exc

        except Exception as exc:
            logger.error("Boto3 validation error: %s", exc)
            raise ValueError(f"Token validation failed: {exc}") from exc

    def validate_self_signed_token(
        self,
        access_token: str,
    ) -> dict[str, Any]:
        if not self.secret_key or not self.jwt_issuer or not self.jwt_audience:
            raise ValueError("Self-signed token validation is not configured")

        try:
            claims = jwt.decode(
                access_token,
                self.secret_key,
                algorithms=["HS256"],
                issuer=self.jwt_issuer,
                audience=self.jwt_audience,
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": True,
                },
                leeway=30,
            )

            token_use = claims.get("token_use")
            if token_use != "access":
                raise ValueError(f"Invalid token_use: {token_use}")

            scope_string = claims.get("scope", "")
            scopes = scope_string.split() if scope_string else []

            logger.info(
                "Successfully validated self-signed token for user: %s",
                claims.get("sub"),
            )

            return {
                "valid": True,
                "method": "self_signed",
                "data": claims,
                "client_id": claims.get("client_id", "user-generated"),
                "username": claims.get("sub", ""),
                "expires_at": claims.get("exp"),
                "scopes": scopes,
                "groups": [],
                "token_type": "user_generated",
            }

        except jwt.ExpiredSignatureError as exc:
            raise ValueError("Self-signed token has expired") from exc
        except jwt.InvalidTokenError as exc:
            raise ValueError(f"Invalid self-signed token: {exc}") from exc
        except Exception as exc:
            logger.error("Self-signed token validation error: %s", exc)
            raise ValueError(f"Self-signed token validation failed: {exc}") from exc

    def validate_token(
        self,
        access_token: str,
        user_pool_id: str,
        client_id: str,
        region: Optional[str] = None,
    ) -> dict[str, Any]:
        if not region:
            region = self.default_region

        if self.jwt_issuer and self.secret_key and self.jwt_audience:
            try:
                unverified_claims = jwt.decode(access_token, options={"verify_signature": False})
                if unverified_claims.get("iss") == self.jwt_issuer:
                    logger.debug("Token appears to be self-signed, validating...")
                    return self.validate_self_signed_token(access_token)
            except Exception:
                pass

        try:
            jwt_claims = self.validate_jwt_token(access_token, user_pool_id, client_id, region)

            scopes: list[str] = []
            if "scope" in jwt_claims:
                scopes = jwt_claims["scope"].split() if jwt_claims["scope"] else []

            return {
                "valid": True,
                "method": "jwt",
                "data": jwt_claims,
                "client_id": jwt_claims.get("client_id") or "",
                "username": jwt_claims.get("cognito:username") or jwt_claims.get("username") or "",
                "expires_at": jwt_claims.get("exp"),
                "scopes": scopes,
                "groups": jwt_claims.get("cognito:groups", []),
            }

        except ValueError as jwt_error:
            logger.debug("JWT validation failed: %s, trying boto3", jwt_error)
            try:
                boto3_data = self.validate_with_boto3(access_token, region)
                return {
                    "valid": True,
                    "method": "boto3",
                    "data": boto3_data,
                    "client_id": "",
                    "username": boto3_data.get("username") or "",
                    "user_attributes": boto3_data.get("user_attributes", {}),
                    "scopes": [],
                    "groups": [],
                }
            except ValueError as boto3_error:
                logger.debug("Boto3 validation failed: %s", boto3_error)
                raise ValueError(
                    "All validation methods failed. "
                    f"JWT: {jwt_error}, Boto3: {boto3_error}",
                ) from boto3_error

