"""
Test constants for MCP Gateway Registry tests.

This module defines constants used across test modules to ensure consistency.
"""

# Test Server Names
TEST_SERVER_NAME_1: str = "com.example.test-server-1"
TEST_SERVER_NAME_2: str = "com.example.test-server-2"
TEST_SERVER_NAME_AUTH: str = "com.example.auth-server"
TEST_SERVER_NAME_TIME: str = "com.example.currenttime"

# Test URLs
TEST_SERVER_URL_1: str = "http://localhost:8080/test-server-1"
TEST_SERVER_URL_2: str = "http://localhost:8080/test-server-2"
TEST_SERVER_URL_AUTH: str = "http://localhost:8080/auth-server"

# Test Agent Names
TEST_AGENT_NAME_1: str = "test-agent-1"
TEST_AGENT_NAME_2: str = "test-agent-2"
TEST_AGENT_PATH_1: str = "/agents/test-agent-1"
TEST_AGENT_PATH_2: str = "/agents/test-agent-2"

# Test Agent URLs
TEST_AGENT_URL_1: str = "http://localhost:9000/agent-1"
TEST_AGENT_URL_2: str = "http://localhost:9000/agent-2"

# Test User Information
TEST_USERNAME: str = "testuser"
TEST_USER_EMAIL: str = "testuser@example.com"
TEST_ADMIN_USERNAME: str = "admin"
TEST_ADMIN_PASSWORD: str = "adminpass"

# Test Authentication
TEST_JWT_SECRET: str = "test-secret-key-for-jwt-tokens"
TEST_SESSION_COOKIE_NAME: str = "mcp_gateway_session"

# Test Groups and Scopes
TEST_USER_GROUPS: list[str] = ["users", "developers"]
TEST_ADMIN_GROUPS: list[str] = ["admins", "users"]
TEST_USER_SCOPES: list[str] = ["read:servers", "read:agents"]
TEST_ADMIN_SCOPES: list[str] = ["read:servers", "write:servers", "read:agents", "write:agents"]

# Test Tags
TEST_TAGS_DATA: list[str] = ["data", "analytics", "ml"]
TEST_TAGS_WEB: list[str] = ["web", "api", "rest"]
TEST_TAGS_AUTH: list[str] = ["auth", "security", "oauth"]

# Test Embeddings
TEST_EMBEDDING_DIM: int = 384
TEST_MODEL_NAME: str = "all-MiniLM-L6-v2"

# Test Search
TEST_SEARCH_QUERY: str = "data processing server"
TEST_SEARCH_LIMIT: int = 10

# Test Tool Information
TEST_TOOL_NAME_1: str = "get_data"
TEST_TOOL_NAME_2: str = "process_data"
TEST_TOOL_DESCRIPTION_1: str = "Retrieve data from source"
TEST_TOOL_DESCRIPTION_2: str = "Process and transform data"

# Test Skill Information
TEST_SKILL_ID_1: str = "data-retrieval"
TEST_SKILL_ID_2: str = "data-processing"
TEST_SKILL_NAME_1: str = "Data Retrieval"
TEST_SKILL_NAME_2: str = "Data Processing"

# Test Repository
TEST_REPO_URL: str = "https://github.com/example/test-server"
TEST_REPO_SOURCE: str = "github"

# Test Package
TEST_PACKAGE_IDENTIFIER: str = "@example/test-server"
TEST_PACKAGE_VERSION: str = "1.0.0"
TEST_PACKAGE_REGISTRY_TYPE: str = "npm"

# Test Pagination
DEFAULT_PAGE_SIZE: int = 20
TEST_CURSOR: str = "test-cursor-value"

# Test Timeouts
TEST_TIMEOUT_SHORT: int = 1
TEST_TIMEOUT_MEDIUM: int = 5
TEST_TIMEOUT_LONG: int = 30

# Test Ratings
TEST_RATING_LOW: float = 2.5
TEST_RATING_MEDIUM: float = 3.5
TEST_RATING_HIGH: float = 4.5
TEST_RATING_MAX: float = 5.0

# Test Visibility
VISIBILITY_PUBLIC: str = "public"
VISIBILITY_PRIVATE: str = "private"
VISIBILITY_GROUP: str = "group-restricted"

# Test Trust Levels
TRUST_UNVERIFIED: str = "unverified"
TRUST_COMMUNITY: str = "community"
TRUST_VERIFIED: str = "verified"
TRUST_TRUSTED: str = "trusted"

# Test Protocol Versions
PROTOCOL_VERSION_1_0: str = "1.0"
PROTOCOL_VERSION_2024_11_05: str = "2024-11-05"

# Test Transport Types
TRANSPORT_STDIO: str = "stdio"
TRANSPORT_HTTP: str = "streamable-http"
TRANSPORT_SSE: str = "sse"

# Test Security Schemes
SECURITY_TYPE_BEARER: str = "http"
SECURITY_TYPE_OAUTH2: str = "oauth2"
SECURITY_TYPE_API_KEY: str = "apiKey"
SECURITY_SCHEME_BEARER: str = "bearer"

# Test Capabilities
DEFAULT_CAPABILITIES: dict[str, bool] = {"streaming": False, "tools": True, "prompts": False}

# Test MIME Types
MIME_TEXT_PLAIN: str = "text/plain"
MIME_APPLICATION_JSON: str = "application/json"
MIME_TEXT_HTML: str = "text/html"

# Test Status Codes
HTTP_OK: int = 200
HTTP_CREATED: int = 201
HTTP_NO_CONTENT: int = 204
HTTP_BAD_REQUEST: int = 400
HTTP_UNAUTHORIZED: int = 401
HTTP_FORBIDDEN: int = 403
HTTP_NOT_FOUND: int = 404
HTTP_CONFLICT: int = 409
HTTP_INTERNAL_ERROR: int = 500
