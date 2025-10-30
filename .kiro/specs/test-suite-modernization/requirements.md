# Requirements Document - Test Suite Modernization

## Introduction

The existing test suite in the tests/ folder needs to be completely rewritten to align with the current architecture, new authentication system, and recent feature additions. Most tests are outdated, use old authentication mocks, and don't cover new functionality. This modernization effort will bring the test suite to 80% code coverage, ensure all tests pass, and provide comprehensive coverage for critical paths including authentication, server registration, and health monitoring.

## Glossary

- **Test Suite**: The collection of all automated tests in the tests/ directory
- **Authentication System**: The Keycloak/OAuth2-based authentication mechanism with M2M (machine-to-machine) support
- **FAISS Service**: The vector search service used for server discovery and indexing
- **MCP Server**: Model Context Protocol server that provides tools and capabilities
- **Coverage**: The percentage of code lines executed during test runs
- **Fixture**: Reusable test setup code provided by pytest
- **Mock**: A test double that simulates external dependencies
- **Integration Test**: Tests that verify multiple components working together
- **Unit Test**: Tests that verify individual components in isolation
- **E2E Test**: End-to-end tests that verify complete user workflows

## Requirements

### Requirement 1: Test Infrastructure Modernization

**User Story:** As a developer, I want modern test fixtures and infrastructure, so that I can write tests that accurately reflect the current authentication system and architecture.

#### Acceptance Criteria

1. WHEN THE Test System initializes fixtures, THE Test System SHALL provide mock Keycloak user context with username, groups, scopes, and permissions
2. WHEN THE Test System initializes fixtures, THE Test System SHALL provide mock admin user context with full access permissions
3. WHEN THE Test System initializes fixtures, THE Test System SHALL provide mock M2M JWT tokens for agent authentication
4. WHEN THE Test System initializes fixtures, THE Test System SHALL provide mock enhanced_auth dependency that returns user context
5. WHERE pytest markers are configured, THE Test System SHALL support unit, integration, and e2e test categorization

### Requirement 2: Authentication Test Coverage

**User Story:** As a developer, I want comprehensive authentication tests, so that I can ensure the Keycloak/OAuth2 system works correctly and securely.

#### Acceptance Criteria

1. WHEN THE Test System validates authentication, THE Test System SHALL test Keycloak OAuth2 login flow with provider redirect
2. WHEN THE Test System validates authentication, THE Test System SHALL test M2M JWT token validation and scope extraction
3. WHEN THE Test System validates authentication, THE Test System SHALL test enhanced_auth dependency with user context creation
4. WHEN THE Test System validates authentication, THE Test System SHALL test permission extraction from groups and scopes
5. WHEN THE Test System validates authentication, THE Test System SHALL test UI permission checking for service operations
6. WHEN THE Test System validates authentication, THE Test System SHALL test session cookie validation and expiration handling
7. WHEN THE Test System validates authentication, THE Test System SHALL test invalid session and expired token scenarios

### Requirement 3: Server Service Test Coverage

**User Story:** As a developer, I want complete server service tests, so that I can ensure server registration, management, and permission filtering work correctly.

#### Acceptance Criteria

1. WHEN THE Test System validates server operations, THE Test System SHALL test get_filtered_servers with user permissions
2. WHEN THE Test System validates server operations, THE Test System SHALL test get_all_servers_with_permissions for admin and regular users
3. WHEN THE Test System validates server operations, THE Test System SHALL test user_can_access_server_path with various permission scenarios
4. WHEN THE Test System validates server operations, THE Test System SHALL test remove_server with cleanup of scopes and FAISS index
5. WHEN THE Test System validates server operations, THE Test System SHALL test server registration with realistic metadata and tool schemas

### Requirement 4: Scopes Manager Test Coverage

**User Story:** As a developer, I want comprehensive scopes manager tests, so that I can ensure server group assignments and scope management work correctly.

#### Acceptance Criteria

1. WHEN THE Test System validates scopes operations, THE Test System SHALL test add_server_to_scopes with unrestricted-only configuration
2. WHEN THE Test System validates scopes operations, THE Test System SHALL test add_server_to_groups with custom group assignments
3. WHEN THE Test System validates scopes operations, THE Test System SHALL test remove_server_from_scopes with complete cleanup
4. WHEN THE Test System validates scopes operations, THE Test System SHALL test remove_server_from_groups with selective removal
5. WHEN THE Test System validates scopes operations, THE Test System SHALL test update_server_scopes with permission changes
6. WHEN THE Test System validates scopes operations, THE Test System SHALL test trigger_auth_server_reload after scope modifications
7. WHEN THE Test System validates scopes operations, THE Test System SHALL test read_scopes_file with YAML parsing
8. WHEN THE Test System validates scopes operations, THE Test System SHALL test write_scopes_file with atomic file operations

### Requirement 5: Internal API Test Coverage

**User Story:** As a developer, I want comprehensive internal API tests, so that I can ensure service management endpoints work correctly with and without authentication.

#### Acceptance Criteria

1. WHEN THE Test System validates internal APIs, THE Test System SHALL test internal register endpoint with authentication headers
2. WHEN THE Test System validates internal APIs, THE Test System SHALL test internal register endpoint without authentication for local access
3. WHEN THE Test System validates internal APIs, THE Test System SHALL test internal remove endpoint with service cleanup
4. WHEN THE Test System validates internal APIs, THE Test System SHALL test internal toggle endpoint with state changes
5. WHEN THE Test System validates internal APIs, THE Test System SHALL test internal healthcheck endpoint with status reporting
6. WHEN THE Test System validates internal APIs, THE Test System SHALL test internal add_to_groups endpoint with scope updates
7. WHEN THE Test System validates internal APIs, THE Test System SHALL test internal remove_from_groups endpoint with permission removal
8. WHEN THE Test System validates internal APIs, THE Test System SHALL test internal list_services endpoint with filtered results

### Requirement 6: CLI Tools Test Coverage

**User Story:** As a developer, I want comprehensive CLI tools tests, so that I can ensure command-line operations work correctly for users and agents.

#### Acceptance Criteria

1. WHEN THE Test System validates CLI operations, THE Test System SHALL test ping command with server connectivity verification
2. WHEN THE Test System validates CLI operations, THE Test System SHALL test list command with service discovery and filtering
3. WHEN THE Test System validates CLI operations, THE Test System SHALL test call command with tool invocation and response handling
4. WHEN THE Test System validates CLI operations, THE Test System SHALL test M2M authentication with JWT token generation
5. WHEN THE Test System validates CLI operations, THE Test System SHALL test error handling with network failures and invalid responses
6. WHEN THE Test System validates CLI operations, THE Test System SHALL test JSON output format for programmatic consumption

### Requirement 7: MCP Gateway Server Test Coverage

**User Story:** As a developer, I want comprehensive MCP Gateway server tests, so that I can ensure the gateway tools work correctly for service management.

#### Acceptance Criteria

1. WHEN THE Test System validates gateway tools, THE Test System SHALL test list_services_tool with service enumeration
2. WHEN THE Test System validates gateway tools, THE Test System SHALL test healthcheck_services_tool with status aggregation
3. WHEN THE Test System validates gateway tools, THE Test System SHALL test register_service_tool with new service addition
4. WHEN THE Test System validates gateway tools, THE Test System SHALL test remove_service_tool with service deletion
5. WHEN THE Test System validates gateway tools, THE Test System SHALL test toggle_service_tool with enable and disable operations
6. WHEN THE Test System validates gateway tools, THE Test System SHALL test add_server_to_scopes_groups_tool with permission assignment
7. WHEN THE Test System validates gateway tools, THE Test System SHALL test remove_server_from_scopes_groups_tool with permission revocation
8. WHEN THE Test System validates gateway tools, THE Test System SHALL test intelligent_tool_finder with semantic search capabilities

### Requirement 8: Integration Test Coverage

**User Story:** As a developer, I want integration tests for complete workflows, so that I can ensure end-to-end functionality works correctly across components.

#### Acceptance Criteria

1. WHEN THE Test System validates workflows, THE Test System SHALL test complete server registration flow from registration through FAISS indexing
2. WHEN THE Test System validates workflows, THE Test System SHALL test complete server deletion flow with cleanup verification
3. WHEN THE Test System validates workflows, THE Test System SHALL test server toggle with immediate health check execution
4. WHEN THE Test System validates workflows, THE Test System SHALL test authentication flow from OAuth2 login through session creation
5. WHEN THE Test System validates workflows, THE Test System SHALL test health monitoring cycle with WebSocket updates

### Requirement 9: End-to-End Test Coverage

**User Story:** As a developer, I want end-to-end tests for complete user scenarios, so that I can ensure the entire system works correctly from a user perspective.

#### Acceptance Criteria

1. WHEN THE Test System validates user scenarios, THE Test System SHALL test new server registration to first tool call workflow
2. WHEN THE Test System validates user scenarios, THE Test System SHALL test admin registering server with UI appearance and health check completion
3. WHEN THE Test System validates user scenarios, THE Test System SHALL test user discovering server via search and calling tools
4. WHEN THE Test System validates user scenarios, THE Test System SHALL test restricted user login with limited server visibility
5. WHEN THE Test System validates user scenarios, THE Test System SHALL test user permission restrictions preventing unauthorized operations
6. WHEN THE Test System validates user scenarios, THE Test System SHALL test user toggling and modifying only permitted servers

### Requirement 10: FAISS Service Test Stability

**User Story:** As a developer, I want stable FAISS service tests, so that tests run without segmentation faults or crashes.

#### Acceptance Criteria

1. WHEN THE Test System tests FAISS operations, THE Test System SHALL use proper mocking strategy that prevents segmentation faults
2. WHEN THE Test System tests FAISS operations, THE Test System SHALL test service initialization with index creation
3. WHEN THE Test System tests FAISS operations, THE Test System SHALL test add_or_update_service with vector embedding
4. WHEN THE Test System tests FAISS operations, THE Test System SHALL test search operations with query vectors
5. WHEN THE Test System tests FAISS operations, THE Test System SHALL test index persistence and loading

### Requirement 11: Code Coverage Achievement

**User Story:** As a developer, I want 80% code coverage, so that I can ensure the codebase is well-tested and maintainable.

#### Acceptance Criteria

1. WHEN THE Test System measures coverage, THE Test System SHALL achieve minimum 80 percent overall code coverage
2. WHEN THE Test System measures coverage, THE Test System SHALL achieve minimum 95 percent coverage for authentication modules
3. WHEN THE Test System measures coverage, THE Test System SHALL achieve minimum 95 percent coverage for server registration modules
4. WHEN THE Test System measures coverage, THE Test System SHALL achieve minimum 95 percent coverage for health check modules
5. WHEN THE Test System measures coverage, THE Test System SHALL achieve minimum 90 percent coverage for new features

### Requirement 12: Test Execution Standards

**User Story:** As a developer, I want all tests to pass and run cleanly, so that I can trust the test suite and use it for continuous integration.

#### Acceptance Criteria

1. WHEN THE Test System executes tests, THE Test System SHALL complete all tests without segmentation faults or crashes
2. WHEN THE Test System executes tests, THE Test System SHALL pass all unit tests with proper mocking
3. WHEN THE Test System executes tests, THE Test System SHALL pass all integration tests with component interaction
4. WHEN THE Test System executes tests, THE Test System SHALL pass all e2e tests with complete workflows
5. WHEN THE Test System executes tests, THE Test System SHALL run without pytest warnings for deprecated fixtures or mocks
6. WHEN THE Test System executes tests, THE Test System SHALL complete in under 60 seconds for fast feedback

### Requirement 13: Test Data Quality

**User Story:** As a developer, I want realistic test data, so that tests accurately simulate production scenarios.

#### Acceptance Criteria

1. WHEN THE Test System generates test data, THE Test System SHALL use factories for server metadata with realistic tool schemas
2. WHEN THE Test System generates test data, THE Test System SHALL provide sample MCP server configurations with valid JSON
3. WHEN THE Test System generates test data, THE Test System SHALL create user contexts with realistic group and scope assignments
4. WHEN THE Test System generates test data, THE Test System SHALL generate JWT tokens with proper claims and expiration
5. WHEN THE Test System generates test data, THE Test System SHALL maintain test data fixtures in version control

### Requirement 14: Test Documentation

**User Story:** As a developer, I want comprehensive test documentation, so that I can understand how to run tests and what they cover.

#### Acceptance Criteria

1. WHEN THE Test System provides documentation, THE Test System SHALL document test execution commands in README files
2. WHEN THE Test System provides documentation, THE Test System SHALL document test fixture usage and available mocks
3. WHEN THE Test System provides documentation, THE Test System SHALL document test data factory patterns and examples
4. WHEN THE Test System provides documentation, THE Test System SHALL document environment variable requirements for tests
5. WHEN THE Test System provides documentation, THE Test System SHALL document coverage targets and current status
6. WHEN THE Test System provides documentation, THE Test System SHALL document troubleshooting steps for common test failures

### Requirement 15: CI/CD Integration

**User Story:** As a developer, I want tests to run in CI/CD pipelines, so that code quality is automatically verified on every commit.

#### Acceptance Criteria

1. WHEN THE Test System runs in CI/CD, THE Test System SHALL execute all tests with environment variable configuration
2. WHEN THE Test System runs in CI/CD, THE Test System SHALL generate coverage reports in HTML and XML formats
3. WHEN THE Test System runs in CI/CD, THE Test System SHALL fail builds when coverage drops below 80 percent
4. WHEN THE Test System runs in CI/CD, THE Test System SHALL mock external dependencies including Keycloak and FAISS
5. WHEN THE Test System runs in CI/CD, THE Test System SHALL complete within GitHub Actions timeout limits
