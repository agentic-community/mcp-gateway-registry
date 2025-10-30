# Implementation Plan - Test Suite Modernization

## Overview

This implementation plan breaks down the test suite modernization into discrete, actionable coding tasks. Each task builds incrementally on previous tasks and references specific requirements from the requirements document.

---

## Phase 1: Test Infrastructure Setup

- [ ] 1. Set up pytest configuration and markers
  - Create or update `pytest.ini` with test markers (unit, integration, e2e)
  - Configure coverage settings and thresholds
  - Set test discovery patterns and paths
  - _Requirements: 1.5, 12.5_

- [ ] 1.1 Update conftest.py with authentication fixtures
  - Implement `mock_keycloak_user_context` fixture with groups and scopes
  - Implement `mock_admin_user_context` fixture with full permissions
  - Implement `mock_m2m_token` fixture for JWT generation
  - Implement `mock_enhanced_auth` fixture for dependency injection
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 1.2 Create auth_fixtures.py module
  - Move authentication fixtures to dedicated module
  - Add OAuth2 provider mocking utilities
  - Add session cookie creation helpers
  - Add JWT token validation helpers
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 1.3 Update factories.py with new data generators
  - Implement `ServerMetadataFactory` with realistic tool schemas
  - Implement `UserContextFactory` with groups and scopes
  - Implement `JWTTokenFactory` for M2M tokens
  - Add helper functions for generating test server configurations
  - _Requirements: 13.1, 13.2, 13.3, 13.4_

- [ ] 1.4 Create FAISS service mock implementation
  - Implement `MockFAISSService` class with in-memory storage
  - Add methods for add_or_update_service, search, and remove_service
  - Create fixture for injecting mock FAISS service
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

---

## Phase 2: Authentication Tests

- [ ] 2. Update authentication dependency tests
  - Update `tests/unit/auth/test_auth_dependencies.py` to use new fixtures
  - Replace old `mock_authenticated_user` with `mock_enhanced_auth`
  - _Requirements: 2.3_

- [ ] 2.1 Add Keycloak OAuth flow tests
  - Test OAuth2 login redirect with provider parameter
  - Test OAuth2 callback with session creation
  - Test OAuth2 callback error handling
  - _Requirements: 2.1_

- [ ] 2.2 Add M2M JWT validation tests
  - Test JWT token parsing and validation
  - Test scope extraction from JWT claims
  - Test expired token handling
  - Test invalid signature handling
  - _Requirements: 2.2_

- [ ] 2.3 Add enhanced_auth dependency tests
  - Test user context creation from session
  - Test user context creation from JWT header
  - Test fallback behavior when both auth methods present
  - _Requirements: 2.3_

- [ ] 2.4 Add permission extraction tests
  - Test group-to-scope mapping
  - Test accessible_servers calculation
  - Test UI permissions generation
  - _Requirements: 2.4_

- [ ] 2.5 Add session validation tests
  - Test valid session cookie validation
  - Test expired session handling
  - Test invalid session signature handling
  - _Requirements: 2.6, 2.7_

---

## Phase 3: Server Service Tests

- [ ] 3. Update server service tests
  - Update `tests/unit/servers/test_server_service.py` with new fixtures
  - Use realistic server data from factories
  - _Requirements: 3.5_

- [ ] 3.1 Add get_filtered_servers tests
  - Test admin user sees all servers
  - Test regular user sees only accessible servers
  - Test filtering by permission scopes
  - _Requirements: 3.1_

- [ ] 3.2 Add get_all_servers_with_permissions tests
  - Test permission annotation for admin users
  - Test permission annotation for regular users
  - Test empty server list handling
  - _Requirements: 3.2_

- [ ] 3.3 Add user_can_access_server_path tests
  - Test access granted for permitted servers
  - Test access denied for restricted servers
  - Test admin bypass of restrictions
  - _Requirements: 3.3_

- [ ] 3.4 Add remove_server tests
  - Test server removal from registry
  - Test scopes cleanup after removal
  - Test FAISS index cleanup after removal
  - _Requirements: 3.4_

---

## Phase 4: Scopes Manager Tests

- [ ] 4. Create scopes manager test file
  - Create `tests/unit/utils/test_scopes_manager.py`
  - Set up fixtures for scopes file mocking
  - _Requirements: 4.1-4.8_

- [ ] 4.1 Test add_server_to_scopes_unrestricted_only
  - Test adding server to unrestricted group
  - Test YAML file update
  - Verify scopes structure
  - _Requirements: 4.1_

- [ ] 4.2 Test add_server_to_groups_custom
  - Test adding server to custom groups
  - Test multiple group assignments
  - Verify group permissions
  - _Requirements: 4.2_

- [ ] 4.3 Test remove_server_from_scopes
  - Test complete server removal from scopes
  - Test YAML file cleanup
  - Verify no orphaned entries
  - _Requirements: 4.3_

- [ ] 4.4 Test remove_server_from_groups
  - Test selective group removal
  - Test partial scope cleanup
  - Verify remaining groups intact
  - _Requirements: 4.4_

- [ ] 4.5 Test update_server_scopes
  - Test scope permission updates
  - Test read/execute permission changes
  - Verify atomic updates
  - _Requirements: 4.5_

- [ ] 4.6 Test trigger_auth_server_reload
  - Test reload signal sent to auth server
  - Test error handling on reload failure
  - _Requirements: 4.6_

- [ ] 4.7 Test read_scopes_file
  - Test YAML parsing
  - Test malformed file handling
  - Test missing file handling
  - _Requirements: 4.7_

- [ ] 4.8 Test write_scopes_file
  - Test atomic file write operations
  - Test backup creation
  - Test write failure handling
  - _Requirements: 4.8_

---

## Phase 5: Internal API Tests

- [ ] 5. Create internal API test file
  - Create `tests/integration/test_internal_api.py`
  - Set up test client with internal endpoints
  - _Requirements: 5.1-5.8_

- [ ] 5.1 Test internal register with authentication
  - Test registration with valid auth headers
  - Test server added to registry
  - Test scopes updated
  - _Requirements: 5.1_

- [ ] 5.2 Test internal register without authentication
  - Test local access without auth
  - Test localhost bypass
  - Test 127.0.0.1 bypass
  - _Requirements: 5.2_

- [ ] 5.3 Test internal remove endpoint
  - Test server removal with auth
  - Test cleanup verification
  - Test error on non-existent server
  - _Requirements: 5.3_

- [ ] 5.4 Test internal toggle endpoint
  - Test enable server operation
  - Test disable server operation
  - Test state persistence
  - _Requirements: 5.4_

- [ ] 5.5 Test internal healthcheck endpoint
  - Test health status reporting
  - Test multiple server status
  - Test error aggregation
  - _Requirements: 5.5_

- [ ] 5.6 Test internal add_to_groups endpoint
  - Test group assignment
  - Test scope updates
  - Test permission propagation
  - _Requirements: 5.6_

- [ ] 5.7 Test internal remove_from_groups endpoint
  - Test group removal
  - Test scope cleanup
  - Test permission revocation
  - _Requirements: 5.7_

- [ ] 5.8 Test internal list_services endpoint
  - Test service enumeration
  - Test filtering by status
  - Test pagination
  - _Requirements: 5.8_

---

## Phase 6: CLI Tools Tests

- [ ] 6. Create CLI tools test file
  - Create `tests/unit/cli/test_mcp_client.py`
  - Set up CLI command mocking
  - _Requirements: 6.1-6.6_

- [ ] 6.1 Test ping command
  - Test successful server ping
  - Test connection failure handling
  - Test timeout handling
  - _Requirements: 6.1_

- [ ] 6.2 Test list command
  - Test service discovery
  - Test filtering options
  - Test output formatting
  - _Requirements: 6.2_

- [ ] 6.3 Test call command
  - Test tool invocation
  - Test argument passing
  - Test response handling
  - _Requirements: 6.3_

- [ ] 6.4 Test M2M authentication
  - Test JWT token generation
  - Test token refresh
  - Test authentication headers
  - _Requirements: 6.4_

- [ ] 6.5 Test error handling
  - Test network errors
  - Test invalid responses
  - Test authentication failures
  - _Requirements: 6.5_

- [ ] 6.6 Test JSON output format
  - Test structured output
  - Test programmatic consumption
  - Test error JSON format
  - _Requirements: 6.6_

---

## Phase 7: MCP Gateway Server Tests

- [ ] 7. Create MCP Gateway server test file
  - Create `tests/unit/servers/test_mcpgw_server.py`
  - Set up gateway tool mocking
  - _Requirements: 7.1-7.8_

- [ ] 7.1 Test list_services_tool
  - Test service enumeration
  - Test metadata inclusion
  - Test filtering capabilities
  - _Requirements: 7.1_

- [ ] 7.2 Test healthcheck_services_tool
  - Test status aggregation
  - Test health metrics
  - Test error reporting
  - _Requirements: 7.2_

- [ ] 7.3 Test register_service_tool
  - Test new service registration
  - Test validation
  - Test duplicate handling
  - _Requirements: 7.3_

- [ ] 7.4 Test remove_service_tool
  - Test service deletion
  - Test cleanup verification
  - Test error on missing service
  - _Requirements: 7.4_

- [ ] 7.5 Test toggle_service_tool
  - Test enable operation
  - Test disable operation
  - Test state verification
  - _Requirements: 7.5_

- [ ] 7.6 Test add_server_to_scopes_groups_tool
  - Test permission assignment
  - Test group management
  - Test scope updates
  - _Requirements: 7.6_

- [ ] 7.7 Test remove_server_from_scopes_groups_tool
  - Test permission revocation
  - Test group removal
  - Test scope cleanup
  - _Requirements: 7.7_

- [ ] 7.8 Test intelligent_tool_finder
  - Test semantic search
  - Test relevance ranking
  - Test query understanding
  - _Requirements: 7.8_

---

## Phase 8: Integration Tests

- [ ] 8. Create server lifecycle integration tests
  - Create `tests/integration/test_server_lifecycle.py`
  - Set up end-to-end test fixtures
  - _Requirements: 8.1, 8.2, 8.3_

- [ ] 8.1 Test complete server registration flow
  - Register new server via API
  - Verify server in registry list
  - Verify scopes.yml updated
  - Verify FAISS index updated
  - Verify health check initiated
  - _Requirements: 8.1_

- [ ] 8.2 Test complete server deletion flow
  - Delete server via API
  - Verify removed from registry list
  - Verify removed from scopes.yml
  - Verify removed from FAISS index
  - Verify cleanup complete
  - _Requirements: 8.2_

- [ ] 8.3 Test server toggle with health check
  - Toggle server on
  - Verify immediate health check
  - Verify FAISS metadata updated
  - Toggle server off
  - Verify status changed
  - _Requirements: 8.3_

- [ ] 8.4 Create authentication flow integration tests
  - Create `tests/integration/test_authentication_flow.py`
  - Set up OAuth and JWT mocking
  - _Requirements: 8.4_

- [ ] 8.5 Test OAuth2 login flow
  - Test provider redirect
  - Test callback handling
  - Test session creation
  - Test permission loading
  - _Requirements: 8.4_

- [ ] 8.6 Test M2M token authentication
  - Test client credentials flow
  - Test JWT generation
  - Test API access with token
  - _Requirements: 8.4_

- [ ] 8.7 Test permission-based access control
  - Test admin access
  - Test user access restrictions
  - Test scope-based filtering
  - _Requirements: 8.4_

- [ ] 8.8 Test health monitoring cycle
  - Test periodic health checks
  - Test WebSocket updates
  - Test status persistence
  - _Requirements: 8.5_

---

## Phase 9: E2E Tests

- [ ] 9. Create complete workflow E2E tests
  - Create `tests/e2e/test_complete_workflows.py`
  - Set up full application stack for E2E
  - _Requirements: 9.1-9.6_

- [ ] 9.1 Test new server registration to first call
  - Admin registers new server
  - Server appears in registry UI
  - Health check completes successfully
  - User discovers server via search
  - User calls server tool
  - _Requirements: 9.2, 9.3, 9.4_

- [ ] 9.2 Test user permission restrictions
  - Restricted user logs in
  - Can only see permitted servers
  - Cannot toggle/modify restricted servers
  - Can toggle/modify permitted servers
  - _Requirements: 9.5, 9.6_

---

## Phase 10: Test Documentation

- [ ] 10. Create test documentation
  - Create `tests/README.md` with overview and quick start
  - Document test execution commands
  - Document coverage targets
  - _Requirements: 14.1, 14.5_

- [ ] 10.1 Create CONTRIBUTING.md
  - Document test writing guidelines
  - Document AAA pattern usage
  - Document fixture usage patterns
  - _Requirements: 14.2_

- [ ] 10.2 Create FIXTURES.md
  - List all available fixtures
  - Provide usage examples for each
  - Document fixture dependencies
  - _Requirements: 14.2, 14.3_

- [ ] 10.3 Create TROUBLESHOOTING.md
  - Document common test failures
  - Provide solutions for each
  - Document environment setup issues
  - _Requirements: 14.4, 14.6_

---

## Phase 11: CI/CD Integration

- [ ] 11. Set up GitHub Actions workflow
  - Create `.github/workflows/test.yml`
  - Configure test execution
  - Configure coverage reporting
  - _Requirements: 15.1, 15.2_

- [ ] 11.1 Configure coverage enforcement
  - Set 80% overall coverage threshold
  - Set 95% critical path threshold
  - Set 90% new feature threshold
  - Fail builds on threshold violations
  - _Requirements: 15.3_

- [ ] 11.2 Configure test environment
  - Set up environment variables
  - Configure external service mocking
  - Set up test database
  - _Requirements: 15.4_

---

## Phase 12: Cleanup and Validation

- [ ] 12. Remove deprecated test code
  - Remove old authentication mocks
  - Remove outdated fixtures
  - Update all test imports
  - _Requirements: 12.5_

- [ ] 12.1 Run full test suite validation
  - Execute all tests
  - Verify 100% pass rate
  - Verify coverage thresholds met
  - Verify no warnings
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.6_

- [ ] 12.2 Performance optimization
  - Optimize slow tests
  - Enable parallel execution
  - Verify 60s total runtime
  - _Requirements: 12.6_

- [ ] 12.3 Final coverage report
  - Generate HTML coverage report
  - Generate XML coverage report
  - Verify all requirements met
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_
