"""
Property-based test for skill security scan repository create-then-retrieve round-trip.

# Feature: skill-scanner-integration, Property 5: Repository create-then-retrieve round-trip

**Validates: Requirements 6.2**

Note: File-backend tests were removed when the file storage backend was
dropped in v1.24.8. DocumentDB-based repository tests live in
tests/integration/test_documentdb_*.py.
"""
