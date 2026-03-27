"""Conftest for Lambda collector tests.

Sets required environment variables before the Lambda module is imported,
since it reads os.environ[] at module level.
"""

import os

os.environ.setdefault("RATE_LIMIT_TABLE", "test-rate-limit-table")
os.environ.setdefault("DOCUMENTDB_SECRET_ARN", "test-secret-arn")
os.environ.setdefault("DOCUMENTDB_ENDPOINT", "test-endpoint:27017")
