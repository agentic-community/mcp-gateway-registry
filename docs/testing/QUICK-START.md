# Test Suite Quick Start Guide

## TL;DR - Just Run Tests Safely

```bash
# Run all tests (memory-safe, serial execution)
uv run pytest

# Or use the test runner script
python scripts/test.py full
```

## Problem Solved

Previously, running tests would crash EC2 instances due to:
- Heavy ML model loading (sentence-transformers, vector search)
- Parallel execution spawning multiple model copies
- Memory multiplication across workers

**Now fixed!** All tests use mocked models by default.

## Quick Commands

### Safe for All EC2 Instances

```bash
# Run unit tests (fast)
python scripts/test.py unit

# Run specific domains
python scripts/test.py auth
python scripts/test.py servers

# Run fast tests (2 workers, still safe)
python scripts/test.py fast

# Full test suite (serial, safest)
python scripts/test.py full
```

### If You Have More Memory (16GB+ RAM)

```bash
# Run with 2 workers
python scripts/test.py full -n 2

# Run with 4 workers (requires 16GB+ RAM)
python scripts/test.py unit -n 4
```

## What Changed

### 1. Mocked Dependencies (Automatic)

All tests now automatically use mocked versions of:
- Vector search service
- Sentence-transformers embedding models
- PyTorch model loading

No changes needed to existing tests - it just works!

### 2. Serial Execution by Default

Tests run one at a time by default to prevent memory issues:

```bash
# Before (would crash)
pytest -n auto  # ❌ Crashes EC2

# Now (safe)
pytest          # ✅ Runs serially, no crash
```

### 3. Optional Parallelization

Use the `-n` flag to control workers:

```bash
# 2 workers (safe for most EC2)
python scripts/test.py unit -n 2

# 4 workers (needs 16GB+ RAM)
python scripts/test.py unit -n 4
```

## Memory Guidelines

| EC2 Instance | Safe Workers | Notes |
|--------------|--------------|-------|
| t3.small (2GB) | 1 (serial) | ✅ Now works! |
| t3.medium (4GB) | 1-2 | ✅ Now works! |
| t3.large (8GB) | 2 | ✅ Recommended |
| t3.xlarge (16GB+) | 2-4 | ✅ Can use more workers |

## Monitoring Memory

While tests run:

```bash
# Check current memory usage
free -h

# Watch memory in real-time
watch -n 1 free -h
```

## Writing New Tests

Tests automatically use mocked models - no special setup needed:

```python
import pytest

@pytest.mark.unit
def test_my_feature(server_service):
    # Search service and embeddings are automatically mocked
    result = server_service.do_something()
    assert result is not None
```

## When Tests Fail

```bash
# Run specific failing test
pytest tests/unit/auth/test_auth_routes.py::test_login -v

# Show debug output
pytest tests/unit/auth/ --log-cli-level=DEBUG

# Stop on first failure
pytest -x
```

## Getting Coverage

```bash
# Generate coverage report
python scripts/test.py coverage

# View in browser
open htmlcov/index.html
```

## More Information

- **[Memory Management Details](./memory-management.md)** - In-depth explanation
- **[Test Categories](./test-categories.md)** - How tests are organized
- **[Main Testing README](./README.md)** - Complete reference

## Still Having Issues?

If tests still crash:

1. **Check you're on the latest version:**
   ```bash
   git pull
   uv sync --extra dev
   ```

2. **Verify mocking is enabled:**
   ```bash
   pytest tests/unit/core/test_config.py -v
   ```
   Should pass quickly (< 1 second) without loading models

3. **Run completely serially:**
   ```bash
   pytest -x  # Stop on first failure
   ```

4. **Check memory before running:**
   ```bash
   free -h  # Should have several GB free
   ```

## Summary

✅ Tests now run safely on any EC2 instance
✅ No more OOM crashes
✅ Automatic model mocking
✅ Serial execution by default
✅ Optional parallelization with `-n` flag
✅ Existing tests work without changes
