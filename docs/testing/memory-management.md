# Test Suite Memory Management

## Problem

Running the full test suite with parallel execution can cause Out-of-Memory (OOM) crashes on EC2 instances, especially smaller instances with limited RAM.

### Root Cause

The test suite includes:
- **38 test files** with over 14,000 lines of test code
- Heavy dependencies including:
  - Sentence-transformers embedding models (~120-200MB per process)
  - Vector search indexes
  - Full FastAPI application stack

When using pytest-xdist with `-n auto`, pytest spawns one worker process per CPU core (4 workers on a 4-core EC2 instance). Each worker loads:
- The embedding model
- Vector search indexes
- Test fixtures and data
- The full application

**Memory multiplication:** 4 workers × ~500MB per worker = ~2GB+ just for test processes

This can overwhelm EC2 instances with 8-16GB of RAM, especially when the OS and other services are also running.

## Solution

### Default Behavior (Serial Execution)

The test suite now runs **serially by default** to prevent OOM crashes:

```bash
# Safe for all EC2 instances - runs tests one at a time
python scripts/test.py full
```

### Parallel Execution (Use with Caution)

If you have sufficient memory (16GB+ RAM), you can enable parallel execution:

```bash
# Run with 2 workers (safer for smaller EC2 instances)
python scripts/test.py full -n 2

# Run fast tests with 2 workers
python scripts/test.py fast

# Run unit tests with 4 workers (requires more memory)
python scripts/test.py unit -n 4
```

### Monitoring Memory Usage

Before running tests with parallelization, check available memory:

```bash
# Check memory usage
free -h

# Monitor memory in real-time
watch -n 1 free -h

# Check processes by memory usage
ps aux --sort=-%mem | head -20
```

### Memory Guidelines

| EC2 Instance Type | Recommended Workers | Notes |
|-------------------|---------------------|-------|
| t3.small (2GB)    | 1 (serial)          | Parallel execution will crash |
| t3.medium (4GB)   | 1 (serial)          | May work with -n 2 for unit tests |
| t3.large (8GB)    | 2                   | Safe for most tests |
| t3.xlarge (16GB)  | 3-4                 | Can handle full parallelization |
| t3.2xlarge (32GB) | auto                | Full parallel execution safe |

## Test Commands

### Recommended Commands for EC2

```bash
# Check dependencies first
python scripts/test.py check

# Run unit tests only (fastest, safest)
python scripts/test.py unit

# Run integration tests
python scripts/test.py integration

# Run fast tests with 2 workers
python scripts/test.py fast

# Run full test suite serially (safe but slow)
python scripts/test.py full

# Generate coverage report (always serial)
python scripts/test.py coverage
```

### Advanced Options

```bash
# Run specific domain tests
python scripts/test.py auth         # Authentication tests
python scripts/test.py servers      # Server management tests
python scripts/test.py search       # Search and AI tests
python scripts/test.py health       # Health monitoring tests
python scripts/test.py core         # Core infrastructure tests

# Enable debug logging
python scripts/test.py unit --debug

# Run with custom worker count
python scripts/test.py unit -n 3
```

## Direct pytest Usage

If using pytest directly, be aware of memory implications:

```bash
# DANGEROUS: May crash EC2 instance
pytest -n auto  # Spawns workers = CPU cores

# SAFER: Limit workers
pytest -n 2

# SAFEST: Serial execution (no -n flag)
pytest
```

## Optimizations

### For Local Development

If running locally with sufficient RAM (16GB+):

```bash
# Fast parallel execution for unit tests
pytest tests/unit -n auto

# Fast parallel for specific domains
pytest tests/unit/auth -n auto
```

### For CI/CD

GitHub Actions and other CI environments typically have limited memory. Use:

```bash
# Serial execution in CI
pytest

# Or limit workers
pytest -n 2
```

### Future Improvements

To further reduce memory usage:

1. **Mock Heavy Dependencies**: Mock sentence-transformers and search dependencies in unit tests
2. **Test Fixtures Optimization**: Share model loading across tests using session-scoped fixtures
3. **Test Categorization**: Split heavy integration tests from lightweight unit tests
4. **Lazy Loading**: Only load ML models when actually needed in tests

## Troubleshooting

### OOM Crash Symptoms

- EC2 instance becomes unresponsive
- SSH connection drops
- Test suite hangs indefinitely
- System logs show "Out of memory: Killed process"

### Recovery Steps

1. Reboot the EC2 instance if unresponsive
2. Run tests serially: `python scripts/test.py full`
3. Consider upgrading to a larger instance type
4. Run tests in batches by domain:
   ```bash
   python scripts/test.py auth
   python scripts/test.py servers
   python scripts/test.py search
   ```

### Debugging Memory Issues

```bash
# Check which process is using memory during tests
watch -n 1 'ps aux --sort=-%mem | head -20'

# Check for OOM killer logs
dmesg | grep -i "out of memory"
sudo journalctl | grep -i "out of memory"
```

## Summary

- **Default:** Tests run serially to prevent OOM crashes
- **Safe Parallel:** Use `-n 2` for faster execution on typical EC2 instances
- **Full Parallel:** Only use `-n auto` or higher worker counts on instances with 16GB+ RAM
- **Monitor:** Always monitor memory usage when experimenting with parallelization
