# Test Suite Modernization - TODO Items

## Pending Enhancements

### 1. Test Timeout Enforcement
**Status**: Not Implemented  
**Requirement**: 12.6 - Complete in under 60 seconds for fast feedback  
**Reason**: `pytest-timeout` plugin not currently in dependencies  

**To Implement:**
1. Add `pytest-timeout>=2.2.0` to `[project.optional-dependencies] dev` in `pyproject.toml`
2. Add `timeout = 60` to `[tool.pytest.ini_options]` in `pyproject.toml`
3. Run `uv sync --extra dev` to install
4. Verify with `uv run pytest --co -q`

**Priority**: Medium (nice to have, not blocking)

---

## Completed Items
- ✅ Pytest configuration with markers
- ✅ Coverage settings (80% threshold)
- ✅ Test discovery patterns
- ✅ Asyncio configuration
