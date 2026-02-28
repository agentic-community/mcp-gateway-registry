"""Unit tests for safe arithmetic evaluation."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# agents/ is a standalone script directory, not an installed package.
# Add it to sys.path so that `from registry_client import ...` inside
# agents/agent.py resolves to agents/registry_client.py.
_AGENTS_DIR = str(Path(__file__).resolve().parents[2] / "agents")
if _AGENTS_DIR not in sys.path:
    sys.path.insert(0, _AGENTS_DIR)

# The root conftest installs a MockFaissModule into sys.modules["faiss"] that
# lacks a __spec__ attribute. When agents.agent imports langchain_anthropic,
# which imports transformers, which calls importlib.util.find_spec("faiss"),
# Python raises ValueError: faiss.__spec__ is not set. Patch __spec__ here so
# the import chain succeeds.
if "faiss" in sys.modules:
    faiss_mod = sys.modules["faiss"]
    if getattr(faiss_mod, "__spec__", None) is None:
        faiss_mod.__spec__ = MagicMock(name="faiss.__spec__")
else:
    _faiss_mock = MagicMock()
    _faiss_mock.__spec__ = MagicMock(name="faiss.__spec__")
    sys.modules["faiss"] = _faiss_mock

import pytest
from agents.agent import _safe_eval_arithmetic


class TestSafeEvalArithmetic:
    """Tests for _safe_eval_arithmetic function."""

    def test_basic_addition(self):
        """Test basic addition."""
        assert _safe_eval_arithmetic("2 + 2") == 4
        assert _safe_eval_arithmetic("10 + 5") == 15

    def test_basic_subtraction(self):
        """Test basic subtraction."""
        assert _safe_eval_arithmetic("10 - 3") == 7
        assert _safe_eval_arithmetic("5 - 10") == -5

    def test_basic_multiplication(self):
        """Test basic multiplication."""
        assert _safe_eval_arithmetic("4 * 5") == 20
        assert _safe_eval_arithmetic("3 * 7") == 21

    def test_basic_division(self):
        """Test basic division."""
        assert _safe_eval_arithmetic("20 / 4") == 5.0
        assert _safe_eval_arithmetic("10 / 2") == 5.0

    def test_exponentiation(self):
        """Test exponentiation."""
        assert _safe_eval_arithmetic("2 ** 3") == 8
        assert _safe_eval_arithmetic("5 ** 2") == 25

    def test_floor_division(self):
        """Test floor division."""
        assert _safe_eval_arithmetic("10 // 3") == 3
        assert _safe_eval_arithmetic("20 // 4") == 5

    def test_modulo(self):
        """Test modulo operation."""
        assert _safe_eval_arithmetic("10 % 3") == 1
        assert _safe_eval_arithmetic("20 % 7") == 6

    def test_complex_expression(self):
        """Test complex nested expression."""
        assert _safe_eval_arithmetic("2 + 3 * 4") == 14
        assert _safe_eval_arithmetic("(2 + 3) * 4") == 20

    def test_negative_numbers(self):
        """Test negative numbers."""
        assert _safe_eval_arithmetic("-5") == -5
        assert _safe_eval_arithmetic("-5 + 3") == -2

    def test_float_operations(self):
        """Test floating point operations."""
        assert _safe_eval_arithmetic("3.5 + 2.5") == 6.0
        assert _safe_eval_arithmetic("10.0 / 4.0") == 2.5

    def test_division_by_zero(self):
        """Test division by zero raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            _safe_eval_arithmetic("10 / 0")

    def test_blocks_import(self):
        """Test that __import__ is blocked."""
        with pytest.raises(ValueError, match="Unsupported expression type"):
            _safe_eval_arithmetic("__import__('os')")

    def test_blocks_eval(self):
        """Test that eval function call is blocked."""
        with pytest.raises(ValueError, match="Unsupported expression type"):
            _safe_eval_arithmetic("eval('2+2')")

    def test_blocks_function_calls(self):
        """Test that arbitrary function calls are blocked."""
        with pytest.raises(ValueError, match="Unsupported expression type"):
            _safe_eval_arithmetic("print(5)")

    def test_blocks_attribute_access(self):
        """Test that attribute access is blocked."""
        with pytest.raises(ValueError, match="Unsupported expression type"):
            _safe_eval_arithmetic("os.system('ls')")

    def test_blocks_names(self):
        """Test that variable names are blocked."""
        with pytest.raises(ValueError, match="Unsupported expression type"):
            _safe_eval_arithmetic("x + 5")

    def test_length_limit_protection(self):
        """Test that long valid expressions are handled correctly."""
        long_expr = " + ".join(["1"] * 50)
        result = _safe_eval_arithmetic(long_expr)
        assert result == 50

    def test_blocks_large_exponents(self):
        """Test that exponents over 100 are blocked."""
        with pytest.raises(ValueError, match="Exponent too large"):
            _safe_eval_arithmetic("2 ** 101")

        with pytest.raises(ValueError, match="Exponent too large"):
            _safe_eval_arithmetic("9 ** 999")

        # Normal exponents should still work
        assert _safe_eval_arithmetic("2 ** 10") == 1024
        assert _safe_eval_arithmetic("3 ** 4") == 81
