"""Tests for function schema creation."""

from __future__ import annotations

import pytest

from py2openai.functionschema import create_schema


def regular_function(x: int) -> str:
    """Regular standalone function.

    Args:
        x: An integer input
    """
    return str(x)


class TestMethods:
    def instance_method(self, x: int) -> str:
        """Regular instance method.

        Args:
            x: An integer input
        """
        return str(x)

    @classmethod
    def class_method(cls, x: int) -> str:
        """Class method example.

        Args:
            x: An integer input
        """
        return str(x)

    @staticmethod
    def static_method(x: int) -> str:
        """Static method example.

        Args:
            x: An integer input
        """
        return str(x)


def test_all_methods() -> None:
    """Test all method types with debug info."""
    for func in [
        regular_function,
        TestMethods.instance_method,
        TestMethods.class_method,
        TestMethods.static_method,
    ]:
        schema = create_schema(func)  # type: ignore[arg-type]
        fn_name = getattr(func, "__name__", "unknown")
        print(f"\n{fn_name} schema params: {list(schema.parameters['properties'])}")
        assert list(schema.parameters["properties"]) == ["x"]


if __name__ == "__main__":
    pytest.main([__file__])
