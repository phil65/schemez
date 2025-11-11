"""Tests for ToolCodeGenerator with focus on generic type exclusion."""

import re

import pytest

from schemez.code_generation.tool_code_generator import ToolCodeGenerator


class RunContext[T]:
    """Mock RunContext class for testing."""

    def __init__(self, data: T):
        self.data = data


class RequestContext[T, U]:
    """Mock multi-parameter generic context."""

    def __init__(self, request: T, user: U):
        self.request = request
        self.user = user


class UserData:
    """Mock user data type."""


class RequestData:
    """Mock request data type."""


def simple_function(name: str, age: int) -> str:
    """Simple function without context parameters.

    Args:
        name: The name
        age: The age

    Returns:
        A greeting string
    """
    return f"Hello {name}, you are {age} years old"


def function_with_generic_context(
    name: str, context: RunContext[UserData], optional_param: int = 42
) -> str:
    """Function with single generic context parameter.

    Args:
        name: The name parameter
        context: Context parameter that should be excluded
        optional_param: Optional parameter

    Returns:
        A greeting string
    """
    return f"Hello {name}!"


def function_with_multi_generic_context(
    query: str, ctx: RequestContext[RequestData, UserData], limit: int = 10
) -> dict:
    """Function with multi-parameter generic context.

    Args:
        query: Search query
        ctx: Request context with multiple type parameters
        limit: Result limit

    Returns:
        Search results
    """
    return {"query": query, "limit": limit}


def function_with_non_generic_context(
    name: str,
    context: RunContext,  # Non-generic usage
    value: float = 1.0,
) -> bool:
    """Function with non-generic context parameter.

    Args:
        name: The name
        context: Non-generic context
        value: Some value

    Returns:
        Success flag
    """
    return True


def function_with_multiple_contexts(
    data: str,
    run_ctx: RunContext[UserData],
    req_ctx: RequestContext[RequestData, UserData],
) -> str:
    """Function with multiple context parameters.

    Args:
        data: Input data
        run_ctx: Run context
        req_ctx: Request context

    Returns:
        Processed data
    """
    return data.upper()


class TestToolCodeGenerator:
    """Test suite for ToolCodeGenerator."""

    def test_simple_function_signature(self):
        """Test signature extraction for simple function without context."""
        generator = ToolCodeGenerator.from_callable(simple_function, exclude_types=[])

        signature = generator.get_function_signature()

        # Should contain all parameters
        assert "name: str" in signature
        assert "age: int" in signature
        assert "simple_function" in signature

    def test_generic_context_exclusion(self):
        """Test that generic context parameters are properly excluded."""
        generator = ToolCodeGenerator.from_callable(
            function_with_generic_context, exclude_types=[RunContext]
        )

        signature = generator.get_function_signature()

        # Context parameter should be excluded
        assert not re.search(r"\bcontext:", signature)

        # Other parameters should be present
        assert "name: str" in signature
        assert "optional_param: int = 42" in signature

    def test_generic_context_not_excluded_when_not_specified(self):
        """Test that generic context parameters appear when not excluded."""
        generator = ToolCodeGenerator.from_callable(
            function_with_generic_context,
            exclude_types=[],  # No exclusions
        )

        signature = generator.get_function_signature()

        # All parameters should be present
        assert re.search(r"\bcontext:", signature)
        assert "name: str" in signature
        assert "optional_param: int = 42" in signature

    def test_multi_parameter_generic_exclusion(self):
        """Test exclusion of multi-parameter generic types."""
        generator = ToolCodeGenerator.from_callable(
            function_with_multi_generic_context, exclude_types=[RequestContext]
        )

        signature = generator.get_function_signature()

        # Context parameter should be excluded
        assert not re.search(r"\bctx:", signature)

        # Other parameters should be present
        assert "query: str" in signature
        assert "limit: int = 10" in signature

    def test_non_generic_context_exclusion(self):
        """Test that non-generic usage of generic types is also excluded."""
        generator = ToolCodeGenerator.from_callable(
            function_with_non_generic_context, exclude_types=[RunContext]
        )

        signature = generator.get_function_signature()

        # Context parameter should be excluded
        assert not re.search(r"\bcontext:", signature)

        # Other parameters should be present
        assert "name: str" in signature
        assert "value: float = 1.0" in signature

    def test_multiple_context_types_exclusion(self):
        """Test exclusion of multiple different context types."""
        generator = ToolCodeGenerator.from_callable(
            function_with_multiple_contexts, exclude_types=[RunContext, RequestContext]
        )

        signature = generator.get_function_signature()

        # Both context parameters should be excluded
        assert not re.search(r"\brun_ctx:", signature)
        assert not re.search(r"\breq_ctx:", signature)

        # Data parameter should be present
        assert "data: str" in signature

    def test_partial_context_exclusion(self):
        """Test excluding only some context types."""
        generator = ToolCodeGenerator.from_callable(
            function_with_multiple_contexts,
            exclude_types=[RunContext],  # Only exclude RunContext
        )

        signature = generator.get_function_signature()

        # Only RunContext should be excluded
        assert not re.search(r"\brun_ctx:", signature)
        assert re.search(r"\breq_ctx:", signature)  # RequestContext should remain

        # Data parameter should be present
        assert "data: str" in signature

    def test_is_context_parameter_with_generics(self):
        """Test the _is_context_parameter method with generic types."""
        generator = ToolCodeGenerator.from_callable(
            function_with_generic_context, exclude_types=[RunContext]
        )

        # Should identify generic context parameter correctly
        assert generator._is_context_parameter("context")
        assert not generator._is_context_parameter("name")
        assert not generator._is_context_parameter("optional_param")

    def test_is_context_parameter_with_multi_generics(self):
        """Test _is_context_parameter with multi-parameter generics."""
        generator = ToolCodeGenerator.from_callable(
            function_with_multi_generic_context, exclude_types=[RequestContext]
        )

        # Should identify multi-parameter generic correctly
        assert generator._is_context_parameter("ctx")
        assert not generator._is_context_parameter("query")
        assert not generator._is_context_parameter("limit")

    def test_no_exclude_types_specified(self):
        """Test behavior when no exclude_types are specified."""
        generator = ToolCodeGenerator.from_callable(function_with_generic_context)

        # Should default to empty list
        assert generator.exclude_types == []

        signature = generator.get_function_signature()

        # All parameters should be present
        assert re.search(r"\bcontext:", signature)
        assert "name: str" in signature

    def test_empty_exclude_types_list(self):
        """Test behavior with explicitly empty exclude_types list."""
        generator = ToolCodeGenerator.from_callable(
            function_with_generic_context, exclude_types=[]
        )

        signature = generator.get_function_signature()

        # All parameters should be present
        assert re.search(r"\bcontext:", signature)
        assert "name: str" in signature

    def test_callable_assignment_bug_fix(self):
        """Test that the callable is properly assigned (regression test)."""
        generator = ToolCodeGenerator.from_callable(simple_function)

        # Should have the actual function, not the builtin callable
        assert generator.callable == simple_function
        assert generator.callable.__name__ == "simple_function"
        assert callable(generator.callable)

    @pytest.mark.parametrize(
        ("func", "exclude_types", "should_exclude"),
        [
            (function_with_generic_context, [RunContext], ["context"]),
            (function_with_multi_generic_context, [RequestContext], ["ctx"]),
            (
                function_with_multiple_contexts,
                [RunContext, RequestContext],
                ["run_ctx", "req_ctx"],
            ),
            (function_with_multiple_contexts, [RunContext], ["run_ctx"]),
            (simple_function, [RunContext], []),  # No context to exclude
        ],
    )
    def test_parametrized_exclusions(self, func, exclude_types, should_exclude):
        """Parametrized test for various exclusion scenarios."""
        generator = ToolCodeGenerator.from_callable(func, exclude_types=exclude_types)
        signature = generator.get_function_signature()

        for param_name in should_exclude:
            assert not re.search(rf"\b{param_name}:", signature), (
                f"Parameter {param_name} should be excluded"
            )

    def test_exclude_types_does_not_affect_return_type(self):
        """Test that excluding types doesn't affect return type inference."""
        generator_with_exclusion = ToolCodeGenerator.from_callable(
            function_with_generic_context, exclude_types=[RunContext]
        )

        generator_without_exclusion = ToolCodeGenerator.from_callable(
            function_with_generic_context, exclude_types=[]
        )

        sig_with = generator_with_exclusion.get_function_signature()
        sig_without = generator_without_exclusion.get_function_signature()

        # Return types should be the same
        return_with = sig_with.split(" -> ")[-1]
        return_without = sig_without.split(" -> ")[-1]

        assert return_with == return_without


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
