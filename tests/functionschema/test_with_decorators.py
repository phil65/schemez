"""Tests for schema generation from decorated functions."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import pytest

from py2openai.functionschema import create_schema


if TYPE_CHECKING:
    from collections.abc import Callable


def unwrapped_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """A basic decorator that doesn't use functools.wraps."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper function."""
        return func(*args, **kwargs)

    return wrapper


def wrapped_decorator[T](func: Callable[..., T]) -> Callable[..., T]:
    """A decorator that uses functools.wraps to preserve metadata."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        """Wrapper function."""
        return func(*args, **kwargs)

    return wrapper


def parameterized_decorator[T](
    *, tag: str
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """A parameterized decorator that uses functools.wraps."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            """Wrapper function."""
            return func(*args, **kwargs)

        return wrapper

    return decorator


def stacked_decorator[T](func: Callable[..., T]) -> Callable[..., T]:
    """Another wrapped decorator for stacking."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        """Wrapper function."""
        return func(*args, **kwargs)

    return wrapper


@unwrapped_decorator
def unwrapped_function(x: int, y: str) -> dict[str, Any]:
    """Test function with unwrapped decorator.

    Args:
        x: An integer
        y: A string

    Returns:
        A dictionary with the input values
    """
    return {"x": x, "y": y}


@wrapped_decorator
def wrapped_function(x: int, y: str) -> dict[str, Any]:
    """Test function with wrapped decorator.

    Args:
        x: An integer
        y: A string

    Returns:
        A dictionary with the input values
    """
    return {"x": x, "y": y}


@parameterized_decorator(tag="test")
def parameterized_function(x: int, y: str) -> dict[str, Any]:
    """Test function with parameterized decorator.

    Args:
        x: An integer
        y: A string

    Returns:
        A dictionary with the input values
    """
    return {"x": x, "y": y}


@stacked_decorator
@wrapped_decorator
@parameterized_decorator(tag="multiple")
def multi_decorated_function(x: int, y: str) -> dict[str, Any]:
    """Test function with multiple decorators.

    Args:
        x: An integer
        y: A string

    Returns:
        A dictionary with the input values
    """
    return {"x": x, "y": y}


def test_unwrapped_decorator() -> None:
    """Test schema generation for function with unwrapped decorator."""
    schema = create_schema(unwrapped_function)

    # The schema should reflect the wrapper function, not the original
    assert schema.name == "wrapper"
    # Should have wrapper's docstring, not the original function's
    assert schema.description == "Wrapper function."

    # When unwrapped, we lose both metadata AND type hints
    # The wrapper function has no type hints, so we get empty properties
    assert schema.parameters == {"type": "object", "properties": {}}
    assert schema.returns == {"type": "object"}


def test_wrapped_decorator() -> None:
    """Test schema generation for function with wrapped decorator."""
    schema = create_schema(wrapped_function)

    # Metadata should be preserved
    assert schema.name == "wrapped_function"
    assert schema.description == "Test function with wrapped decorator."

    # Parameters and return type should be correct
    props = schema.parameters["properties"]
    assert props["x"]["type"] == "integer"
    assert props["y"]["type"] == "string"
    assert props["x"].get("description") == "An integer"
    assert props["y"].get("description") == "A string"
    assert schema.returns == {"type": "object"}


def test_parameterized_decorator() -> None:
    """Test schema generation for function with parameterized decorator."""
    schema = create_schema(parameterized_function)

    # Metadata should be preserved
    assert schema.name == "parameterized_function"
    assert schema.description == "Test function with parameterized decorator."

    # Parameters and return type should be correct
    props = schema.parameters["properties"]
    assert props["x"]["type"] == "integer"
    assert props["y"]["type"] == "string"
    assert props["x"].get("description") == "An integer"
    assert props["y"].get("description") == "A string"
    assert schema.returns == {"type": "object"}


def test_multi_decorated_function() -> None:
    """Test schema generation for function with multiple decorators."""
    schema = create_schema(multi_decorated_function)

    # Metadata should be preserved through all decorators
    assert schema.name == "multi_decorated_function"
    assert schema.description == "Test function with multiple decorators."

    # Parameters and return type should be correct
    props = schema.parameters["properties"]
    assert props["x"]["type"] == "integer"
    assert props["y"]["type"] == "string"
    assert props["x"].get("description") == "An integer"
    assert props["y"].get("description") == "A string"
    assert schema.returns == {"type": "object"}


def test_callable_class_decorator() -> None:
    """Test schema generation for function decorated with a callable class."""

    class CallableDecorator:
        """A decorator implemented as a callable class."""

        def __call__[T](self, func: Callable[..., T]) -> Callable[..., T]:
            """Make the class callable as a decorator."""

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                print("Class decorator")
                return func(*args, **kwargs)

            return wrapper

    decorator = CallableDecorator()

    @decorator
    def class_decorated_function(x: int, y: str) -> dict[str, Any]:
        """Test function with class decorator.

        Args:
            x: An integer
            y: A string

        Returns:
            A dictionary with the input values
        """
        return {"x": x, "y": y}

    schema = create_schema(class_decorated_function)

    # Metadata should be preserved
    assert schema.name == "class_decorated_function"
    assert schema.description == "Test function with class decorator."

    # Parameters and return type should be correct
    props = schema.parameters["properties"]
    assert props["x"]["type"] == "integer"
    assert props["y"]["type"] == "string"
    assert props["x"].get("description") == "An integer"
    assert props["y"].get("description") == "A string"
    assert schema.returns == {"type": "object"}


if __name__ == "__main__":
    pytest.main([__file__])
