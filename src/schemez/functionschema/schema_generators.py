from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
import dataclasses
import importlib
import inspect
import types
from typing import Any, Literal, get_type_hints

import pydantic

from schemez.functionschema import FunctionSchema, create_schema, resolve_type_annotation
from schemez.functionschema.typedefs import ToolParameters


def create_schemas_from_callables(
    callables: dict[str, Callable[..., Any]],
    prefix: str | Literal[False] | None = None,
    exclude_private: bool = True,
) -> dict[str, FunctionSchema]:
    """Generate OpenAI function schemas from a dictionary of callables.

    Args:
        callables: Dictionary mapping names to callable objects
        prefix: Schema name prefix to prepend to function names.
               If None, no prefix. If False, use raw name.
               If string, uses that prefix.
        exclude_private: Whether to exclude callables starting with underscore

    Returns:
        Dictionary mapping qualified names to FunctionSchema objects

    Example:
        >>> def foo(x: int) -> str: ...
        >>> def bar(y: float) -> int: ...
        >>> callables = {'foo': foo, 'bar': bar}
        >>> schemas = create_schemas_from_callables(callables, prefix='math')
        >>> print(schemas['math.foo'])
    """
    schemas = {}
    for name, callable_obj in callables.items():
        if exclude_private and name.startswith("_"):  # Skip private members if requested
            continue
        # Generate schema key based on prefix setting
        key = name if prefix is False else f"{prefix}.{name}" if prefix else name
        schemas[key] = create_schema(callable_obj)
    return schemas


def create_schemas_from_class(
    cls: type,
    prefix: str | Literal[False] | None = None,
) -> dict[str, FunctionSchema]:
    """Generate OpenAI function schemas for all public methods in a class.

    Args:
        cls: The class to generate schemas from
        prefix: Schema name prefix. If None, uses class name.
               If False, no prefix. If string, uses that prefix.

    Returns:
        Dictionary mapping qualified method names to FunctionSchema objects

    Example:
        >>> class MyClass:
        ...     def my_method(self, x: int) -> str:
        ...         return str(x)
        >>> schemas = create_schemas_from_class(MyClass)
        >>> print(schemas['MyClass.my_method'])
    """
    callables: dict[str, Callable[..., Any]] = {}

    for name, attr in inspect.getmembers(cls):  # Get all attributes of the class
        if inspect.isfunction(attr) or inspect.ismethod(attr):
            callables[name] = attr
        elif isinstance(attr, classmethod | staticmethod):
            callables[name] = attr.__get__(None, cls)

    # Use default prefix of class name if not specified
    effective_prefix = cls.__name__ if prefix is None else prefix
    return create_schemas_from_callables(callables, prefix=effective_prefix)


def create_constructor_schema(cls: type) -> FunctionSchema:
    """Create OpenAI function schema from class constructor.

    Args:
        cls: Class to create schema for

    Returns:
        OpenAI function schema for class constructor
    """
    if isinstance(cls, type) and issubclass(cls, pydantic.BaseModel):
        properties = {}
        required = []
        for name, field in cls.model_fields.items():
            param_type = field.annotation
            properties[name] = resolve_type_annotation(
                param_type,
                description=field.description,
                default=field.default,
            )
            if field.is_required():
                required.append(name)

    # Handle dataclasses
    elif dataclasses.is_dataclass(cls):
        properties = {}
        required = []
        dc_fields = dataclasses.fields(cls)
        hints = get_type_hints(cls)

        for dc_field in dc_fields:
            param_type = hints[dc_field.name]
            properties[dc_field.name] = resolve_type_annotation(
                param_type, default=dc_field.default
            )
            if (
                dc_field.default is dataclasses.MISSING
                and dc_field.default_factory is dataclasses.MISSING
            ):
                required.append(dc_field.name)

    # Handle regular classes
    else:
        sig = inspect.signature(cls.__init__)
        hints = get_type_hints(cls.__init__)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            param_type = hints.get(name, Any)
            properties[name] = resolve_type_annotation(
                param_type,
                default=param.default,
            )

            if param.default is param.empty:
                required.append(name)

    name = f"create_{cls.__name__}"
    description = inspect.getdoc(cls) or f"Create {cls.__name__} instance"

    # Create parameters with required list included
    params = ToolParameters({
        "type": "object",
        "properties": properties,
        "required": required,
    })

    return FunctionSchema(
        name=name, description=description, parameters=params, required=required
    )


def create_schemas_from_module(
    module: types.ModuleType | str,
    include_functions: list[str] | None = None,
    prefix: str | Literal[False] | None = None,
) -> dict[str, FunctionSchema]:
    """Generate OpenAI function schemas from a Python module's functions.

    Args:
        module: Either a ModuleType object or string name of module to analyze
        include_functions: Optional list of function names to specifically include
        prefix: Schema name prefix. If None, uses module name.
                If False, no prefix. If string, uses that prefix.

    Returns:
        Dictionary mapping function names to FunctionSchema objects

    Raises:
        ImportError: If module string name cannot be imported

    Example:
        >>> import math
        >>> schemas = create_schemas_from_module(math, ['sin', 'cos'])
        >>> print(schemas['math.sin'])
    """
    # Resolve module if string name provided
    mod = (
        module
        if isinstance(module, types.ModuleType)
        else importlib.import_module(module)
    )

    # Get all functions from module
    callables: dict[str, Callable[..., Any]] = {
        name: func
        for name, func in inspect.getmembers(mod, predicate=inspect.isfunction)
        if include_functions is None
        or (name in include_functions and func.__module__.startswith(mod.__name__))
    }

    # Use default prefix of module name if not specified
    effective_prefix = mod.__name__ if prefix is None else prefix
    return create_schemas_from_callables(callables, prefix=effective_prefix)


if __name__ == "__main__":
    schemas = create_schemas_from_module(__name__)
    print(schemas)
