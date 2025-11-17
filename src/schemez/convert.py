"""BaseModel tools."""

from __future__ import annotations

import dataclasses
import inspect
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, Field, create_model

from schemez.docstrings import get_docstring_info
from schemez.schema import Schema


if TYPE_CHECKING:
    from llmling_agent.common_types import AnyCallable
    from pydantic.fields import FieldInfo


def get_union_args(tp: Any) -> tuple[Any, ...]:
    """Extract arguments of a Union type."""
    if isinstance(tp, TypeAliasType):
        tp = tp.__value__

    origin = get_origin(tp)
    if origin is Union or origin is UnionType:
        return get_args(tp)
    return ()


def get_function_model(func: AnyCallable, *, name: str | None = None) -> type[Schema]:
    """Convert a function's signature to a Pydantic model.

    Args:
        func: The function to convert (can be method)
        name: Optional name for the model

    Returns:
        Pydantic model representing the function parameters

    Example:
        >>> def greet(name: str, age: int | None = None) -> str:
        ...     '''Greet someone.
        ...     Args:
        ...         name: Person's name
        ...         age: Optional age
        ...     '''
        ...     return f"Hello {name}"
        >>> model = get_function_model(greet)
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)
    fields: dict[str, tuple[type, FieldInfo]] = {}
    description, param_docs = get_docstring_info(func, sig)

    for param_name, param in sig.parameters.items():
        # Skip self/cls for methods
        if param_name in {"self", "cls"}:
            continue

        type_hint = hints.get(param_name, Any)

        # Handle unions (including Optional)
        if union_args := get_union_args(type_hint):  # noqa: SIM102
            if len(union_args) == 2 and type(None) in union_args:  # noqa: PLR2004
                type_hint = next(t for t in union_args if t is not type(None))

        # Create field with defaults if available
        field = Field(
            default=... if param.default is param.empty else param.default,
            description=param_docs.get(param_name),  # TODO: Add docstring parsing
        )
        fields[param_name] = (type_hint, field)

    model_name = name or f"{func.__name__}Params"
    return create_model(model_name, **fields, __base__=Schema, __doc__=description)  # type: ignore[no-any-return, call-overload]


def get_ctor_basemodel(cls: type) -> type[Schema]:
    """Convert a class constructor to a Pydantic model.

    Args:
        cls: The class whose constructor to convert

    Returns:
        Pydantic model for the constructor parameters

    Example:
        >>> class Person:
        ...     def __init__(self, name: str, age: int | None = None):
        ...         self.name = name
        ...         self.age = age
        >>> model = get_ctor_basemodel(Person)
    """
    if issubclass(cls, BaseModel):
        if issubclass(cls, Schema):
            return cls

        # Create a new Schema-based model with the same fields
        fields = {}
        for field_name, field_info in cls.model_fields.items():
            field_type = field_info.annotation
            field_default = (
                field_info.default if field_info.default is not Ellipsis else ...
            )
            fields[field_name] = (field_type, field_default)

        return create_model(cls.__name__, **fields, __base__=Schema)  # type: ignore[call-overload, no-any-return]

    if dataclasses.is_dataclass(cls):
        fields = {}
        hints = get_type_hints(cls)
        for field in dataclasses.fields(cls):
            fields[field.name] = (hints[field.name], ...)
        return create_model(cls.__name__, __base__=Schema, **fields)  # type: ignore[no-any-return, call-overload]
    return get_function_model(cls.__init__, name=cls.__name__)


if __name__ == "__main__":

    class Person:
        """Person class."""

        def __init__(self, name: str, age: int | None = None) -> None:
            self.name = name
            self.age = age

        def func_google(self, name: str, age: int | None = None) -> None:
            """Do something."""

    model = get_function_model(Person.func_google)
    instance = model(name="Test", age=30)  # type: ignore[call-arg]
    print(instance, isinstance(instance, BaseModel))
