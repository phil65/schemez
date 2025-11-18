"""Module for creating OpenAI function schemas from Python functions."""

from __future__ import annotations

from collections.abc import (
    Callable,  # noqa: TC003
    Sequence,  # noqa: F401
)
import dataclasses
from datetime import date, datetime, time, timedelta, timezone
import decimal
import enum
import inspect
import ipaddress
from pathlib import Path
import re
import types
import typing
from typing import (
    Annotated,
    Any,
    Literal,
    NotRequired,
    Required,
    TypeGuard,
    get_args,
    get_origin,
)
from uuid import UUID

from schemez import log


if typing.TYPE_CHECKING:
    from pydantic import BaseModel

    from schemez.functionschema.typedefs import Property


logger = log.get_logger(__name__)

FunctionType = Literal["sync", "async", "sync_generator", "async_generator"]


def resolve_type_annotation(
    typ: Any,
    description: str | None = None,
    default: Any = inspect.Parameter.empty,
    is_parameter: bool = True,
) -> Property:
    """Resolve a type annotation into an OpenAI schema type.

    Args:
        typ: Type to resolve
        description: Optional description
        default: Default value if any
        is_parameter: Whether this is for a parameter (affects dict schema)
    """
    from schemez.functionschema.typedefs import _create_simple_property

    schema: dict[str, Any] = {}

    # Handle anyOf/oneOf fields
    if isinstance(typ, dict) and ("anyOf" in typ or "oneOf" in typ):
        # For simplicity, we'll treat it as a string that can be null
        # This is a common pattern for optional fields
        schema["type"] = "string"
        if default is not None:
            schema["default"] = default
        if description:
            schema["description"] = description
        return _create_simple_property(
            type_str="string",
            description=description,
            default=default,
        )

    # Handle Annotated types first
    if get_origin(typ) is Annotated:
        # Get the underlying type (first argument)
        base_type = get_args(typ)[0]
        return resolve_type_annotation(
            base_type,
            description=description,
            default=default,
            is_parameter=is_parameter,
        )

    origin = get_origin(typ)
    args = get_args(typ)

    # Handle Union types (including Optional)
    if origin in {typing.Union, types.UnionType}:  # pyright: ignore
        # For Optional (union with None), filter out None type
        non_none_types = [t for t in args if t is not type(None)]
        if non_none_types:
            prop = resolve_type_annotation(
                non_none_types[0],
                description=description,
                default=default,
                is_parameter=is_parameter,
            )
            schema.update(prop)
        else:
            schema["type"] = "string"  # Fallback for Union[]

    # Handle dataclasses
    elif dataclasses.is_dataclass(typ):
        fields = dataclasses.fields(typ)
        properties = {}
        required = []
        for field in fields:
            properties[field.name] = resolve_type_annotation(
                field.type,
                is_parameter=is_parameter,
            )
            # Field is required if it has no default value and no default_factory
            if (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            ):
                required.append(field.name)

        schema = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
    elif typing.is_typeddict(typ):
        properties = {}
        required = []
        for field_name, field_type in typ.__annotations__.items():
            # Check if field is wrapped in Required/NotRequired
            origin = get_origin(field_type)
            if origin is Required:
                is_required = True
                field_type = get_args(field_type)[0]
            elif origin is NotRequired:
                is_required = False
                field_type = get_args(field_type)[0]
            else:
                # Fall back to checking __required_keys__
                is_required = field_name in getattr(
                    typ, "__required_keys__", {field_name}
                )

            properties[field_name] = resolve_type_annotation(
                field_type,
                is_parameter=is_parameter,
            )
            if is_required:
                required.append(field_name)

        schema.update({"type": "object", "properties": properties})
        if required:
            schema["required"] = required
    # Handle mappings - updated check
    elif (
        origin in {dict, typing.Dict}  # noqa: UP006
        or (origin is not None and isinstance(origin, type) and issubclass(origin, dict))
    ):
        schema["type"] = "object"
        if is_parameter:  # Only add additionalProperties for parameters
            # Dict[K, V] should have at least 2 type arguments for key and value
            min_dict_args = 2
            if (
                len(args) >= min_dict_args
            ):  # Dict[K, V] - use value type for additionalProperties
                value_type = args[1]
                # Special case: Any should remain as True for backward compatibility
                if value_type is Any:
                    schema["additionalProperties"] = True
                else:
                    schema["additionalProperties"] = resolve_type_annotation(
                        value_type,
                        is_parameter=is_parameter,
                    )
            else:
                schema["additionalProperties"] = True

    # Handle sequences
    elif origin in {
        list,
        set,
        tuple,
        frozenset,
        typing.List,  # noqa: UP006  # pyright: ignore
        typing.Set,  # noqa: UP006  # pyright: ignore
    } or (
        origin is not None
        and origin.__module__ == "collections.abc"
        and origin.__name__ in {"Sequence", "MutableSequence", "Collection"}
    ):
        schema["type"] = "array"
        item_type = args[0] if args else Any
        schema["items"] = resolve_type_annotation(
            item_type,
            is_parameter=is_parameter,
        )

    # Handle literals
    elif origin is typing.Literal:
        schema["type"] = "string"
        schema["enum"] = list(args)

    # Handle basic types
    elif isinstance(typ, type):
        if issubclass(typ, enum.Enum):
            schema["type"] = "string"
            schema["enum"] = [e.value for e in typ]

        # Basic types
        elif typ in {str, Path, UUID, re.Pattern}:
            schema["type"] = "string"
        elif typ is int:
            schema["type"] = "integer"
        elif typ in {float, decimal.Decimal}:
            schema["type"] = "number"
        elif typ is bool:
            schema["type"] = "boolean"

        # String formats
        elif typ is datetime:
            schema["type"] = "string"
            schema["format"] = "date-time"
            if description:
                description = f"{description} (ISO 8601 format)"
        elif typ is date:
            schema["type"] = "string"
            schema["format"] = "date"
            if description:
                description = f"{description} (ISO 8601 format)"
        elif typ is time:
            schema["type"] = "string"
            schema["format"] = "time"
            if description:
                description = f"{description} (ISO 8601 format)"
        elif typ is timedelta:
            schema["type"] = "string"
            if description:
                description = f"{description} (ISO 8601 duration)"
        elif typ is timezone:
            schema["type"] = "string"
            if description:
                description = f"{description} (IANA timezone name)"
        elif typ is UUID:
            schema["type"] = "string"
        elif typ in (bytes, bytearray):
            schema["type"] = "string"
            if description:
                description = f"{description} (base64 encoded)"
        elif typ is ipaddress.IPv4Address or typ is ipaddress.IPv6Address:
            schema["type"] = "string"
        elif typ is complex:
            schema.update({
                "type": "object",
                "properties": {
                    "real": {"type": "number"},
                    "imag": {"type": "number"},
                },
            })
        # Check for Pydantic BaseModel
        elif hasattr(typ, "model_fields"):
            # It's a Pydantic v1 or v2 model
            fields = typ.model_fields
            properties = {}
            required = []
            for field_name, field_info in fields.items():
                field_type = field_info.annotation
                properties[field_name] = resolve_type_annotation(
                    field_type,
                    is_parameter=is_parameter,
                )
                if field_info.is_required():
                    required.append(field_name)

            schema = {"type": "object", "properties": properties}
            if required:
                schema["required"] = required
        # Default to object for unknown types
        else:
            schema["type"] = "object"
    else:
        # Default for unmatched types
        schema["type"] = "string"

    # Add description if provided
    if description is not None:
        schema["description"] = description

    # Add default if provided and not empty
    if default is not inspect.Parameter.empty:
        schema["default"] = default

    from schemez.functionschema.typedefs import (
        _create_array_property,
        _create_object_property,
        _create_simple_property,
    )

    if schema["type"] == "array":
        return _create_array_property(
            items=schema["items"],
            description=schema.get("description"),
        )
    if schema["type"] == "object":
        prop = _create_object_property(description=schema.get("description"))
        if "properties" in schema:
            prop["properties"] = schema["properties"]
        if "additionalProperties" in schema:
            prop["additionalProperties"] = schema["additionalProperties"]  # pyright: ignore[reportGeneralTypeIssues]
        if "required" in schema:
            prop["required"] = schema["required"]
        return prop

    return _create_simple_property(
        type_str=schema["type"],
        description=schema.get("description"),
        enum_values=schema.get("enum"),
        default=default if default is not inspect.Parameter.empty else None,
        fmt=schema.get("format"),
    )


def determine_function_type(func: Callable[..., Any]) -> FunctionType:
    """Determine the type of the function."""
    if inspect.isasyncgenfunction(func):
        return "async_generator"
    if inspect.isgeneratorfunction(func):
        return "sync_generator"
    if inspect.iscoroutinefunction(func):
        return "async"
    return "sync"


def get_param_type(param_details: Property) -> type[Any]:
    """Get the Python type for a parameter based on its schema details."""
    if "enum" in param_details:
        # For enum parameters, we just use str since we can't reconstruct
        # the exact enum class
        return str

    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return type_map.get(param_details.get("type", "string"), Any)


def is_optional_type(typ: type) -> TypeGuard[type]:
    """Check if a type is Optional[T] or T | None.

    Args:
        typ: Type to check

    Returns:
        True if the type is Optional, False otherwise
    """
    origin = get_origin(typ)
    if origin not in {typing.Union, types.UnionType}:  # pyright: ignore
        return False
    args = get_args(typ)
    # Check if any of the union members is None or NoneType
    return any(arg is type(None) for arg in args)


def types_match(annotation: Any, exclude_type: type) -> bool:
    """Check if annotation matches exclude_type using various strategies."""
    try:
        # Direct type match
        if annotation is exclude_type:
            return True

        # Handle generic types - get origin for comparison
        origin_annotation = get_origin(annotation)
        if origin_annotation is exclude_type:
            return True

        # String-based comparison for forward references and __future__.annotations
        annotation_str = str(annotation)
        exclude_type_name = exclude_type.__name__
        exclude_type_full_name = f"{exclude_type.__module__}.{exclude_type.__name__}"

        # Check various string representations
        if (
            exclude_type_name in annotation_str
            or exclude_type_full_name in annotation_str
        ):
            # Be more specific to avoid false positives
            # Check if it's the exact type name, not just a substring
            import re

            patterns = [
                rf"\b{re.escape(exclude_type_name)}\b",
                rf"\b{re.escape(exclude_type_full_name)}\b",
            ]
            if any(re.search(pattern, annotation_str) for pattern in patterns):
                return True

    except Exception:  # noqa: BLE001
        pass

    return False


def pydantic_model_to_signature(
    model: type[BaseModel],
    return_type: type,
) -> inspect.Signature:
    """Convert a Pydantic model to an inspect.Signature.

    Model fields are represented as keyword-only parameters.

    Args:
        model: The Pydantic model to convert.
        return_type: The return type of the function.
    """
    parameters: list[inspect.Parameter] = []
    for name, field in model.model_fields.items():
        default = inspect.Parameter.empty if field.is_required() else field.default
        param = inspect.Parameter(
            name=name,
            kind=inspect.Parameter.KEYWORD_ONLY,
            annotation=field.annotation,
            default=default,
        )
        parameters.append(param)
    return inspect.Signature(parameters=parameters, return_annotation=return_type)
