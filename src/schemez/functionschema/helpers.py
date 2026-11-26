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
from schemez.introspection import get_resolved_hints


if typing.TYPE_CHECKING:
    from pydantic import BaseModel

    from schemez.functionschema.typedefs import Property


logger = log.get_logger(__name__)

FunctionType = Literal["sync", "async", "sync_generator", "async_generator"]


_EMPTY = inspect.Parameter.empty
_JSON_SCALARS = (str, int, float, bool, enum.Enum)


@dataclasses.dataclass
class _Context:
    """State of one schema walk: recursive types end up in `defs` behind a `$ref`."""

    defs: dict[str, Any]
    stack: list[Any] = dataclasses.field(default_factory=list)
    recursive: set[Any] = dataclasses.field(default_factory=set)
    names: dict[Any, str] = dataclasses.field(default_factory=dict)

    def def_name(self, typ: Any) -> str:
        if typ not in self.names:
            base = getattr(typ, "__name__", "Model")
            name, i = base, 2
            while name in self.names.values():
                name, i = f"{base}{i}", i + 1
            self.names[typ] = name
        return self.names[typ]


def resolve_type_annotation(
    typ: Any,
    description: str | None = None,
    default: Any = inspect.Parameter.empty,
    is_parameter: bool = True,
    defs: dict[str, Any] | None = None,
) -> Property:
    """Resolve a type annotation into an OpenAI schema type.

    Args:
        typ: Type to resolve
        description: Optional description
        default: Default value if any
        is_parameter: Whether this is for a parameter (affects dict schema)
        defs: Collects the definitions of recursive types, referenced as
            `#/$defs/<name>`; the caller puts them at the root of its schema.
            Without it, they are added to the returned schema as `$defs`.
    """
    ctx = _Context(defs={} if defs is None else defs)
    schema = _resolve(typ, ctx, is_parameter=is_parameter, optional=default is None)
    if description and (hint := _description_hint(typ)):
        description = f"{description} ({hint})"
    schema = _with_meta(schema, description, default)
    if defs is None and ctx.defs:
        schema["$defs"] = ctx.defs
    return typing.cast("Property", schema)


def _with_meta(schema: dict[str, Any], description: str | None, default: Any) -> dict[str, Any]:
    if description is not None:
        schema["description"] = description
    if default is not _EMPTY and default is not None:
        is_scalar = isinstance(default, _JSON_SCALARS)
        is_container = isinstance(default, list | tuple | dict)
        if (is_scalar and schema.get("type") not in {"array", "object"}) or (
            is_container and schema.get("type") in {"array", "object"}
        ):
            schema["default"] = default.value if isinstance(default, enum.Enum) else default
    return schema


def _description_hint(typ: Any) -> str | None:
    """Format hint appended to the description of string-encoded types."""
    while True:
        if get_origin(typ) is Annotated:
            typ = get_args(typ)[0]
            continue
        if get_origin(typ) in {typing.Union, types.UnionType}:  # pyright: ignore
            members = [t for t in get_args(typ) if t is not type(None)]
            if len(members) == 1:
                typ = members[0]
                continue
        break
    if typ in {datetime, date, time}:
        return "ISO 8601 format"
    hints = {
        timedelta: "ISO 8601 duration",
        timezone: "IANA timezone name",
        bytes: "base64 encoded",
        bytearray: "base64 encoded",
    }
    return hints.get(typ)


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def _enum_schema(values: list[Any]) -> dict[str, Any]:
    """`enum` schema whose `type` matches the values (`anyOf` per type if mixed)."""
    by_type: dict[str, list[Any]] = {}
    for value in values:
        by_type.setdefault(_json_type(value), []).append(value)
    if len(by_type) == 1:
        [(type_str, vals)] = by_type.items()
        return {"type": type_str, "enum": vals}
    return {"anyOf": [{"type": t, "enum": vals} for t, vals in by_type.items()]}


def _resolve_class_fields(typ: Any, ctx: _Context, is_parameter: bool) -> dict[str, Any]:
    """Object schema of a dataclass, TypedDict or pydantic model (recursion via `$ref`)."""
    if typ in ctx.stack:
        ctx.recursive.add(typ)
        return {"$ref": f"#/$defs/{ctx.def_name(typ)}"}
    ctx.stack.append(typ)
    try:
        schema = _object_schema(typ, ctx, is_parameter)
    finally:
        ctx.stack.pop()
    if typ in ctx.recursive:
        name = ctx.def_name(typ)
        ctx.defs[name] = schema
        return {"$ref": f"#/$defs/{name}"}
    return schema


def _object_schema(typ: Any, ctx: _Context, is_parameter: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    if dataclasses.is_dataclass(typ):
        hints = get_resolved_hints(typ)
        for field in dataclasses.fields(typ):
            has_default = field.default is not dataclasses.MISSING
            field_default = field.default if has_default else _EMPTY
            prop = _resolve(
                hints.get(field.name, field.type),
                ctx,
                is_parameter=is_parameter,
                optional=field_default is None,
            )
            properties[field.name] = _with_meta(prop, None, field_default)
            if not has_default and field.default_factory is dataclasses.MISSING:
                required.append(field.name)
    elif typing.is_typeddict(typ):
        hints = get_resolved_hints(typ)
        required_keys = getattr(typ, "__required_keys__", set(hints))
        for field_name, field_type in hints.items():
            origin = get_origin(field_type)
            if origin is Required:
                is_required = True
                field_type = get_args(field_type)[0]
            elif origin is NotRequired:
                is_required = False
                field_type = get_args(field_type)[0]
            else:
                is_required = field_name in required_keys
            properties[field_name] = _resolve(field_type, ctx, is_parameter=is_parameter)
            if is_required:
                required.append(field_name)
    else:  # pydantic model
        for field_name, field_info in typ.model_fields.items():
            field_default = _EMPTY if field_info.is_required() else field_info.default
            prop = _resolve(
                field_info.annotation,
                ctx,
                is_parameter=is_parameter,
                optional=field_default is None,
            )
            properties[field_name] = _with_meta(prop, field_info.description, field_default)
            if field_info.is_required():
                required.append(field_name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _resolve(  # noqa: PLR0911
    typ: Any,
    ctx: _Context,
    *,
    is_parameter: bool,
    optional: bool = False,
) -> dict[str, Any]:
    """JSON schema of a type, without description and default.

    `optional`: the value defaults to None, so a `None` union member is dropped
    (the field can be omitted) instead of becoming `{"type": "null"}`.
    """
    # Handle anyOf/oneOf fields
    if isinstance(typ, dict) and ("anyOf" in typ or "oneOf" in typ):
        # For simplicity, treat it as a string (common pattern for optional fields)
        return {"type": "string"}

    if typ is Any or typ is object:
        return {}
    if typ is None or typ is type(None):
        return {"type": "null"}
    if isinstance(typ, str | typing.ForwardRef):  # unresolvable annotation
        return {}
    if isinstance(typ, typing.TypeAliasType):
        return _resolve(typ.__value__, ctx, is_parameter=is_parameter, optional=optional)
    if isinstance(typ, typing.NewType):
        return _resolve(typ.__supertype__, ctx, is_parameter=is_parameter, optional=optional)

    origin = get_origin(typ)
    args = get_args(typ)

    if origin is Annotated:
        return _resolve(args[0], ctx, is_parameter=is_parameter, optional=optional)

    # Union types (including Optional)
    if origin in {typing.Union, types.UnionType}:  # pyright: ignore
        has_none = type(None) in args
        members = [t for t in args if t is not type(None)]
        if not members:
            return {"type": "null"}
        variants = [_resolve(t, ctx, is_parameter=is_parameter) for t in members]
        if has_none and not optional:
            variants.append({"type": "null"})
        unique: list[dict[str, Any]] = []
        for variant in variants:
            if variant not in unique:
                unique.append(variant)
        if any(v == {} for v in unique):
            return {}
        return unique[0] if len(unique) == 1 else {"anyOf": unique}

    if origin is Literal:
        return _enum_schema([a.value if isinstance(a, enum.Enum) else a for a in args])

    if dataclasses.is_dataclass(typ) or typing.is_typeddict(typ):
        return _resolve_class_fields(typ, ctx, is_parameter)

    # Mappings
    if origin in {dict, typing.Dict} or (  # noqa: UP006
        origin is not None and isinstance(origin, type) and issubclass(origin, dict)
    ):
        schema: dict[str, Any] = {"type": "object"}
        if is_parameter:  # Only add additionalProperties for parameters
            min_dict_args = 2
            if len(args) >= min_dict_args and args[1] is not Any:
                schema["additionalProperties"] = _resolve(args[1], ctx, is_parameter=True)
            else:
                schema["additionalProperties"] = True
        return schema

    # Fixed-length tuples
    if origin is tuple and args and not (len(args) == 2 and args[1] is Ellipsis):  # noqa: PLR2004
        if args == ((),):
            return {"type": "array", "maxItems": 0}
        items = [_resolve(t, ctx, is_parameter=is_parameter) for t in args]
        return {"type": "array", "prefixItems": items, "minItems": len(args), "maxItems": len(args)}

    # Sequences
    if origin in {
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
        item_type = args[0] if args else Any
        return {"type": "array", "items": _resolve(item_type, ctx, is_parameter=is_parameter)}

    if not isinstance(typ, type):
        # Default for unmatched types
        return {"type": "string"}

    if issubclass(typ, enum.Enum):
        return _enum_schema([e.value for e in typ])
    if typ in {list, set, frozenset, tuple}:
        return {"type": "array", "items": {}}
    if typ is dict:
        schema = {"type": "object"}
        if is_parameter:
            schema["additionalProperties"] = True
        return schema
    if typ in {str, Path, UUID, re.Pattern, timedelta, timezone, bytes, bytearray}:
        return {"type": "string"}
    if typ in {ipaddress.IPv4Address, ipaddress.IPv6Address}:
        return {"type": "string"}
    if typ is bool:
        return {"type": "boolean"}
    if typ is int:
        return {"type": "integer"}
    if typ in {float, decimal.Decimal}:
        return {"type": "number"}
    if typ is datetime:
        return {"type": "string", "format": "date-time"}
    if typ is date:
        return {"type": "string", "format": "date"}
    if typ is time:
        return {"type": "string", "format": "time"}
    if typ is complex:
        return {
            "type": "object",
            "properties": {"real": {"type": "number"}, "imag": {"type": "number"}},
        }
    if hasattr(typ, "model_fields"):  # pydantic model
        return _resolve_class_fields(typ, ctx, is_parameter)
    # Default to object for unknown types
    return {"type": "object"}


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
        if exclude_type_name in annotation_str or exclude_type_full_name in annotation_str:
            # Be more specific to avoid false positives
            # Check if it's the exact type name, not just a substring
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
