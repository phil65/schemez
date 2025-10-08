"""Convert JSON schema to appropriate Python type with validation.

Credits to Marvin / prefect for original code.
"""

from copy import deepcopy
from dataclasses import MISSING, field, make_dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
import re
from typing import Annotated, Any, ForwardRef, Literal, Optional, Union

from pydantic import AnyUrl, EmailStr, Field, Json, StringConstraints, model_validator


__all__ = ["jsonschema_to_type", "merge_defaults"]


FORMAT_TYPES = {"date-time": datetime, "email": EmailStr, "uri": AnyUrl, "json": Json}

_classes: dict[str | tuple[str, ...], Any] = {}


def jsonschema_to_type(
    schema: dict[str, Any], name: str | None = None
) -> type | ForwardRef | Enum:
    # Always use the top-level schema for references
    if schema.get("type") == "object":
        return _create_dataclass(schema, name, schemas=schema)
    if name:
        msg = f"Can not apply name to non-object schema: {name}"
        raise ValueError(msg)
    return schema_to_type(schema, schemas=schema)


def _hash_schema(schema: dict[str, Any]) -> str:
    """Generate a deterministic hash for schema caching."""
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def resolve_ref(ref: str, schemas: dict[str, Any]) -> dict[str, Any]:
    """Resolve JSON Schema reference to target schema."""
    path = ref.replace("#/", "").split("/")
    current = schemas
    for part in path:
        current = current.get(part, {})
    return current


def create_string_type(schema: dict[str, Any]) -> type | Annotated:  # type: ignore
    """Create string type with optional constraints."""
    if "const" in schema:
        return Literal[schema["const"]]  # type: ignore

    if fmt := schema.get("format"):
        if fmt == "uri":
            return AnyUrl
        if fmt == "uri-reference":
            return str
        return FORMAT_TYPES.get(fmt, str)

    constraints = {
        k: v
        for k, v in {
            "min_length": schema.get("minLength"),
            "max_length": schema.get("maxLength"),
            "pattern": schema.get("pattern"),
        }.items()
        if v is not None
    }

    return Annotated[str, StringConstraints(**constraints)] if constraints else str


def create_numeric_type(
    base: type[int | float], schema: dict[str, Any]
) -> type | Annotated:  # type: ignore
    """Create numeric type with optional constraints."""
    if "const" in schema:
        return Literal[schema["const"]]  # type: ignore

    constraints = {
        k: v
        for k, v in {
            "gt": schema.get("exclusiveMinimum"),
            "ge": schema.get("minimum"),
            "lt": schema.get("exclusiveMaximum"),
            "le": schema.get("maximum"),
            "multiple_of": schema.get("multipleOf"),
        }.items()
        if v is not None
    }

    return Annotated[base, Field(**constraints)] if constraints else base


def create_enum(name: str, values: list[Any]) -> type | Enum:
    """Create enum type from list of values."""
    if all(isinstance(v, str) for v in values):
        return Enum(name, {v.upper(): v for v in values})
    return Literal[tuple(values)]  # type: ignore


def create_array_type(
    schema: dict[str, Any], schemas: dict[str, Any]
) -> type | Annotated:  # type: ignore
    """Create list/set type with optional constraints."""
    items = schema.get("items", {})
    if isinstance(items, list):
        # Handle positional item schemas
        item_types = [schema_to_type(s, schemas) for s in items]
        combined = Union[tuple(item_types)]  # type: ignore # noqa: UP007
        base = list[combined]  # type: ignore
    else:
        # Handle single item schema
        item_type = schema_to_type(items, schemas)
        base = set if schema.get("uniqueItems") else list  # type: ignore
        base = base[item_type]  # type: ignore

    constraints = {
        k: v
        for k, v in {
            "min_length": schema.get("minItems"),
            "max_length": schema.get("maxItems"),
        }.items()
        if v is not None
    }

    return Annotated[base, Field(**constraints)] if constraints else base


def schema_to_type(  # noqa: PLR0911
    schema: dict[str, Any], schemas: dict[str, Any]
) -> type | ForwardRef | Enum:
    """Convert schema to appropriate Python type."""
    if not schema:
        return object
    if "type" not in schema and "properties" in schema:
        return _create_dataclass(schema, schema.get("title"), schemas)

    # Handle references first
    if "$ref" in schema:
        ref = schema["$ref"]
        # Handle self-reference
        if ref == "#":
            return ForwardRef(schema.get("title", "Root"))
        return schema_to_type(resolve_ref(ref, schemas), schemas)

    if "const" in schema:
        return Literal[schema["const"]]  # type: ignore

    if "enum" in schema:
        return create_enum(f"Enum_{len(_classes)}", schema["enum"])

    schema_type = schema.get("type")
    if not schema_type:
        return Any  # type: ignore

    if isinstance(schema_type, list):
        # Create a copy of the schema for each type, but keep all constraints
        types = []
        for t in schema_type:
            type_schema = schema.copy()
            type_schema["type"] = t
            types.append(schema_to_type(type_schema, schemas))
        has_null = type(None) in types
        types = [t for t in types if t is not type(None)]
        if has_null:
            return Optional[tuple(types) if len(types) > 1 else types[0]]  # type: ignore  # noqa: UP045
        return Union[tuple(types)]  # type: ignore  # noqa: UP007

    type_handlers = {
        "string": lambda s: create_string_type(s),
        "integer": lambda s: create_numeric_type(int, s),
        "number": lambda s: create_numeric_type(float, s),
        "boolean": lambda _: bool,
        "null": lambda _: type(None),
        "array": lambda s: create_array_type(s, schemas),
        "object": lambda s: _create_dataclass(s, s.get("title"), schemas),
    }

    return type_handlers.get(schema_type, lambda _: Any)(schema)


def sanitize_name(name: str) -> str:
    """Convert string to valid Python identifier."""
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    cleaned = re.sub(r"__+", "_", cleaned)
    cleaned = cleaned.lower()
    if not name or not re.match(r"[a-zA-Z]", name[0]):
        cleaned = f"field_{cleaned}"
    return re.sub(r"__+", "_", cleaned).strip("_")


def get_default_value(
    schema: dict[str, Any], prop_name: str, parent_default: dict[str, Any] | None = None
) -> Any:
    """Get default value with proper priority ordering.

    1. Value from parent's default if it exists
    2. Property's own default if it exists
    3. None
    """
    if parent_default is not None and prop_name in parent_default:
        return parent_default[prop_name]
    return schema.get("default")


def create_field_with_default(
    field_type: type,
    default_value: Any,
    schema: dict[str, Any],
) -> Any:
    """Create a field with simplified default handling."""
    if isinstance(default_value, dict | list) or default_value is None:
        return field(default=None)
    return field(default=default_value)


def _create_dataclass(
    schema: dict[str, Any], name: str | None = None, schemas: dict[str, Any] | None = None
) -> type | ForwardRef:
    """Create dataclass from object schema."""
    name = name or schema.get("title", "Root")
    assert name
    schema_hash = _hash_schema(schema)
    cache_key = (schema_hash, name)
    original_schema = schema.copy()  # Store copy for validator
    if cache_key in _classes:
        existing = _classes[cache_key]
        if existing is None:
            return ForwardRef(name)
        return existing
    _classes[cache_key] = None
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref == "#":
            return ForwardRef(name)
        schema = resolve_ref(ref, schemas or {})
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    fields = []
    for prop_name, prop_schema in properties.items():
        field_name = sanitize_name(prop_name)
        if prop_schema.get("$ref") == "#":
            field_type: type | ForwardRef | Enum = ForwardRef(name)
        else:
            assert schemas
            field_type = schema_to_type(prop_schema, schemas)
        default_val = prop_schema.get("default", MISSING)
        is_required = prop_name in required
        meta = {"alias": prop_name}
        if default_val is not MISSING:
            if isinstance(default_val, dict | list):
                field_def = field(
                    default_factory=lambda d=default_val: deepcopy(d), metadata=meta
                )
            else:
                field_def = field(default=default_val, metadata=meta)
        elif is_required:
            field_def = field(metadata=meta)
        else:
            field_def = field(default=None, metadata=meta)

        if (is_required and default_val is not MISSING) or is_required:
            fields.append((field_name, field_type, field_def))
        else:
            fields.append((field_name, Optional[field_type], field_def))  # type: ignore  # noqa: UP045

    cls = make_dataclass(name, fields, kw_only=True)

    @model_validator(mode="before")
    @classmethod
    def _apply_defaults(cls, data):
        if isinstance(data, dict):
            return merge_defaults(data, original_schema)
        return data

    cls._apply_defaults = _apply_defaults  # type: ignore
    _classes[cache_key] = cls
    return cls


def merge_defaults(
    data: dict[str, Any],
    schema: dict[str, Any],
    parent_default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge defaults with provided data at all levels."""
    if not data:
        if parent_default:
            result = dict(parent_default)
        elif "default" in schema:
            result = dict(schema["default"])
        else:
            result = {}
    elif parent_default:
        result = dict(parent_default)
        for key, value in data.items():
            if (
                isinstance(value, dict)
                and key in result
                and isinstance(result[key], dict)
            ):
                # recursively merge nested dicts
                result[key] = merge_defaults(value, {"properties": {}}, result[key])
            else:
                result[key] = value
    else:
        result = dict(data)

    # For each property in the schema
    for prop_name, prop_schema in schema.get("properties", {}).items():
        # If property is missing, apply defaults in priority order
        if prop_name not in result:
            if parent_default and prop_name in parent_default:
                result[prop_name] = parent_default[prop_name]
            elif "default" in prop_schema:
                result[prop_name] = prop_schema["default"]

        # If property exists and is an object, recursively merge
        if (
            prop_name in result
            and isinstance(result[prop_name], dict)
            and prop_schema.get("type") == "object"
        ):
            # Get the appropriate default for this nested object
            nested_default = None
            if parent_default and prop_name in parent_default:
                nested_default = parent_default[prop_name]
            elif "default" in prop_schema:
                nested_default = prop_schema["default"]

            result[prop_name] = merge_defaults(
                result[prop_name], prop_schema, nested_default
            )

    return result
