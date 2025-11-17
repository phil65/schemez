"""Convert JSON schema to appropriate Python type with validation.

Credits to Marvin / prefect for original code.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING, field, make_dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
import re
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ForwardRef,
    Literal,
    NotRequired,
    Union,
)

from pydantic import (
    AnyUrl,
    EmailStr,
    Field,
    Json,
    StringConstraints,
    model_validator,
)
from typing_extensions import TypedDict


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pydantic import BaseModel


__all__ = ["JSONSchema", "jsonschema_to_type"]


FORMAT_TYPES: dict[str, Any] = {
    "date-time": datetime,
    "email": EmailStr,
    "uri": AnyUrl,
    "json": Json,
}

_classes: dict[tuple[str, Any], type | None] = {}


def jsonschema_to_type(
    schema: Mapping[str, Any],
    name: str | None = None,
) -> type:
    """Convert JSON schema to appropriate Python type with validation.

    Args:
        schema: A JSON Schema dictionary defining the type structure and validation rules
        name: Optional name for object schemas. Only allowed when schema type is "object".
            If not provided for objects, name will be inferred from schema's "title"
            property or default to "Root".

    Returns:
        A Python type (typically a dataclass for objects) with Pydantic validation

    Raises:
        ValueError: If a name is provided for a non-object schema

    Examples:
        Create a dataclass from an object schema:
        ```python
        schema = {
            "type": "object",
            "title": "Person",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "age": {"type": "integer", "minimum": 0},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["name", "age"]
        }

        Person = jsonschema_to_type(schema)
        # Creates a dataclass with name, age, and optional email fields:
        # @dataclass
        # class Person:
        #     name: str
        #     age: int
        #     email: str | None = None
        ```
        Person(name="John", age=30)

        Create a scalar type with constraints:
        ```python
        schema = {
            "type": "string",
            "minLength": 3,
            "pattern": "^[A-Z][a-z]+$"
        }

        NameType = jsonschema_to_type(schema)
        # Creates Annotated[str, StringConstraints(min_length=3, pattern="^[A-Z][a-z]+$")]

        @dataclass
        class Name:
            name: NameType
        ```
    """
    # Always use the top-level schema for references
    if schema.get("type") == "object":
        return create_dataclass(schema, name, schemas=schema)
    if name:
        msg = f"Can not apply name to non-object schema: {name}"
        raise ValueError(msg)
    return schema_to_type(schema, schemas=schema)


def hash_schema(schema: Mapping[str, Any]) -> str:
    """Generate a deterministic hash for schema caching."""
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def resolve_ref(ref: str, schemas: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve JSON Schema reference to target schema."""
    path = ref.replace("#/", "").split("/")
    current = schemas
    for part in path:
        current = current.get(part, {})
    return current


def create_string_type(schema: Mapping[str, Any]) -> type | Annotated[Any, ...]:
    """Create string type with optional constraints."""
    if "const" in schema:
        return Literal[schema["const"]]

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
    base: type[int | float], schema: Mapping[str, Any]
) -> type | Annotated[Any, ...]:
    """Create numeric type with optional constraints."""
    if "const" in schema:
        return Literal[schema["const"]]

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
    return Literal[tuple(values)]  # type: ignore[return-value]


def create_array_type(
    schema: Mapping[str, Any], schemas: Mapping[str, Any]
) -> type | Annotated[Any, ...]:
    """Create list/set type with optional constraints."""
    items = schema.get("items", {})
    if isinstance(items, list):
        # Handle positional item schemas
        item_types = [schema_to_type(s, schemas) for s in items]
        from typing import Union

        combined = Union[tuple(item_types)]  # type: ignore[valid-type] # noqa: UP007
        base = list[combined]  # type: ignore[valid-type]
    else:
        # Handle single item schema
        item_type = schema_to_type(items, schemas)
        base = set if schema.get("uniqueItems") else list  # type: ignore[assignment, misc]
        base = base[item_type]  # type: ignore[valid-type, type-arg, misc]

    constraints = {
        k: v
        for k, v in {
            "min_length": schema.get("minItems"),
            "max_length": schema.get("maxItems"),
        }.items()
        if v is not None
    }

    return Annotated[base, Field(**constraints)] if constraints else base


def _return_any() -> Any:
    return Any


def _get_from_type_handler(
    schema: Mapping[str, Any], schemas: Mapping[str, Any]
) -> Callable[..., Any]:
    """Get the appropriate type handler for the schema."""
    type_handlers: dict[str, Callable[..., Any]] = {  # TODO
        "string": lambda s: create_string_type(s),
        "integer": lambda s: create_numeric_type(int, s),
        "number": lambda s: create_numeric_type(float, s),
        "boolean": lambda _: bool,
        "null": lambda _: type(None),
        "array": lambda s: create_array_type(s, schemas),
        "object": lambda s: create_dataclass(s, s.get("title"), schemas),
    }
    return type_handlers.get(schema.get("type", None), _return_any)


def schema_to_type(  # noqa: PLR0911
    schema: Mapping[str, Any],
    schemas: Mapping[str, Any],
) -> type:
    """Convert schema to appropriate Python type."""
    if not schema:
        return object

    if "type" not in schema and "properties" in schema:
        return create_dataclass(schema, schema.get("title", "<unknown>"), schemas)

    # Handle references first
    if "$ref" in schema:
        ref = schema["$ref"]
        # Handle self-reference
        if ref == "#":
            return ForwardRef(schema.get("title", "Root"))  # type: ignore[return-value]
        return schema_to_type(resolve_ref(ref, schemas), schemas)

    if "const" in schema:
        return Literal[schema["const"]]  # type: ignore[return-value]

    if "enum" in schema:
        return create_enum(f"Enum_{len(_classes)}", schema["enum"])  # type: ignore[return-value]

    schema_type = schema.get("type")
    if not schema_type:
        return Any  # pyright: ignore[reportReturnType]

    if isinstance(schema_type, list):
        # Create a copy of the schema for each type, but keep all constraints
        types: list[type | Any] = []
        for t in schema_type:
            type_schema = dict(schema).copy()
            type_schema["type"] = t
            types.append(schema_to_type(type_schema, schemas))
        has_null = type(None) in types
        types = [t for t in types if t is not type(None)]
        if has_null:
            from typing import Union

            combined = Union[tuple(types)] if len(types) > 1 else types[0]  # noqa: UP007
            return combined | None  # type: ignore[return-value]
        from typing import Union

        return Union[tuple(types)]  # type: ignore  # noqa: UP007

    return _get_from_type_handler(schema, schemas)(schema)  # type: ignore[no-any-return]


def sanitize_name(name: str) -> str:
    """Convert string to valid Python identifier."""
    # Step 1: replace everything except [0-9a-zA-Z_] with underscores
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    # Step 2: deduplicate underscores
    cleaned = re.sub(r"__+", "_", cleaned)
    # Step 3: if the first char of original name isn't a letter, prepend field_
    if not name or not re.match(r"[a-zA-Z]", name[0]):
        cleaned = f"field_{cleaned}"
    # Step 4: deduplicate again and strip trailing underscores
    return re.sub(r"__+", "_", cleaned).strip("_")


def get_default_value(
    schema: dict[str, Any],
    prop_name: str,
    parent_default: dict[str, Any] | None = None,
) -> Any:
    """Get default value with proper priority ordering.

    1. Value from parent's default if it exists
    2. Property's own default if it exists
    3. None.
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
    # Always use None as default for complex types
    if isinstance(default_value, (dict, list)) or default_value is None:
        return field(default=None)

    # For simple types, use the value directly
    return field(default=default_value)


def create_dataclass(
    schema: Mapping[str, Any],
    name: str | None = None,
    schemas: Mapping[str, Any] | None = None,
) -> type:
    """Create dataclass from object schema."""
    name = name or schema.get("title")
    # Sanitize name for class creation
    sanitized_name = sanitize_name(name or "Root")
    schema_hash = hash_schema(schema)
    cache_key = (schema_hash, sanitized_name)
    original_schema = dict(schema)  # Store copy for validator

    # Return existing class if already built
    if cache_key in _classes:
        existing = _classes[cache_key]
        if existing is None:
            return ForwardRef(sanitized_name)  # type: ignore[return-value]
        return existing

    # Place placeholder for recursive references
    _classes[cache_key] = None

    if "$ref" in schema:
        ref = schema["$ref"]
        if ref == "#":
            return ForwardRef(sanitized_name)  # type: ignore[return-value]
        schema = resolve_ref(ref, schemas or {})

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    fields: list[tuple[Any, ...]] = []
    for prop_name, prop_schema in properties.items():
        field_name = sanitize_name(prop_name)

        # Check for self-reference in property
        if prop_schema.get("$ref") == "#":
            field_type = ForwardRef(sanitized_name)
        else:
            field_type = schema_to_type(prop_schema, schemas or {})  # type: ignore[assignment]

        default_val = prop_schema.get("default", MISSING)
        is_required = prop_name in required

        # Include alias in field metadata
        meta = {"alias": prop_name}

        if default_val is not MISSING:
            if isinstance(default_val, (dict, list)):
                field_def = field(
                    default_factory=lambda d=default_val: deepcopy(d),  # type: ignore[misc]
                    metadata=meta,
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
            # Use Union[field_type, None] instead of field_type | None
            # to maintain compatibility with code that expects __origin__
            fields.append((field_name, Union[field_type, None], field_def))  # noqa: UP007

    cls = make_dataclass(sanitized_name, fields, kw_only=True)

    # Add model validator for defaults
    @model_validator(mode="before")  # type: ignore[misc]
    @classmethod
    def _apply_defaults(
        cls: type[BaseModel], data: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if isinstance(data, dict):
            return merge_defaults(data, original_schema)
        return data

    cls._apply_defaults = _apply_defaults  # type: ignore[attr-defined]

    # Store completed class
    _classes[cache_key] = cls
    return cls


def merge_defaults(
    data: Mapping[str, Any],
    schema: Mapping[str, Any],
    parent_default: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge defaults with provided data at all levels."""
    # If we have no data
    if not data:
        # Start with parent default if available
        if parent_default:
            result = dict(parent_default)
        # Otherwise use schema default if available
        elif "default" in schema:
            result = dict(schema["default"])
        # Otherwise start empty
        else:
            result = {}
    # If we have data and a parent default, merge them
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
    # Otherwise just use the data
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


class JSONSchema(TypedDict):
    type: NotRequired[str | list[str]]
    properties: NotRequired[dict[str, JSONSchema]]
    required: NotRequired[list[str]]
    additionalProperties: NotRequired[bool | JSONSchema]
    items: NotRequired[JSONSchema | list[JSONSchema]]
    enum: NotRequired[list[Any]]
    const: NotRequired[Any]
    default: NotRequired[Any]
    description: NotRequired[str]
    title: NotRequired[str]
    examples: NotRequired[list[Any]]
    format: NotRequired[str]
    allOf: NotRequired[list[JSONSchema]]
    anyOf: NotRequired[list[JSONSchema]]
    oneOf: NotRequired[list[JSONSchema]]
    not_: NotRequired[JSONSchema]
    definitions: NotRequired[dict[str, JSONSchema]]
    dependencies: NotRequired[dict[str, JSONSchema | list[str]]]
    pattern: NotRequired[str]
    minLength: NotRequired[int]
    maxLength: NotRequired[int]
    minimum: NotRequired[int | float]
    maximum: NotRequired[int | float]
    exclusiveMinimum: NotRequired[int | float]
    exclusiveMaximum: NotRequired[int | float]
    multipleOf: NotRequired[int | float]
    uniqueItems: NotRequired[bool]
    minItems: NotRequired[int]
    maxItems: NotRequired[int]
    additionalItems: NotRequired[bool | JSONSchema]


if __name__ == "__main__":
    schema = {
        "$id": "https://example.com/person.schema.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Person",
        "type": "object",
        "properties": {
            "firstName": {"type": "string", "description": "The person's first name."},
            "lastName": {"type": "string", "description": "The person's last name."},
            "age": {
                "description": "Age in years, must be equal to or greater than zero.",
                "type": "integer",
                "minimum": 0,
            },
        },
    }
    model = jsonschema_to_type(schema)
    import devtools

    devtools.debug(model)
