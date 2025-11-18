from __future__ import annotations

import inspect
from typing import Any, Literal, NotRequired, TypedDict


class PropertyBase(TypedDict, total=False):
    """Base schema property with common fields."""

    description: str
    format: str
    default: Any


class SimpleProperty(PropertyBase):
    """Schema property for primitive types."""

    type: Literal["string", "number", "integer", "boolean"]
    enum: NotRequired[list[Any]]


class ArrayProperty(PropertyBase):
    """Schema property for array types."""

    type: Literal["array"]
    items: Property


class ObjectProperty(PropertyBase):
    """Schema property for nested object types."""

    type: Literal["object"]
    properties: NotRequired[dict[str, Property]]
    required: NotRequired[list[str]]
    additionalProperties: NotRequired[bool]


Property = ArrayProperty | ObjectProperty | SimpleProperty


# Use dict[str, Any] for maximum flexibility with JSON schemas
ToolParameters = dict[str, Any]
"""Schema for tool parameters."""


class OpenAIFunctionDefinition(TypedDict):
    """Schema for the function definition part of an OpenAI tool.

    This represents the inner "function" object that contains the actual
    function metadata and parameters.
    """

    name: str
    description: str
    parameters: ToolParameters


class OpenAIFunctionTool(TypedDict):
    """Complete OpenAI tool definition for function calling.

    This represents the top-level tool object that wraps a function definition
    and identifies it as a function tool type.
    """

    type: Literal["function"]
    function: OpenAIFunctionDefinition


def _create_simple_property(
    type_str: Literal["string", "number", "integer", "boolean"],
    description: str | None = None,
    enum_values: list[Any] | None = None,
    default: Any = None,
    fmt: str | None = None,
) -> SimpleProperty:
    """Create a simple property."""
    prop: SimpleProperty = {"type": type_str}
    if description is not None:
        prop["description"] = description
    if enum_values is not None:
        prop["enum"] = enum_values
    if default is not inspect.Parameter.empty and default is not None:
        prop["default"] = default
    if fmt is not None:
        prop["format"] = fmt
    return prop


def _create_array_property(
    items: Property,
    description: str | None = None,
) -> ArrayProperty:
    """Create an array property."""
    prop: ArrayProperty = {
        "type": "array",
        "items": items,
    }
    if description is not None:
        prop["description"] = description
    return prop


def _create_object_property(
    description: str | None = None,
    properties: dict[str, Property] | None = None,
    required: list[str] | None = None,
    additional_properties: bool | None = None,
) -> ObjectProperty:
    """Create an object property.

    Args:
        description: Optional property description
        properties: Optional dict of property definitions
        required: Optional list of required property names
        additional_properties: Whether to allow additional properties

    Returns:
        Object property definition
    """
    prop: ObjectProperty = {"type": "object"}
    if description is not None:
        prop["description"] = description
    if properties is not None:
        prop["properties"] = properties
    if required is not None:
        prop["required"] = required
    if additional_properties is not None:
        prop["additionalProperties"] = additional_properties
    return prop


def clean_property(
    prop: dict[str, Any],
    description: str | None = None,
) -> Property:
    """Convert complex schema properties to simple OpenAI-compatible types.

    Args:
        prop: Complex property definition
        description: Optional property description

    Returns:
        Simplified Property compatible with OpenAI
    """
    if "anyOf" in prop or "oneOf" in prop:
        types = []
        for subschema in prop.get("anyOf", prop.get("oneOf", [])):
            if isinstance(subschema, dict):
                types.append(subschema.get("type"))  # noqa: PERF401

        # If null is allowed, treat as optional string
        if "null" in types:
            return _create_simple_property(
                type_str="string",
                description=description or prop.get("description"),
                default=None,
            )

        # Get first non-null type or default to string
        first_type = next((t for t in types if t != "null"), "string")
        if first_type not in {"string", "number", "integer", "boolean"}:
            first_type = "string"

        return _create_simple_property(
            type_str=first_type,  # type: ignore # Valid since we checked above
            description=description or prop.get("description"),
            default=prop.get("default"),
        )

    if prop.get("type") == "array":
        items = prop.get("items", {"type": "string"})
        if isinstance(items, dict):
            items = clean_property(items)
        return _create_array_property(
            items=items,
            description=description or prop.get("description"),
        )

    if prop.get("type") == "object":
        sub_props = prop.get("properties", {})
        cleaned_props = {name: clean_property(p) for name, p in sub_props.items()}
        return _create_object_property(
            description=description or prop.get("description"),
            properties=cleaned_props,
            required=prop.get("required"),
        )

    type_str = prop.get("type", "string")
    if type_str not in {"string", "number", "integer", "boolean"}:
        type_str = "string"

    return _create_simple_property(
        type_str=type_str,
        description=description or prop.get("description"),
        enum_values=prop.get("enum"),
        default=prop.get("default"),
        fmt=prop.get("format"),
    )
