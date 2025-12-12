"""Utility functions for schema -> YAML conversion."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from pydantic import BaseModel


def get_description(field_schema: dict[str, Any]) -> str | None:
    """Get first line of field description."""
    desc = field_schema.get("description", "")
    if desc:
        first_line = desc.split("\n")[0].strip()
        return first_line[:100] + "..." if len(first_line) > 100 else first_line  # type: ignore[no-any-return]  # noqa: PLR2004
    return None


def find_schema_for_path(schema_obj: dict[str, Any], path: list[str]) -> dict[str, Any] | None:
    """Navigate schema to find definition for nested path."""
    current = schema_obj

    def resolve_ref(schema_part: dict[str, Any]) -> dict[str, Any] | None:
        """Resolve a $ref reference."""
        if "$ref" in schema_part:
            ref_path = schema_part["$ref"].replace("#/$defs/", "")
            if "$defs" in schema_obj and ref_path in schema_obj["$defs"]:
                return schema_obj["$defs"][ref_path]  # type: ignore[no-any-return]
        return None

    def resolve_anyof_oneof(schema_part: dict[str, Any]) -> dict[str, Any] | None:
        """Resolve anyOf/oneOf.

        Resolves  by finding the first non-null object type with $ref.
        """
        for union_key in ["anyOf", "oneOf"]:
            if union_key in schema_part:
                for option in schema_part[union_key]:
                    if "$ref" in option:
                        resolved = resolve_ref(option)
                        if resolved:
                            return resolved
                    elif option.get("type") == "object" and "properties" in option:
                        return option  # type: ignore[no-any-return]
        return None

    for i, segment in enumerate(path):
        # Handle properties
        if "properties" in current and segment in current["properties"]:
            current = current["properties"][segment]

            # If this is the last segment, return the field directly
            if i == len(path) - 1:
                return current  # type: ignore[no-any-return]

            # For non-last segments, we need to resolve to continue navigation
            # First try direct $ref
            resolved = resolve_ref(current)
            if resolved:
                current = resolved
                continue

            # Then try anyOf/oneOf
            resolved = resolve_anyof_oneof(current)
            if resolved:
                current = resolved
                continue

            # If no resolution possible, we can't continue
            return None

        # Handle array items
        if "items" in current:
            current = current["items"]
            resolved = resolve_ref(current)
            if resolved:
                current = resolved

        # Handle additionalProperties
        elif "additionalProperties" in current and isinstance(
            current["additionalProperties"], dict
        ):
            current = current["additionalProperties"]
            resolved = resolve_ref(current)
            if resolved:
                current = resolved
        else:
            return None
    return current


def create_yaml_description(
    model: type[BaseModel],
    exclude_none: bool = True,
    exclude_defaults: bool = False,
    exclude_unset: bool = False,
    indent: int = 2,
    default_flow_style: bool | None = None,
    allow_unicode: bool = True,
    comments: bool = False,
    sort_keys: bool = True,
    validate: bool = False,
    mode: Literal["json", "python"] = "python",
    expand_mode: Literal["minimal", "maximal", "default"] = "default",
) -> str:
    """Dump configuration to YAML string.

    Args:
        model: Model to dump
        exclude_none: Exclude fields with None values
        exclude_defaults: Exclude fields with default values
        exclude_unset: Exclude fields that are not set
        indent: Indentation level for YAML output
        default_flow_style: Default flow style for YAML output
        allow_unicode: Allow unicode characters in YAML output
        comments: Include descriptions as comments in the YAML output
        sort_keys: Sort keys in the YAML output
        mode: Output mode, either "json" or "python"
        validate: Validate the generated YAML against the JSON schema
        expand_mode: Expand mode, either "minimal", "maximal", or "default"

    Returns:
        YAML string representation of the model
    """
    import yamling

    from schemez.commented_yaml import process_yaml_lines
    from schemez.generators import SchemaDataGenerator

    json_schema = model.model_json_schema()
    generator = SchemaDataGenerator(json_schema)

    if expand_mode == "minimal":
        data = generator.generate_minimal()
    elif expand_mode == "maximal":
        data = generator.generate_maximal()
    else:  # default
        data = generator.generate()

    if validate:
        instance = model.model_validate(data)
    instance = model.model_construct(**data)
    data = instance.model_dump(
        exclude_none=exclude_none,
        exclude_defaults=exclude_defaults,
        exclude_unset=exclude_unset,
        mode=mode,
    )
    base_yaml = yamling.dump_yaml(
        data,
        sort_keys=sort_keys,
        indent=indent,
        default_flow_style=default_flow_style,
        allow_unicode=allow_unicode,
    )
    if not comments:
        return base_yaml

    schema = model.model_json_schema()
    yaml_lines = base_yaml.strip().split("\n")
    commented_lines = process_yaml_lines(yaml_lines, schema, as_listitem=False, wrapped_in=None)

    return "\n".join(commented_lines)


def process_yaml_lines(
    yaml_lines: list[str],
    schema: dict[str, Any],
    as_listitem: bool = True,
    wrapped_in: str | None = None,
) -> list[str]:
    """Add comments to YAML lines based on schema descriptions."""
    result: list[str] = []
    path_stack: list[str] = []
    in_multiline_string = False
    multiline_quote_char = None

    # Determine the base indent level for root model fields based on wrapping options
    base_indent = 0
    if wrapped_in:
        base_indent += 1
    if as_listitem:
        base_indent += 1

    for line in yaml_lines:
        original_line = line
        stripped = line.lstrip()
        indent_level = (len(line) - len(stripped)) // 2

        # Check if we're entering or exiting a multi-line quoted string
        if not in_multiline_string:
            # Check if line starts a multi-line quoted string (ends with opening quote)
            if ":" in stripped and not stripped.startswith("#"):
                field_match = re.match(r"^([^:]+):\s*(.*)$", stripped)
                if field_match:
                    _field_name, value_part = field_match.groups()
                    # Check if value starts with a quote but doesn't end with matching quote
                    if value_part.startswith("'") and not (
                        len(value_part) > 1 and value_part.endswith("'")
                    ):
                        in_multiline_string = True
                        multiline_quote_char = "'"
                    elif value_part.startswith('"') and not (
                        len(value_part) > 1 and value_part.endswith('"')
                    ):
                        in_multiline_string = True
                        multiline_quote_char = '"'
        else:
            # Check if this line ends the multi-line string
            if multiline_quote_char and stripped.endswith(multiline_quote_char):
                in_multiline_string = False
                multiline_quote_char = None
            result.append(original_line)
            continue

        # Check if this is a list item start
        if stripped.startswith("- "):
            result.append(original_line)
            continue

        # Adjust path stack based on indentation relative to base
        effective_indent = indent_level - base_indent
        while len(path_stack) > effective_indent and path_stack:
            path_stack.pop()

        # Check if this is a field definition
        if ":" in stripped and not stripped.startswith("#"):
            field_match = re.match(r"^([^:]+):\s*(.*)", stripped)
            if field_match:
                field_name, _value = field_match.groups()
                field_name = field_name.strip().strip('"').strip("'")

                # Update path stack
                if len(path_stack) == effective_indent:
                    path_stack.append(field_name)
                else:
                    path_stack = [*path_stack[:effective_indent], field_name]

                # Find schema for this field and add comment if description exists
                field_schema = find_schema_for_path(schema, path_stack)
                if field_schema:
                    desc = get_description(field_schema)
                    if desc:
                        result.append(f"{line}  # {desc}")
                        continue

        # Keep original line if no comment added
        result.append(original_line)

    return result
