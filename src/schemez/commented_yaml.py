"""Utility functions for schema -> YAML conversion."""

from __future__ import annotations

import re
from typing import Any


def get_description(field_schema: dict[str, Any]) -> str | None:
    """Get first line of field description."""
    desc = field_schema.get("description", "")
    if desc:
        first_line = desc.split("\n")[0].strip()
        return first_line[:100] + "..." if len(first_line) > 100 else first_line  # type: ignore[no-any-return]  # noqa: PLR2004
    return None


def find_schema_for_path(
    schema_obj: dict[str, Any],
    path: list[str],
) -> dict[str, Any] | None:
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


def process_yaml_lines(yaml_lines: list[str], schema: dict[str, Any]) -> list[str]:
    """Add comments to YAML lines based on schema descriptions."""
    result: list[str] = []
    path_stack: list[str] = []

    for line in yaml_lines:
        original_line = line
        stripped = line.lstrip()
        indent_level = (len(line) - len(stripped)) // 2

        # Adjust path stack based on indentation
        while len(path_stack) > indent_level:
            path_stack.pop()

        # Check if this is a field definition
        if ":" in stripped and not stripped.startswith("#") and not stripped.startswith("-"):
            field_match = re.match(r"^([^:]+):\s*(.*)", stripped)
            if field_match:
                field_name, value = field_match.groups()
                field_name = field_name.strip().strip('"').strip("'")

                # Update path stack
                if len(path_stack) == indent_level:
                    path_stack.append(field_name)
                else:
                    path_stack = [*path_stack[:indent_level], field_name]

                # Find schema for this field
                field_schema = find_schema_for_path(schema, path_stack)
                if field_schema:
                    desc = get_description(field_schema)
                    if desc:
                        # Add comment
                        if value.strip() and not value.strip().startswith("|"):
                            result.append(f"{line}  # {desc}")
                        else:
                            result.append(f"{line}  # {desc}")
                        continue

        # Keep original line if no comment added
        result.append(original_line)

    return result
