"""Module containing code generation utilities."""

from __future__ import annotations

import asyncio


def json_schema_to_python_type(param_info: dict) -> str:
    """Convert JSON schema parameter to Python type annotation."""
    param_type = param_info.get("type", "Any")

    type_mapping = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }

    return type_mapping.get(param_type, "Any")


def clean_generated_code(code: str) -> str:
    """Clean up generated code by removing redundant imports and headers."""
    lines = code.split("\n")
    cleaned_lines = []
    skip_until_class = True

    for line in lines:
        # Skip lines until we find a class or other meaningful content
        if skip_until_class:
            if line.strip().startswith("class ") or (
                line.strip()
                and not line.startswith("#")
                and not line.startswith("from __future__")
                and not line.startswith("from pydantic import")
                and not line.startswith("from typing import")
                and not line.startswith("from datetime import")
            ):
                skip_until_class = False
                cleaned_lines.append(line)
            continue
        # Skip redundant imports that are already in the header
        if (
            line.strip().startswith("from __future__")
            or line.strip().startswith("from pydantic import")
            or line.strip().startswith("from typing import")
            or line.strip().startswith("from datetime import")
        ):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def generate_tool_stub(generator) -> str:
    """Generate a tool stub when source code isn't available."""
    from schemez.helpers import json_schema_to_pydantic_code

    func_name = generator.name
    schema = generator.schema

    # Generate parameter type hints from schema
    params = []
    if schema and schema.parameters:
        # Use schemez to convert JSON schema to Python types
        try:
            param_code = json_schema_to_pydantic_code(
                schema.parameters,
                class_name=f"{func_name.title()}Params",
            )
            # Extract field definitions from generated code
            import re

            field_matches = re.findall(r"(\w+):\s*([^=\n]+)", param_code)
            for field_name, field_type in field_matches:
                if field_name not in ("model_config", "__doc__"):
                    # Clean up type annotation
                    field_type = field_type.strip().rstrip(",")
                    params.append(f"{field_name}: {field_type}")
        except Exception:  # noqa: BLE001
            # Fallback to simple type mapping
            for param_name, param_info in schema.parameters.get("properties", {}).items():
                param_type = json_schema_to_python_type(param_info)
                params.append(f"{param_name}: {param_type}")

    param_str = ", ".join(params) if params else ""

    # Determine if function should be async
    is_async = (
        asyncio.iscoroutinefunction(generator.callable) if generator.callable else True
    )
    async_keyword = "async " if is_async else ""

    description = (
        schema.description or f"Tool: {func_name}" if schema else f"Tool: {func_name}"
    )

    return f'''{async_keyword}def {func_name}({param_str}) -> Any:
"""
{description}

This is a generated stub. Original implementation not available.
"""
# TODO: Implement actual tool logic here
# This could call external APIs, MCP servers, etc.
msg = f"Tool {func_name} implementation needed"
raise NotImplementedError(msg)'''


def generate_tool_module(generator) -> str:
    """Generate a complete Python module for a single tool."""
    from schemez.code_generation.tool_code_generator import ToolCodeGenerator

    if not isinstance(generator, ToolCodeGenerator):
        msg = f"Expected ToolCodeGenerator, got {type(generator)}"
        raise ValueError(msg)  # noqa: TRY004

    if not generator.callable:
        msg = f"Generator {generator.name} has no callable"
        raise ValueError(msg)

    # Extract function source or recreate it
    import inspect
    import textwrap

    try:
        # Try to get source code
        source = inspect.getsource(generator.callable)
        # Clean up indentation
        source = textwrap.dedent(source)
    except (OSError, TypeError):
        # If we can't get source, create a stub
        source = generate_tool_stub(generator)

    # Build module content
    module_parts = [
        '"""Generated tool module."""',
        "",
        "import asyncio",
        "from typing import Any",
        "",
    ]

    # Add imports if the function needs them
    if "httpx" in source or "requests" in source:
        module_parts.extend(["import httpx", ""])

    # Add the function
    module_parts.append(source)

    return "\n".join(module_parts)
