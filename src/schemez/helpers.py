"""Helpers for BaseModels."""

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel
from pydantic_core import to_json


StrPath = str | os.PathLike[str]
PythonVersion = Literal["3.13", "3.14", "3.15"]

if TYPE_CHECKING:
    from collections.abc import Callable


def import_callable(path: str) -> Callable[..., Any]:
    """Import a callable from a dotted path.

    Supports both dot and colon notation:
    - Dot notation: module.submodule.Class.method
    - Colon notation: module.submodule:Class.method

    Args:
        path: Import path using dots and/or colon

    Raises:
        ValueError: If path cannot be imported or result isn't callable
    """
    if not path:
        msg = "Import path cannot be empty"
        raise ValueError(msg)

    # Normalize path - replace colon with dot if present
    normalized_path = path.replace(":", ".")
    parts = normalized_path.split(".")

    # Try importing progressively smaller module paths
    for i in range(len(parts), 0, -1):
        try:
            # Try current module path
            module_path = ".".join(parts[:i])
            module = importlib.import_module(module_path)

            # Walk remaining parts as attributes
            obj = module
            for part in parts[i:]:
                obj = getattr(obj, part)

            # Check if we got a callable
            if callable(obj):
                return obj

            msg = f"Found object at {path} but it isn't callable"
            raise ValueError(msg)

        except ImportError:
            # Try next shorter path
            continue
        except AttributeError:
            # Attribute not found - try next shorter path
            continue

    # If we get here, no import combination worked
    msg = f"Could not import callable from path: {path}"
    raise ValueError(msg)


def import_class(path: str) -> type:
    """Import a class from a dotted path.

    Args:
        path: Dot-separated path to the class

    Returns:
        The imported class

    Raises:
        ValueError: If path is invalid or doesn't point to a class
    """
    try:
        obj = import_callable(path)
        if not isinstance(obj, type):
            msg = f"{path} is not a class"
            raise TypeError(msg)  # noqa: TRY301
    except Exception as exc:
        msg = f"Failed to import class from {path}"
        raise ValueError(msg) from exc
    else:
        return obj


def merge_models[T: BaseModel](base: T, overlay: T) -> T:
    """Deep merge two Pydantic models."""
    if not isinstance(overlay, type(base)):
        msg = f"Cannot merge different types: {type(base)} and {type(overlay)}"
        raise TypeError(msg)

    merged_data = base.model_dump()
    overlay_data = overlay.model_dump(exclude_none=True)
    for field_name, field_value in overlay_data.items():
        base_value = merged_data.get(field_name)

        match (base_value, field_value):
            case (list(), list()):
                merged_data[field_name] = [
                    *base_value,
                    *(item for item in field_value if item not in base_value),
                ]
            case (dict(), dict()):
                merged_data[field_name] = base_value | field_value
            case _:
                merged_data[field_name] = field_value

    return base.__class__.model_validate(merged_data)


def resolve_type_string(type_string: str, safe: bool = True) -> type:
    """Convert a string representation to an actual Python type.

    Args:
        type_string: String representation of a type (e.g. "list[str]", "int")
        safe: If True, uses a limited set of allowed types. If False, allows any valid
              Python type expression but has potential security implications
              if input is untrusted

    Returns:
        The corresponding Python type

    Raises:
        ValueError: If the type string cannot be resolved
    """
    if safe:
        # Create a safe context with just the allowed types
        type_context = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "Any": Any,
            # Add other safe types as needed
        }

        try:
            return eval(type_string, {"__builtins__": {}}, type_context)
        except Exception as e:
            msg = f"Failed to resolve type {type_string} in safe mode"
            raise ValueError(msg) from e
    else:  # unsafe mode
        # Import common typing modules to make them available
        import collections.abc
        import typing

        # Create a context with full typing module available
        type_context = {
            **vars(typing),
            **vars(collections.abc),
            **{t.__name__: t for t in __builtins__.values() if isinstance(t, type)},  # type: ignore
        }

        try:
            return eval(type_string, {"__builtins__": {}}, type_context)
        except Exception as e:
            msg = f"Failed to resolve type {type_string} in unsafe mode"
            raise ValueError(msg) from e


async def _detect_command(command: str, *, test_flag: str = "--version") -> list[str]:
    """Detect the correct command prefix for running a command.

    Tries 'uv run' first, then falls back to direct execution.

    Args:
        command: The command to detect
        test_flag: Flag to test command availability with

    Returns:
        Command prefix list (empty for direct execution)

    Raises:
        RuntimeError: If command is not available
    """
    cmd_prefixes = [["uv", "run"], []]

    for prefix in cmd_prefixes:
        try:
            proc = await asyncio.create_subprocess_exec(
                *prefix,
                command,
                test_flag,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode == 0:
                return prefix
        except FileNotFoundError:
            continue

    msg = f"{command} not available (tried both 'uv run' and direct execution)"
    raise RuntimeError(msg)


async def model_to_python_code(
    model: type[BaseModel],
    *,
    class_name: str | None = None,
    target_python_version: PythonVersion | None = None,
    model_type: str = "pydantic.BaseModel",
) -> str:
    """Convert a BaseModel to Python code asynchronously.

    Args:
        model: The BaseModel class to convert
        class_name: Optional custom class name for the generated code
        target_python_version: Target Python version for code generation.
            Defaults to current system Python version.
        model_type: Type of the generated model. Defaults to "pydantic.BaseModel".

    Returns:
        Generated Python code as string

    Raises:
        RuntimeError: If datamodel-codegen is not available
        subprocess.CalledProcessError: If code generation fails
    """
    working_prefix = await _detect_command("datamodel-codegen")

    schema = model.model_json_schema()
    name = class_name or model.__name__
    py = target_python_version or f"{sys.version_info.major}.{sys.version_info.minor}"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        # Use pydantic_core.to_json for proper schema serialization
        schema_json = to_json(schema, indent=2).decode()
        f.write(schema_json)
        schema_file = Path(f.name)

    args = [
        "--input",
        str(schema_file),
        "--input-file-type",
        "jsonschema",
        "--output-model-type",
        model_type,
        "--class-name",
        name,
        "--disable-timestamp",
        "--use-union-operator",
        "--use-schema-description",
        "--enum-field-as-literal",
        "all",
        "--target-python-version",
        py,
    ]

    try:  # Generate model using datamodel-codegen
        proc = await asyncio.create_subprocess_exec(
            *working_prefix,
            "datamodel-codegen",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _stderr = await proc.communicate()

        if proc.returncode != 0:
            code = proc.returncode or -1
            raise subprocess.CalledProcessError(code, "datamodel-codegen")

        return stdout.decode().strip()

    finally:  # Cleanup temp file
        schema_file.unlink(missing_ok=True)
