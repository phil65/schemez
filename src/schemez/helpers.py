"""Helpers for BaseModels."""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel


StrPath = str | os.PathLike[str]


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
