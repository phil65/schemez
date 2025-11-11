"""Helpers for BaseModels."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Literal, overload

from pydantic import BaseModel


PythonVersionStr = Literal["3.12", "3.13", "3.14"]


def json_schema_to_pydantic_code(
    schema_source: str | dict,
    *,
    class_name: str = "Model",
    target_python_version: PythonVersionStr | None = None,
    base_class: str = "pydantic.BaseModel",
) -> str:
    """Generate Pydantic model code from a JSON schema using datamodel-codegen.

    Args:
        schema_source: JSON schema as string or dict
        class_name: Name for the generated class
        target_python_version: Python version target (3.12, 3.13, 3.14)
        base_class: Base class for generated model

    Returns:
        Generated Python code string
    """
    import json

    from datamodel_code_generator import DataModelType, LiteralType, PythonVersion
    from datamodel_code_generator.model import get_data_model_types
    from datamodel_code_generator.parser.jsonschema import JsonSchemaParser

    # Convert schema to JSON string if needed
    if isinstance(schema_source, dict):
        source = json.dumps(schema_source)
    else:
        source = str(schema_source)

    # Determine Python version
    match target_python_version:
        case "3.12":
            py = PythonVersion.PY_312
        case "3.13" | None:
            py = PythonVersion.PY_313
        case "3.14":
            py = PythonVersion.PY_314
        case None:
            py = PythonVersion.PY_313

    # Get model types
    model_types = get_data_model_types(
        DataModelType.PydanticV2BaseModel,
        target_python_version=py,
    )

    # Create parser with standard configuration
    parser = JsonSchemaParser(
        source=source,
        data_model_type=model_types.data_model,
        data_model_root_type=model_types.root_model,
        data_model_field_type=model_types.field_model,
        data_type_manager_type=model_types.data_type_manager,
        dump_resolve_reference_action=model_types.dump_resolve_reference_action,
        class_name=class_name,
        base_class=base_class,
        use_union_operator=True,
        use_schema_description=True,
        enum_field_as_literal=LiteralType.All,
    )

    result = parser.parse()
    assert isinstance(result, str)
    return result


@overload
def json_schema_to_pydantic_class[TBaseModel: BaseModel](
    json_schema: str | dict,
    class_name: str = "DynamicModel",
    *,
    base_class: type[TBaseModel],
) -> type[TBaseModel]: ...


@overload
def json_schema_to_pydantic_class(
    json_schema: str | dict,
    class_name: str = "DynamicModel",
    *,
    base_class: str = "pydantic.BaseModel",
) -> type[BaseModel]: ...


def json_schema_to_pydantic_class(
    json_schema: str | dict,
    class_name: str = "DynamicModel",
    *,
    base_class: type[BaseModel] | str = "pydantic.BaseModel",
) -> type[BaseModel]:
    """Create a Pydantic v2 model class from a JSON schema.

    Args:
        json_schema: The JSON schema to create a model from
        class_name: Name for the generated class
        base_class: Base class for the generated model

    Returns:
        A new Pydantic v2 model class based on the JSON schema
    """
    # Generate code and create class dynamically
    if isinstance(base_class, str):
        base_class_str = base_class
        namespace: dict[str, Any] = {}
        original_base_class = None
    else:
        # For class objects, use simple name and add to namespace
        base_class_str = base_class.__name__
        namespace = {base_class.__name__: base_class}
        original_base_class = base_class

    code = json_schema_to_pydantic_code(
        schema_source=json_schema,
        class_name=class_name,
        target_python_version="3.13",
        base_class=base_class_str,
    )

    # Clean up generated code for custom base classes
    if not isinstance(base_class, str):
        # Remove import lines for custom base classes
        lines = code.split("\n")
        cleaned_lines = []
        for line in lines:
            if line.strip().startswith(f"import {base_class.__name__}"):
                continue  # Skip the import line
            cleaned_lines.append(line)
        code = "\n".join(cleaned_lines)

    # First attempt: try with original base class
    try:
        exec(code, namespace, namespace)
    except TypeError as e:
        # If it fails due to use_attribute_docstrings, try with modified base class
        if "built-in class" in str(e) and original_base_class is not None:
            # Check if base class has use_attribute_docstrings=True
            config = getattr(original_base_class, "model_config", None)
            if config and config.get("use_attribute_docstrings", False):
                # Create a subclass that disables use_attribute_docstrings
                from pydantic import ConfigDict

                # Handle both dict and ConfigDict types
                if isinstance(config, dict):
                    config_copy = config.copy()
                    config_copy["use_attribute_docstrings"] = False
                    new_config = ConfigDict(**config_copy)
                else:
                    config_dict = config.__dict__.copy()
                    config_dict["use_attribute_docstrings"] = False
                    new_config = ConfigDict(**config_dict)

                class FallbackBase(original_base_class):  # type: ignore[valid-type]
                    model_config = new_config

                # Update namespace and code with fallback base
                namespace[base_class_str] = FallbackBase
                exec(code, namespace, namespace)
            else:
                raise
        else:
            raise

    # Find the generated model class by name
    model = namespace.get(class_name)
    if model and isinstance(model, type) and issubclass(model, BaseModel):
        model.__module__ = __name__
        return model

    # Fallback: find any BaseModel subclass
    for v in namespace.values():
        if isinstance(v, type) and issubclass(v, BaseModel) and v != BaseModel:
            model = v
            break

    if not model:
        msg = (
            f"Could not find generated model class '{class_name}' "
            f"in: {list(namespace.keys())}"
        )
        raise Exception(msg)  # noqa: TRY002

    model.__module__ = __name__
    return model


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


def model_to_python_code(
    model: type[BaseModel] | dict[str, Any],
    *,
    class_name: str | None = None,
    target_python_version: PythonVersionStr | None = None,
    model_type: str = "pydantic.BaseModel",
) -> str:
    """Convert a BaseModel or schema dict to Python code.

    Args:
        model: The BaseModel class or schema dictionary to convert
        class_name: Optional custom class name for the generated code
        target_python_version: Target Python version for code generation.
            Defaults to current system Python version.
        model_type: Type of the generated model. Defaults to "pydantic.BaseModel".

    Returns:
        Generated Python code as string

    Raises:
        ValueError: If schema parsing fails
    """
    if isinstance(model, dict):
        schema = model
        name = class_name or "GeneratedModel"
    else:
        schema = model.model_json_schema()
        name = class_name or model.__name__

    return json_schema_to_pydantic_code(
        schema_source=schema,
        class_name=name,
        target_python_version=target_python_version,
        base_class=model_type,
    )


if __name__ == "__main__":

    class TestModel(BaseModel):
        test_int: int = 1
        test_str: str = "test"
        test_float: float = 1.1
        test_bool: bool = True

    code = model_to_python_code(TestModel)
    print(code)
