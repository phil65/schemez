"""Helpers for BaseModels."""

from __future__ import annotations

from typing import Any, overload

from pydantic import BaseModel


@overload
def json_schema_to_pydantic_class[TBaseModel: BaseModel](
    json_schema: str | dict[str, Any],
    class_name: str = "DynamicModel",
    *,
    base_class: type[TBaseModel],
) -> type[TBaseModel]: ...


@overload
def json_schema_to_pydantic_class(
    json_schema: str | dict[str, Any],
    class_name: str = "DynamicModel",
    *,
    base_class: str = "pydantic.BaseModel",
) -> type[BaseModel]: ...


def json_schema_to_pydantic_class(
    json_schema: str | dict[str, Any],
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
    from schemez.helpers import json_schema_to_pydantic_code

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
        json_schema,
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
            config = original_base_class.model_config
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

                class FallbackBase(original_base_class):  # type: ignore[valid-type, misc]
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
        return model  # type: ignore[no-any-return]

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
    model = json_schema_to_pydantic_class(schema)
    print(model)
