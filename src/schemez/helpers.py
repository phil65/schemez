"""Helpers for BaseModels."""

from __future__ import annotations

import collections.abc
import importlib
import types
import typing
from typing import TYPE_CHECKING, Any, Literal, assert_never, get_args, get_origin

from pydantic import BaseModel


PythonVersionStr = Literal["3.12", "3.13", "3.14"]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def json_schema_to_pydantic_code(
    schema: str | dict[str, Any],
    *,
    class_name: str = "Model",
    target_python_version: PythonVersionStr | None = None,
    base_class: str = "pydantic.BaseModel",
) -> str:
    """Generate Pydantic model code from a JSON schema using datamodel-codegen.

    Args:
        schema: JSON schema as string or dict
        class_name: Name for the generated class
        target_python_version: Python version target (3.12, 3.13, 3.14)
        base_class: Base class for generated model

    Returns:
        Generated Python code string
    """
    from datamodel_code_generator import DataModelType, LiteralType, PythonVersion
    from datamodel_code_generator.model import get_data_model_types
    from datamodel_code_generator.parser.jsonschema import JsonSchemaParser
    from pydantic_core import to_json

    source = to_json(schema).decode() if isinstance(schema, dict) else str(schema)
    match target_python_version:
        case "3.12":
            py = PythonVersion.PY_312
        case "3.13" | None:
            py = PythonVersion.PY_313
        case "3.14":
            py = PythonVersion.PY_314
        case _ as unreachable:
            assert_never(unreachable)

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
        use_standard_collections=True,  # Use list/dict instead of List/Dict
        enum_field_as_literal=LiteralType.All,
    )

    result = parser.parse()
    assert isinstance(result, str)
    return result


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
            return eval(type_string, {"__builtins__": {}}, type_context)  # type: ignore[no-any-return]
        except Exception as e:
            msg = f"Failed to resolve type {type_string} in safe mode"
            raise ValueError(msg) from e
    else:  # unsafe mode
        # Import common typing modules to make them available

        # Create a context with full typing module available
        type_context = {
            **vars(typing),
            **vars(collections.abc),
            **{t.__name__: t for t in __builtins__.values() if isinstance(t, type)},  # type: ignore[attr-defined]
        }

        try:
            return eval(type_string, {"__builtins__": {}}, type_context)  # type: ignore[no-any-return]
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
        schema,
        class_name=name,
        target_python_version=target_python_version,
        base_class=model_type,
    )


def openapi_to_code(input_path: str, class_name: str | None = None) -> str:
    """Generate Pydantic model code from an OpenAPI specification URL."""
    from datamodel_code_generator import DataModelType, LiteralType, PythonVersion
    from datamodel_code_generator.model import get_data_model_types
    from datamodel_code_generator.parser.openapi import OpenAPIParser

    data_model_types = get_data_model_types(
        DataModelType.PydanticV2BaseModel,
        target_python_version=PythonVersion.PY_312,
    )
    parser = OpenAPIParser(
        source=input_path,
        data_model_type=data_model_types.data_model,
        data_model_root_type=data_model_types.root_model,
        data_model_field_type=data_model_types.field_model,
        data_type_manager_type=data_model_types.data_type_manager,
        dump_resolve_reference_action=data_model_types.dump_resolve_reference_action,
        class_name=class_name,
        use_union_operator=True,
        use_schema_description=True,
        enum_field_as_literal=LiteralType.All,
    )
    return str(parser.parse())


def jsonschema_to_code(
    schema_json: str,
    class_name: str | None = None,
    input_path: str | None = None,
) -> str:
    """Generate Pydantic model code from a JSON schema."""
    from datamodel_code_generator import DataModelType, LiteralType, PythonVersion
    from datamodel_code_generator.model import get_data_model_types
    from datamodel_code_generator.parser.jsonschema import JsonSchemaParser

    data_model_types = get_data_model_types(
        DataModelType.PydanticV2BaseModel,
        target_python_version=PythonVersion.PY_312,
    )
    parser = JsonSchemaParser(
        source=schema_json,
        data_model_type=data_model_types.data_model,
        data_model_root_type=data_model_types.root_model,
        data_model_field_type=data_model_types.field_model,
        data_type_manager_type=data_model_types.data_type_manager,
        dump_resolve_reference_action=data_model_types.dump_resolve_reference_action,
        class_name=class_name or (input_path or "DefaultClass").split(".")[-1],
        use_union_operator=True,
        use_schema_description=True,
        enum_field_as_literal=LiteralType.All,
    )
    return str(parser.parse())


def get_object_name(fn: Callable[..., Any] | types.ModuleType, fallback: str = "<unknown>") -> str:
    """Get the name of a function."""
    name = getattr(fn, "__name__", None)
    if name is None:
        return fallback
    assert isinstance(name, str)
    return name


def get_object_qualname(
    fn: Callable[..., Any] | types.ModuleType, fallback: str = "<unknown>"
) -> str:
    """Get the qualified name of a function."""
    name = getattr(fn, "__qualname__", None)
    if name is None:
        return fallback
    assert isinstance(name, str)
    return name


def iter_submodels(model: BaseModel, *, recursive: bool = True) -> Iterator[tuple[str, BaseModel]]:
    """Iterate through all nested BaseModel instances in fields.

    Supports field types:
    - BaseModel (direct instance)
    - list[BaseModel]
    - dict[str, BaseModel]

    Args:
        model: The BaseModel instance to iterate
        recursive: If True, also iterate through submodels of submodels

    Yields:
        Tuples of (path, submodel) where path is like "field", "field[0]", "field['key']"
    """

    def _iter(current: BaseModel, prefix: str) -> Iterator[tuple[str, BaseModel]]:
        for field_name in current.model_fields:
            value = getattr(current, field_name)
            if value is None:
                continue
            path = f"{prefix}.{field_name}" if prefix else field_name
            match value:
                case BaseModel() as submodel:
                    yield path, submodel
                    if recursive:
                        yield from _iter(submodel, path)
                case list() as items:
                    for idx, item in enumerate(items):
                        if isinstance(item, BaseModel):
                            item_path = f"{path}[{idx}]"
                            yield item_path, item
                            if recursive:
                                yield from _iter(item, item_path)
                case dict() as mapping:
                    for key, item in mapping.items():
                        if isinstance(item, BaseModel):
                            item_path = f"{path}[{key!r}]"
                            yield item_path, item
                            if recursive:
                                yield from _iter(item, item_path)

    yield from _iter(model, "")


def iter_submodel_types(
    model_cls: type[BaseModel], *, recursive: bool = True, include_union_members: bool = True
) -> Iterator[tuple[str, type[BaseModel]]]:
    """Iterate through all nested BaseModel types in field annotations.

    Supports field types:
    - BaseModel (direct type)
    - list[BaseModel]
    - dict[str, BaseModel]
    - BaseModel | OtherModel (unions, when include_union_members=True)

    Args:
        model_cls: The BaseModel class to iterate
        recursive: If True, also iterate through submodel types of submodels
        include_union_members: If True, yield each union member separately

    Yields:
        Tuples of (path, model_type) where path is like "field", "field[]", "field{}"
    """

    def _extract_model_types(
        annotation: type, *, expand_unions: bool = True
    ) -> Iterator[tuple[str, type[BaseModel]]]:
        """Extract BaseModel types from an annotation."""
        origin = get_origin(annotation)

        # Handle unions (X | Y or Union[X, Y])
        if origin is types.UnionType or origin is type(int | str):
            union_args = [a for a in get_args(annotation) if a is not type(None)]
            # X | None is Optional, always expand; X | Y is a true union
            is_optional = len(union_args) == 1
            if is_optional or expand_unions:
                for arg in union_args:
                    yield from _extract_model_types(arg, expand_unions=expand_unions)
            return

        # Handle list[X]
        if origin is list:
            list_args = get_args(annotation)
            if list_args:
                for suffix, model_type in _extract_model_types(
                    list_args[0], expand_unions=expand_unions
                ):
                    yield f"[]{suffix}", model_type
            return

        # Handle dict[str, X]
        if origin is dict:
            dict_args = get_args(annotation)
            if len(dict_args) >= 2:  # noqa: PLR2004
                for suffix, model_type in _extract_model_types(
                    dict_args[1], expand_unions=expand_unions
                ):
                    yield f"{{}}{suffix}", model_type
            return

        # Direct BaseModel subclass
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            yield "", annotation

    def _iter(
        cls: type[BaseModel], prefix: str, seen: set[type[BaseModel]]
    ) -> Iterator[tuple[str, type[BaseModel]]]:
        for field_name, field_info in cls.model_fields.items():
            annotation = field_info.annotation
            if annotation is None:
                continue
            base_path = f"{prefix}.{field_name}" if prefix else field_name

            for suffix, model_type in _extract_model_types(
                annotation, expand_unions=include_union_members
            ):
                path = f"{base_path}{suffix}"
                yield path, model_type
                if recursive and model_type not in seen:
                    seen.add(model_type)
                    yield from _iter(model_type, path, seen)

    yield from _iter(model_cls, "", set())


def models_to_markdown_docs(
    *models: type[BaseModel],
    header_level: int = 2,
    expand_mode: Literal["minimal", "maximal", "default"] = "default",
    seed: int = 0,
    validate: bool = False,
    exclude_none: bool = True,
    exclude_defaults: bool = False,
    exclude_unset: bool = False,
    indent: int = 2,
    sort_keys: bool = True,
) -> str:
    """Generate markdown documentation with commented YAML examples for BaseModel classes.

    Creates markdown chapters for each model with:
    - Model name as header
    - Model docstring
    - Commented YAML example with test data

    Args:
        models: One or more BaseModel classes to document
        header_level: Starting header level (2 = h2, 3 = h3, etc.)
        expand_mode: YAML generation mode - "minimal", "maximal", or "default"
        seed: Seed for deterministic YAML generation
        validate: Whether to validate the generated test data
        exclude_none: Exclude fields with None values in YAML
        exclude_defaults: Exclude fields with default values in YAML
        exclude_unset: Exclude fields that are not set in YAML
        indent: Indentation level for YAML output
        sort_keys: Sort keys in the YAML output

    Returns:
        Markdown string with chapters for each model

    Example:
        ```python
        from pydantic import BaseModel, Field

        class Person(BaseModel):
            '''A person with basic information.'''

            name: str = Field(description='Full name')
            age: int = Field(description='Age in years')
            email: str | None = Field(default=None, description='Email address')

        class Company(BaseModel):
            '''A company entity.'''

            name: str = Field(description='Company name')
            founded: int = Field(description='Year founded')

        # Generate documentation
        docs = models_to_markdown_docs(Person, Company)
        print(docs)
        ```

        Output:
        ```markdown
        ## Person

        A person with basic information.

        ```yaml
        age: 0  # Age in years
        name: a  # Full name
        ```

        ## Company

        A company entity.

        ```yaml
        founded: 0  # Year founded
        name: a  # Company name
        ```
        ```
    """
    from schemez.commented_yaml import create_yaml_description

    sections = []
    header_prefix = "#" * header_level

    for model in models:
        model_name = model.__name__
        model_doc = model.__doc__ or ""
        model_doc = model_doc.strip()

        yaml_example = create_yaml_description(
            model,
            exclude_none=exclude_none,
            exclude_defaults=exclude_defaults,
            exclude_unset=exclude_unset,
            indent=indent,
            comments=True,
            sort_keys=sort_keys,
            validate=validate,
            expand_mode=expand_mode,
        )

        section = f"{header_prefix} {model_name}\n\n"
        if model_doc:
            section += f"{model_doc}\n\n"
        section += f"```yaml\n{yaml_example}\n```"

        sections.append(section)

    return "\n\n".join(sections)


if __name__ == "__main__":

    class TestModel(BaseModel):
        test_int: int = 1
        test_str: str = "test"
        test_float: float = 1.1
        test_bool: bool = True

    code = model_to_python_code(TestModel)
    print(code)
