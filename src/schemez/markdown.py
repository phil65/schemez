"""Schema to Markdown conversion utilities."""

from __future__ import annotations

import importlib
import inspect
import types
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, get_args, get_origin


if TYPE_CHECKING:
    import jinja2
    from pydantic import BaseModel

OutputMode = Literal["table", "python_code", "yaml"]
HeaderStyle = Literal["default", "pymdownx"]
SerializationMode = Literal["json", "python"]


def _format_code_header(
    path: str,
    line_start: int,
    line_end: int,
    style: HeaderStyle = "default",
    title: str | None = None,
    language: str = "python",
) -> str:
    """Format code block header based on style.

    Args:
        path: File path or identifier
        line_start: Starting line number
        line_end: Ending line number
        style: Header style - "default" or "pymdownx"
        title: Optional title (for pymdownx style)
        language: Language for syntax highlighting (for pymdownx style)

    Returns:
        Formatted header string for the code block
    """
    if style == "pymdownx":
        # Format: ```python title="filename" linenums="1"
        header = f"```{language}"
        if title:
            header += f' title="{title}"'
        if line_start > 0:
            header += f' linenums="{line_start}"'
        return header
    # Format: ```path#L1-10
    return f"```{path}#L{line_start}-{line_end}"


DEFAULT_TEMPLATE = """\
{%- macro render_model(model, level) %}
{{ '#' * level }} {{ model.name }}
{% if model.description %}
{{ model.description }}
{% endif %}
{%- if model.fields %}

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
{% for field in model.fields -%}
| `{{ field.name }}` | `{{ field.type }}` | {{ "✓" if field.required else "" }} | {{ field.default if field.default is not none else "-" }} | {{ field.description or "" }} |
{% endfor %}
{%- if include_constraints and model.constraints %}

{{ '#' * (level + 1) }} Constraints
{% for field_name, constraints in model.constraints.items() %}
- `{{ field_name }}`: {{ constraints | join(", ") }}
{%- endfor %}
{% endif %}
{%- if include_examples and model.examples %}

{{ '#' * (level + 1) }} Examples
{% for field_name, examples in model.examples.items() %}
- `{{ field_name }}`: {{ examples | join(", ") }}
{%- endfor %}
{% endif %}
{%- endif %}
{%- endmacro %}
{{ render_model(root, header_level) }}
{%- if nested_models %}

{{ '#' * header_level }} Nested Models
{% for model in nested_models.values() %}
{{ render_model(model, header_level + 1) }}
{% endfor %}
{%- endif %}
"""  # noqa: E501


def _clean_markdown(text: str) -> str:
    """Remove excessive blank lines from markdown output."""
    import re

    # Replace 3+ newlines with 2 newlines
    return re.sub(r"\n{3,}", "\n\n", text)


def _get_secret_fields(model: type[BaseModel]) -> set[str]:
    """Get field names that are SecretStr types.

    Args:
        model: Pydantic model class

    Returns:
        Set of field names that are SecretStr or Optional[SecretStr]
    """
    from pydantic import SecretStr

    secret_fields = set()
    for field_name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        # Check if field is SecretStr or Optional[SecretStr]
        if annotation is SecretStr:
            secret_fields.add(field_name)
        elif get_origin(annotation) is Union:
            args = get_args(annotation)
            if SecretStr in args:
                secret_fields.add(field_name)
    return secret_fields


def _process_secret_str(
    data: Any, model: type[BaseModel] | None = None, mask_value: str | None = None
) -> Any:
    """Recursively convert SecretStr objects to masked strings.

    Args:
        data: Data structure potentially containing SecretStr objects
        model: Optional Pydantic model to identify SecretStr fields
        mask_value: Placeholder string to use for masked values. Defaults to "!ENV ENV_VAR_NAME"

    Returns:
        Data with SecretStr objects converted to strings
    """
    from pydantic import SecretStr

    mask_value = mask_value or "!ENV ENV_VAR_NAME"
    if isinstance(data, SecretStr):
        return mask_value
    if isinstance(data, dict) and model is not None:
        secret_fields = _get_secret_fields(model)
        return {
            k: mask_value if k in secret_fields else _process_secret_str(v, mask_value=mask_value)
            for k, v in data.items()
        }
    if isinstance(data, dict):
        return {k: _process_secret_str(v, mask_value=mask_value) for k, v in data.items()}
    if isinstance(data, list):
        return [_process_secret_str(item, mask_value=mask_value) for item in data]
    return data


def _resolve_type_name(type_schema: dict[str, Any], defs: dict[str, Any]) -> str:  # noqa: PLR0911
    """Resolve a JSON schema type to a human-readable type name."""
    if "$ref" in type_schema:
        return type_schema["$ref"].rsplit("/", 1)[-1]  # type: ignore[no-any-return]

    if "anyOf" in type_schema:
        types = []
        for option in type_schema["anyOf"]:
            if option.get("type") == "null":
                continue
            types.append(_resolve_type_name(option, defs))
        if len(types) == 1:
            return f"{types[0]} | None"
        return " | ".join(types) + " | None"

    if "oneOf" in type_schema:
        types = [_resolve_type_name(opt, defs) for opt in type_schema["oneOf"]]
        return " | ".join(types)

    if "allOf" in type_schema:
        # Usually just one item for inheritance
        if len(type_schema["allOf"]) == 1:
            return _resolve_type_name(type_schema["allOf"][0], defs)
        types = [_resolve_type_name(opt, defs) for opt in type_schema["allOf"]]
        return " & ".join(types)

    type_val = type_schema.get("type", "any")

    match type_val:
        case "array":
            items = type_schema.get("items", {})
            item_type = _resolve_type_name(items, defs) if items else "Any"
            return f"list[{item_type}]"
        case "object":
            if "additionalProperties" in type_schema:
                additional = type_schema["additionalProperties"]
                if isinstance(additional, dict):
                    val_type = _resolve_type_name(additional, defs)
                    return f"dict[str, {val_type}]"
                # additionalProperties: true/false - just a generic dict
                return "dict"
            return "dict"
        case "string":
            if "enum" in type_schema:
                return "Literal[" + ", ".join(f'"{v}"' for v in type_schema["enum"]) + "]"
            if "format" in type_schema:
                return f"str ({type_schema['format']})"
            return "str"
        case "integer":
            return "int"
        case "number":
            return "float"
        case "boolean":
            return "bool"
        case "null":
            return "None"
        case list() as types:
            # Handle ["string", "null"] style
            non_null = [t for t in types if t != "null"]
            has_null = "null" in types
            if len(non_null) == 1:
                base = _resolve_type_name({"type": non_null[0]}, defs)
                return f"{base} | None" if has_null else base
            type_strs = [_resolve_type_name({"type": t}, defs) for t in non_null]
            result = " | ".join(type_strs)
            return f"{result} | None" if has_null else result
        case _:
            return str(type_val)


def _extract_constraints(field_schema: dict[str, Any]) -> list[str]:
    """Extract constraint descriptions from a field schema."""
    constraints = []
    constraint_keys = {
        "minimum": "min",
        "maximum": "max",
        "exclusiveMinimum": "exclusive_min",
        "exclusiveMaximum": "exclusive_max",
        "minLength": "min_length",
        "maxLength": "max_length",
        "pattern": "pattern",
        "minItems": "min_items",
        "maxItems": "max_items",
        "uniqueItems": "unique",
        "multipleOf": "multiple_of",
    }
    for key, label in constraint_keys.items():
        if key in field_schema:
            value = field_schema[key]
            if key == "uniqueItems" and value:
                constraints.append(f"`{label}`")
            elif key == "pattern":
                # Use inline code for pattern to preserve regex
                constraints.append(f"{label}=`{value}`")
            else:
                constraints.append(f"{label}=`{value}`")
    return constraints


def _format_default(value: Any) -> str:
    """Format a default value for display."""
    if value is None:
        return "None"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, dict | list):
        return repr(value)
    return str(value)


def _extract_field_info(
    field_name: str,
    field_schema: dict[str, Any],
    required_fields: set[str],
    defs: dict[str, Any],
) -> dict[str, Any]:
    """Extract field information from a JSON schema property."""
    # Handle anyOf/oneOf for the actual schema
    actual_schema = field_schema
    if "anyOf" in field_schema:
        for option in field_schema["anyOf"]:
            if option.get("type") != "null":
                actual_schema = option
                break

    return {
        "name": field_name,
        "type": _resolve_type_name(field_schema, defs),
        "description": field_schema.get("description", ""),
        "required": field_name in required_fields,
        "default": _format_default(field_schema.get("default"))
        if "default" in field_schema
        else None,
        "examples": field_schema.get("examples", []),
        "constraints": _extract_constraints(actual_schema),
        "is_nested": "$ref" in field_schema or "$ref" in actual_schema,
    }


def _extract_model_info(
    name: str,
    schema: dict[str, Any],
    defs: dict[str, Any],
    model_path: str | None = None,
) -> dict[str, Any]:
    """Extract model information from a JSON schema."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields = [
        _extract_field_info(fname, fschema, required, defs) for fname, fschema in properties.items()
    ]

    # Collect constraints and examples per field
    constraints: dict[str, list[str]] = {}
    examples: dict[str, list[Any]] = {}
    for field in fields:
        if field["constraints"]:
            constraints[field["name"]] = field["constraints"]
        if field["examples"]:
            examples[field["name"]] = field["examples"]

    # Use title from schema if available, otherwise use name
    display_name = schema.get("x-doc-title", schema.get("title", name))

    # Get description and prepend model path if available
    description = schema.get("description", "")
    if model_path and display_name != name:
        # Only add model path line if we're using a custom title
        description = (
            f"Model path: `{model_path}`\n\n{description}"
            if description
            else f"Model path: `{model_path}`"
        )

    return {
        "name": display_name,
        "description": description,
        "fields": fields,
        "constraints": constraints,
        "examples": examples,
    }


def schema_to_markdown_context(
    model: type[BaseModel],
    *,
    header_level: int = 1,
    include_defaults: bool = True,
    include_examples: bool = True,
    include_constraints: bool = True,
) -> dict[str, Any]:
    """Convert a Pydantic model to a Jinja2 template context.

    Args:
        model: The Pydantic model class to convert
        header_level: Starting header level (1 = h1)
        include_defaults: Include default values in output
        include_examples: Include examples in output
        include_constraints: Include constraints section

    Returns:
        Dictionary suitable for rendering with the markdown template
    """
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})

    # Build model path for the root model
    model_path = f"{model.__module__}.{model.__qualname__}"
    root = _extract_model_info(model.__name__, schema, defs, model_path=model_path)

    nested_models: dict[str, dict[str, Any]] = {}
    for def_name, def_schema in defs.items():
        nested_models[def_name] = _extract_model_info(def_name, def_schema, defs)

    return {
        "root": root,
        "nested_models": nested_models,
        "header_level": header_level,
        "include_defaults": include_defaults,
        "include_examples": include_examples,
        "include_constraints": include_constraints,
    }


def _resolve_from_import_path(import_path: str) -> Any:
    """Resolve any object from an import path.

    Supports both 'module.submodule:Name' and 'module.submodule.Name' formats.
    """
    if ":" in import_path:
        # Format: module.path:Name
        module_path, attr_name = import_path.split(":", 1)
    else:
        # Format: module.path.Name - split on last dot
        if "." not in import_path:
            msg = f"Import path must contain module and attribute, got: {import_path}"
            raise ValueError(msg)
        module_path, attr_name = import_path.rsplit(".", 1)

    module = importlib.import_module(module_path)

    if not hasattr(module, attr_name):
        msg = f"'{attr_name}' not found in module '{module_path}'"
        raise AttributeError(msg)

    return getattr(module, attr_name)


def _resolve_model_from_import_path(import_path: str) -> type[BaseModel]:
    """Resolve a model class from an import path.

    Supports both 'module.submodule:ClassName' and 'module.submodule.ClassName' formats.
    """
    model_class = _resolve_from_import_path(import_path)

    # Basic check that it's a BaseModel - we can't do full isinstance check due to imports
    if not hasattr(model_class, "model_json_schema"):
        msg = f"'{import_path}' is not a Pydantic BaseModel class"
        raise TypeError(msg)

    return model_class  # type: ignore[no-any-return]


def model_to_markdown(
    model: type[BaseModel] | str,
    *,
    template: str | None = None,
    header_level: int = 1,
    include_defaults: bool = True,
    include_examples: bool = True,
    include_constraints: bool = True,
    display_mode: OutputMode = "table",
    mode: Literal["minimal", "maximal", "default"] = "default",
    exclude_none: bool = False,
    exclude_defaults: bool = False,
    exclude_unset: bool = False,
    indent: int = 2,
    header_style: HeaderStyle = "default",
    serialization_mode: SerializationMode = "json",
    mask_value: str | None = None,
    as_listitem: bool = True,
    wrapped_in: str | None = None,
) -> str:
    """Convert a Pydantic model class to Markdown documentation.

    Args:
        model: The Pydantic model class to document or import path string
            like 'module.path:ClassName' or 'module.path.ClassName'
        template: Custom Jinja2 template string (uses DEFAULT_TEMPLATE if None)
        header_level: Starting header level (1 = h1, 2 = h2, etc.)
        include_defaults: Include default values in the table
        include_examples: Include examples section
        include_constraints: Include constraints section
        display_mode: Output format - "table", "python_code", or "yaml"
        mode: Generation mode for YAML examples
        exclude_none: Exclude None values from YAML
        exclude_defaults: Exclude default values from YAML
        exclude_unset: Exclude unset values from YAML
        indent: YAML indentation
        header_style: Code block header style - "default" or "pymdownx"
        serialization_mode: Pydantic serialization mode - "json" (default) or "python"
        mask_value: Mask value for SecretStr objects
        as_listitem: Wrap the YAML config in a list (default: True)
        wrapped_in: Wrap the config in a dict with this key (default: None)

    Returns:
        Markdown string documenting the model
    """
    import jinja2

    if isinstance(model, str):
        model = _resolve_model_from_import_path(model)

    if display_mode == "python_code":
        try:
            source = inspect.getsource(model)
            num_lines = len(source.splitlines())
            path = f"{model.__module__}.{model.__name__}"
            header = _format_code_header(
                path, 1, num_lines, style=header_style, title=model.__name__, language="python"
            )
        except OSError:
            # Source code not available (e.g., dynamically created classes)
            path = f"{model.__module__}.{model.__name__}"
            header = _format_code_header(
                path, 1, 1, style=header_style, title=model.__name__, language="python"
            )
            return f"{header}\n# Source code not available for {model.__name__}\n```\n"
        else:
            return f"{header}\n{source}```\n"

    if display_mode == "yaml":
        import yamling

        from schemez.commented_yaml import process_yaml_lines
        from schemez.generators import SchemaDataGenerator

        json_schema = model.model_json_schema()
        generator = SchemaDataGenerator(json_schema)

        if mode == "minimal":
            data = generator.generate_minimal()
        elif mode == "maximal":
            data = generator.generate_maximal()
        else:  # default
            data = generator.generate()

        try:
            instance = model.model_validate(data)
        except Exception:  # noqa: BLE001
            # Validation may fail due to constraints, fall back to model_construct
            instance = model.model_construct(**data)

        # Use 'python' mode to avoid SecretStr serialization issues
        yaml_data: dict[str, Any] | list[Any] = instance.model_dump(
            exclude_none=exclude_none,
            exclude_defaults=exclude_defaults,
            exclude_unset=exclude_unset,
            mode="python",
        )

        # Filter out synthetic example_name field added by generator
        if isinstance(yaml_data, dict):
            yaml_data.pop("example_name", None)
            yaml_data.pop("example_name_", None)  # In case of collision

        # Convert SecretStr objects to masked strings
        if isinstance(yaml_data, dict):
            yaml_data = _process_secret_str(yaml_data, model, mask_value=mask_value)

        # Apply wrapping options
        if as_listitem:
            yaml_data = [yaml_data]
        if wrapped_in:
            yaml_data = {wrapped_in: yaml_data}

        base_yaml = yamling.dump_yaml(
            yaml_data,
            sort_keys=False,
            indent=indent,
            default_flow_style=None,
            allow_unicode=True,
        )

        yaml_lines = base_yaml.strip().split("\n")
        commented_lines = process_yaml_lines(
            yaml_lines, json_schema, as_listitem=as_listitem, wrapped_in=wrapped_in
        )
        yaml_content = "\n".join(commented_lines)

        # Build output with header and docstring
        result_parts = []

        # Get model title from schema or use class name
        json_schema = model.model_json_schema()
        model_title = json_schema.get("x-doc-title", json_schema.get("title", model.__name__))

        # Add markdown header
        result_parts.append(f"{'#' * header_level} {model_title}\n\n")

        # Add class docstring if available
        if model.__doc__:
            docstring = inspect.cleandoc(model.__doc__)
            result_parts.append(f"{docstring}\n\n")

        # Add YAML code block
        path = f"{model.__module__}.{model.__name__}.yaml"
        header = _format_code_header(
            path,
            1,
            len(commented_lines),
            style=header_style,
            title=f"{model_title} (YAML)",
            language="yaml",
        )
        result_parts.append(f"{header}\n{yaml_content}\n```\n")

        return "".join(result_parts)

    # Default table mode
    context = schema_to_markdown_context(
        model,
        header_level=header_level,
        include_defaults=include_defaults,
        include_examples=include_examples,
        include_constraints=include_constraints,
    )

    env = jinja2.Environment(autoescape=False)
    tmpl = env.from_string(template or DEFAULT_TEMPLATE)
    result = tmpl.render(**context).strip() + "\n"
    return _clean_markdown(result)


def instance_to_markdown(
    instance: BaseModel,
    *,
    template: str | None = None,
    header_level: int = 1,
    include_defaults: bool = True,
    include_examples: bool = True,
    include_constraints: bool = True,
    include_values: bool = True,
    display_mode: OutputMode = "table",
    exclude_none: bool = False,
    exclude_defaults_yaml: bool = False,
    exclude_unset: bool = False,
    indent: int = 2,
    header_style: HeaderStyle = "default",
    serialization_mode: SerializationMode = "json",
    mask_value: str | None = None,
    as_listitem: bool = True,
    wrapped_in: str | None = None,
) -> str:
    """Convert a Pydantic model instance to Markdown documentation.

    When include_values is True, the current values are shown alongside defaults.

    Args:
        instance: The Pydantic model instance to document
        template: Custom Jinja2 template string
        header_level: Starting header level (1 = h1, 2 = h2, etc.)
        include_defaults: Include default values in the table
        include_examples: Include examples section
        include_constraints: Include constraints section
        include_values: Include current instance values
        display_mode: Output format - "table", "python_code", or "yaml"
        exclude_none: Exclude None values from YAML
        exclude_defaults_yaml: Exclude default values from YAML
        exclude_unset: Exclude unset values from YAML
        indent: YAML indentation
        header_style: Code block header style - "default" or "pymdownx"
        serialization_mode: Pydantic serialization mode - "json" (default) or "python"
        mask_value: Placeholder string to use for masked values. Defaults to "!ENV ENV_VAR_NAME"
        as_listitem: Wrap the YAML config in a list (default: True)
        wrapped_in: Wrap the config in a dict with this key (default: None)

    Returns:
        Markdown string documenting the model instance
    """
    if display_mode == "python_code":
        model_class = type(instance)
        try:
            source = inspect.getsource(model_class)
            num_lines = len(source.splitlines())
            path = f"{model_class.__module__}.{model_class.__name__}"
            header = _format_code_header(
                path,
                1,
                num_lines,
                style=header_style,
                title=model_class.__name__,
                language="python",
            )
        except OSError:
            # Source code not available (e.g., dynamically created classes)
            path = f"{model_class.__module__}.{model_class.__name__}"
            header = _format_code_header(
                path, 1, 1, style=header_style, title=model_class.__name__, language="python"
            )
            return f"{header}\n# Source code not available for {model_class.__name__}\n```\n"
        else:
            return f"{header}\n{source}```\n"

    if display_mode == "yaml":
        import yamling

        from schemez.commented_yaml import process_yaml_lines

        yaml_data: dict[str, Any] | list[Any] = instance.model_dump(
            exclude_none=exclude_none,
            exclude_defaults=exclude_defaults_yaml,
            exclude_unset=exclude_unset,
            mode=serialization_mode,
        )

        # Filter out synthetic example_name field added by generator
        if isinstance(yaml_data, dict):
            yaml_data.pop("example_name", None)
            yaml_data.pop("example_name_", None)  # In case of collision

        # Apply wrapping options
        if as_listitem:
            yaml_data = [yaml_data]
        if wrapped_in:
            yaml_data = {wrapped_in: yaml_data}

        base_yaml = yamling.dump_yaml(
            yaml_data,
            sort_keys=False,
            indent=indent,
            default_flow_style=None,
            allow_unicode=True,
        )

        json_schema = instance.model_json_schema()
        yaml_lines = base_yaml.strip().split("\n")
        commented_lines = process_yaml_lines(yaml_lines, json_schema)
        yaml_content = "\n".join(commented_lines)
        # Build output with header and docstring
        model_class = type(instance)
        result_parts = []
        # Get model title from schema or use class name
        model_title = json_schema.get("x-doc-title", json_schema.get("title", model_class.__name__))
        # Add markdown header
        result_parts.append(f"{'#' * header_level} {model_title}\n\n")
        # Add class docstring if available
        if model_class.__doc__:
            docstring = inspect.cleandoc(model_class.__doc__)
            result_parts.append(f"{docstring}\n\n")

        # Add YAML code block
        path = f"{model_class.__module__}.{model_class.__name__}.yaml"
        header = _format_code_header(
            path,
            1,
            len(commented_lines),
            style=header_style,
            title=f"{model_title} (YAML)",
            language="yaml",
        )
        result_parts.append(f"{header}\n{yaml_content}\n```\n")

        return "".join(result_parts)

    # Default table mode
    result = model_to_markdown(
        type(instance),
        template=template,
        header_level=header_level,
        include_defaults=include_defaults,
        include_examples=include_examples,
        include_constraints=include_constraints,
        display_mode="table",
        header_style=header_style,
        mask_value=mask_value,
    )

    if include_values:
        values_md = f"\n{'#' * header_level} Current Values\n\n"
        values_md += "| Field | Value |\n|-------|-------|\n"
        for field_name, value in instance.model_dump().items():
            values_md += f"| `{field_name}` | `{_format_default(value)}` |\n"
        result = result.rstrip() + "\n" + values_md

    return result


def model_union_to_markdown(
    union_type: Any,  # type | str | types.UnionType | typing._SpecialForm,
    *,
    template: str | None = None,
    header_level: int = 1,
    include_defaults: bool = True,
    include_examples: bool = True,
    include_constraints: bool = True,
    display_mode: OutputMode = "table",
    mode: Literal["minimal", "maximal", "default"] = "default",
    exclude_none: bool = False,
    exclude_defaults: bool = False,
    exclude_unset: bool = False,
    indent: int = 2,
    header_style: HeaderStyle = "default",
    serialization_mode: SerializationMode = "json",
    mask_value: str | None = None,
    as_listitem: bool = True,
    wrapped_in: str | None = None,
) -> str:
    """Convert a Union type containing Pydantic models to Markdown documentation.

    Runtime-inspects the Union and generates markdown for all included BaseModel types.

    Args:
        union_type: A Union type containing Pydantic model classes or import path string
            like 'module.path:UnionAlias' or 'module.path.UnionAlias'
        template: Custom Jinja2 template string (uses DEFAULT_TEMPLATE if None)
        header_level: Starting header level (1 = h1, 2 = h2, etc.)
        include_defaults: Include default values in the table
        include_examples: Include examples section
        include_constraints: Include constraints section
        display_mode: Output format - "table", "python_code", or "yaml"
        mode: Generation mode for YAML examples
        exclude_none: Exclude None values from YAML
        exclude_defaults: Exclude default values from YAML
        exclude_unset: Exclude unset values from YAML
        indent: YAML indentation
        header_style: Code block header style - "default" or "pymdownx"
        serialization_mode: Pydantic serialization mode - "json" (default) or "python"
        mask_value: Placeholder string to use for masked values. Defaults to "!ENV ENV_VAR_NAME"
        as_listitem: Wrap the YAML config in a list (default: True)
        wrapped_in: Wrap the config in a dict with this key (default: None)

    Returns:
        Markdown string documenting all models in the union
    """
    # Resolve import path if string
    if isinstance(union_type, str):
        union_type = _resolve_from_import_path(union_type)

    # Unwrap Annotated if needed
    origin = get_origin(union_type)
    if origin is Annotated:
        # Get the actual type from Annotated[Union[...], ...]
        union_type = get_args(union_type)[0]
        origin = get_origin(union_type)

    # Check if it's a Union type
    if origin not in (Union, types.UnionType):  # Check if it's a Union
        msg = f"Expected Union type, got: {union_type}"
        raise TypeError(msg)

    # Get all types in the union
    union_args = get_args(union_type)

    # Filter for BaseModel classes
    model_classes = [
        i
        for i in union_args
        if inspect.isclass(i) and hasattr(i, "model_json_schema") and i is not type(None)
    ]

    if not model_classes:
        msg = f"No Pydantic BaseModel classes found in union: {union_type}"
        raise ValueError(msg)

    # Generate markdown for each model
    markdown_parts = []
    for model_class in model_classes:
        model_md = model_to_markdown(
            model_class,
            template=template,
            header_level=header_level,
            include_defaults=include_defaults,
            include_examples=include_examples,
            include_constraints=include_constraints,
            display_mode=display_mode,
            mode=mode,
            exclude_none=exclude_none,
            exclude_defaults=exclude_defaults,
            exclude_unset=exclude_unset,
            indent=indent,
            header_style=header_style,
            serialization_mode=serialization_mode,
            mask_value=mask_value,
            as_listitem=as_listitem,
            wrapped_in=wrapped_in,
        )
        markdown_parts.append(model_md)

    # Combine all markdown with separators
    if display_mode in ("python_code", "yaml"):
        result = "\n\n".join(markdown_parts)
    else:
        result = "\n\n---\n\n".join(markdown_parts)
    return _clean_markdown(result)


def setup_jinjarope_filters(env: jinja2.Environment) -> None:
    """Set up schemez markdown filters for jinjarope.

    This function is used as an entry point for the jinjarope.environment
    extension group to add markdown-related filters to Jinja2 environments.

    Args:
        env: The Jinja2 environment to modify
    """
    # Add filters to the environment
    env.filters["schema_to_markdown"] = model_to_markdown
    env.filters["instance_to_markdown"] = instance_to_markdown
    env.filters["union_to_markdown"] = model_union_to_markdown


if __name__ == "__main__":
    # Test with llmling-agent configs
    from llmling_agent_config.event_handlers import EventHandlerConfig
    from llmling_agent_config.mcp_server import MCPServerConfig
    from llmling_agent_config.teams import TeamConfig

    print("# MCPServerConfig Union\n")
    md = model_union_to_markdown(
        MCPServerConfig,
        display_mode="yaml",
        header_style="pymdownx",
    )
    print(md)

    print("\n# EventHandlerConfig Union\n")
    md = model_union_to_markdown(
        EventHandlerConfig,
        display_mode="yaml",
        header_style="pymdownx",
    )
    print(md)

    print("\n# TeamConfig Model\n")
    md = model_to_markdown(
        TeamConfig,
        display_mode="yaml",
        header_style="pymdownx",
    )
    print(md)
