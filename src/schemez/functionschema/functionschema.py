"""Module for creating OpenAI function schemas from Python functions."""

from __future__ import annotations

from collections.abc import (
    Callable,  # noqa: TC003
    Sequence,  # noqa: F401
)
import inspect
import typing
from typing import Any, Literal, get_args, get_origin

import pydantic
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema

from schemez import log
from schemez.functionschema.helpers import (
    determine_function_type,
    resolve_type_annotation,
)
from schemez.functionschema.typedefs import (
    OpenAIFunctionDefinition,
    OpenAIFunctionTool,
    ToolParameters,
)


if typing.TYPE_CHECKING:
    from schemez.functionschema.typedefs import Property


logger = log.get_logger(__name__)

TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list[Any],
    "object": dict[str, Any],
}


SchemaType = Literal["jsonschema", "openai", "simple"]


class FunctionSchema(pydantic.BaseModel):
    """Schema representing an OpenAI function definition and metadata.

    This class encapsulates all the necessary information to describe a function to the
    OpenAI API, including its name, description, parameters, return type, and execution
    characteristics. It follows the OpenAI function calling format while adding
    additional metadata useful for Python function handling.
    """

    name: str
    """The name of the function as it will be presented to the OpenAI API."""

    description: str | None = None
    """Optional description of what the function does."""

    parameters: ToolParameters = pydantic.Field(
        default_factory=lambda: ToolParameters(type="object", properties={}, required=[]),
    )
    """JSON Schema object describing the function's parameters."""

    required: list[str] = pydantic.Field(default_factory=list)
    """
    List of parameter names that are required (do not have default values).
    """

    returns: dict[str, Any] = pydantic.Field(
        default_factory=lambda: {"type": "object"},
    )
    """JSON Schema object describing the function's return type."""

    model_config = pydantic.ConfigDict(frozen=True)

    def _create_pydantic_model(self) -> type[pydantic.BaseModel]:
        """Create a Pydantic model from the schema parameters."""
        fields: dict[str, tuple[type[Any] | Literal, pydantic.Field]] = {}  # type: ignore[valid-type]
        properties = self.parameters.get("properties", {})
        required = self.parameters.get("required", self.required)

        for name, details in properties.items():
            if name.startswith("_"):  # TODO: kwarg for renaming instead perhaps?
                logger.debug("Skipping parameter %s due to leading underscore", name)
                continue
            # Get base type
            if "enum" in details:
                values = tuple(details["enum"])
                param_type: Any = Literal[values]
            else:
                param_type = TYPE_MAP.get(details.get("type", "string"), Any)

            # Handle optional types (if there's a default of None)
            default_value = details.get("default")
            if default_value is None and name not in required:
                param_type = param_type | None

            # Create a proper pydantic Field
            field = (
                param_type,
                pydantic.Field(default=... if name in required else default_value),
            )
            fields[name] = field

        return pydantic.create_model(f"{self.name}_params", **fields)  # type: ignore[call-overload, no-any-return]

    def model_dump_openai(self) -> OpenAIFunctionTool:
        """Convert the schema to OpenAI's function calling format.

        Returns:
            A dictionary matching OpenAI's complete function tool definition format.
            {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        """  # noqa: E501
        parameters: ToolParameters = {
            "type": "object",
            "properties": self.parameters["properties"],
            "required": self.required,
        }

        # First create the function definition
        function_def = OpenAIFunctionDefinition(
            name=self.name,
            description=self.description or "",
            parameters=parameters,
        )

        return OpenAIFunctionTool(type="function", function=function_def)

    def to_python_signature(self) -> inspect.Signature:
        """Convert the schema back to a Python function signature.

        This method creates a Python function signature from the OpenAI schema,
        mapping JSON schema types back to their Python equivalents.

        Returns:
            A function signature representing the schema parameters

        Example:
            ```python
            schema = FunctionSchema(...)
            sig = schema.to_python_signature()
            print(str(sig))  # -> (location: str, unit: str = None, ...)
            ```
        """
        model = self._create_pydantic_model()
        parameters: list[inspect.Parameter] = []
        for name, field in model.model_fields.items():
            default = inspect.Parameter.empty if field.is_required() else field.default
            param = inspect.Parameter(
                name=name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=field.annotation,
                default=default,
            )
            parameters.append(param)
        param_type = TYPE_MAP.get(self.returns.get("type", "string"), Any)
        return inspect.Signature(parameters=parameters, return_annotation=param_type)

    def to_return_model_code(self, class_name: str | None = None) -> str:
        """Generate Pydantic model code for return type using datamodel-codegen.

        Args:
            class_name: Name for the generated class (default: {name}Response)

        Returns:
            Generated Python code string

        Raises:
            ValueError: If schema parsing fails
        """
        from schemez.helpers import json_schema_to_pydantic_code

        name = class_name or f"{self.name.title()}Response"
        return json_schema_to_pydantic_code(
            self.returns,
            class_name=name,
            target_python_version="3.13",
        )

    def to_parameter_model_code(self, class_name: str | None = None) -> str:
        """Generate Pydantic model code for parameters using datamodel-codegen.

        Args:
            class_name: Name for the generated class (default: {name}Params)

        Returns:
            Generated Python code string

        Raises:
            ValueError: If schema parsing fails
        """
        from schemez.helpers import json_schema_to_pydantic_code

        name = class_name or f"{self.name.title()}Params"
        return json_schema_to_pydantic_code(
            self.parameters,
            class_name=name,
            target_python_version="3.13",
        )

    def get_annotations(self, return_type: Any = str) -> dict[str, type[Any]]:
        """Get a dictionary of parameter names to their Python types.

        This can be used directly for __annotations__ assignment.

        Returns:
            Dictionary mapping parameter names to their Python types.
        """
        model = self._create_pydantic_model()
        annotations: dict[str, type[Any]] = {}
        for name, field in model.model_fields.items():
            annotations[name] = field.annotation  # type: ignore[assignment]
        annotations["return"] = return_type
        return annotations

    @classmethod
    def from_dict(
        cls,
        schema: dict[str, Any],
        output_schema: dict[str, Any] | None = None,
    ) -> FunctionSchema:
        """Create a FunctionSchema from a raw schema dictionary.

        Args:
            schema: OpenAI function schema dictionary.
                Can be either a direct function definition or a tool wrapper.
            output_schema: Optional dictionary specifying the return type.
                If not provided, an object type will be set.

        Returns:
            New FunctionSchema instance

        Raises:
            ValueError: If schema format is invalid or missing required fields
        """
        from schemez.functionschema.typedefs import clean_property

        # Handle tool wrapper format
        if "type" in schema and schema["type"] == "function":
            if "function" not in schema:
                msg = 'Tool with type "function" must have a "function" field'
                raise ValueError(msg)
            schema = schema["function"]
        elif "type" in schema and schema.get("type") != "function":
            msg = f"Unknown tool type: {schema.get('type')}"
            raise ValueError(msg)

        # Get function name
        name = schema.get("name", schema.get("function", {}).get("name"))
        if not name:
            msg = 'Schema must have a "name" field'
            raise ValueError(msg)

        # Extract parameters
        param_dict = schema.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(param_dict, dict):
            msg = "Schema parameters must be a dictionary"
            raise ValueError(msg)  # noqa: TRY004

        # Clean up properties that have advanced JSON Schema features
        properties = param_dict.get("properties", {})
        cleaned_props: dict[str, Property] = {}
        for prop_name, prop in properties.items():
            cleaned_props[prop_name] = clean_property(prop)
        required = param_dict.get("required", [])
        parameters: ToolParameters = {"type": "object", "properties": cleaned_props}
        if required:
            parameters["required"] = required
        return cls(
            name=name,
            description=schema.get("description"),
            parameters=parameters,
            required=required,
            returns=output_schema or {"type": "object"},
        )


def _types_match(annotation: Any, exclude_type: type) -> bool:
    """Check if annotation matches exclude_type using various strategies."""
    try:
        # Direct type match
        if annotation is exclude_type:
            return True

        # Handle generic types - get origin for comparison
        origin_annotation = get_origin(annotation)
        if origin_annotation is exclude_type:
            return True

        # String-based comparison for forward references and __future__.annotations
        annotation_str = str(annotation)
        exclude_type_name = exclude_type.__name__
        exclude_type_full_name = f"{exclude_type.__module__}.{exclude_type.__name__}"

        # Check various string representations
        if (
            exclude_type_name in annotation_str
            or exclude_type_full_name in annotation_str
        ):
            # Be more specific to avoid false positives
            # Check if it's the exact type name, not just a substring
            import re

            patterns = [
                rf"\b{re.escape(exclude_type_name)}\b",
                rf"\b{re.escape(exclude_type_full_name)}\b",
            ]
            if any(re.search(pattern, annotation_str) for pattern in patterns):
                return True

    except Exception:  # noqa: BLE001
        pass

    return False


def create_schema(
    func: Callable[..., Any],
    name_override: str | None = None,
    description_override: str | None = None,
    exclude_types: list[type] | None = None,
    mode: SchemaType = "simple",
) -> FunctionSchema:
    """Create an OpenAI function schema from a Python function.

    If an iterator is passed, the schema return type is a list of the iterator's
    element type.
    Variable arguments (*args) and keyword arguments (**kwargs) are not
    supported in OpenAI function schemas and will be ignored with a warning.

    Args:
        func: Function to create schema for
        name_override: Optional name override (otherwise the function name)
        description_override: Optional description override
                              (otherwise the function docstring)
        exclude_types: Types to exclude from parameters (e.g., context types)
        mode: Schema generation mode
              - simple" (default) uses custom simple implementation,
              - "openai" for OpenAI function calling via Pydantic,
              - "jsonschema" for standard JSON schema via Pydantic

    Returns:
        Schema representing the function

    Raises:
        TypeError: If input is not callable
    """
    if not callable(func):
        msg = f"Expected callable, got {type(func)}"
        raise TypeError(msg)

    exclude_types = exclude_types or []
    if mode == "simple":
        return _create_schema_simple(
            func, name_override, description_override, exclude_types
        )
    return _create_schema_pydantic(
        func,
        name_override,
        description_override,
        exclude_types,
        use_openai_format=mode == "openai",
    )


def _create_schema_pydantic(
    func: Callable[..., Any],
    name_override: str | None,
    description_override: str | None,
    exclude_types: list[type],
    use_openai_format: bool,
) -> FunctionSchema:
    """Create schema using Pydantic's internal schema generation."""
    from pydantic._internal import _generate_schema, _typing_extra
    from pydantic._internal._config import ConfigWrapper
    from pydantic_core import core_schema

    # Try to use pydantic-ai's OpenAI-compatible generator if available
    schema_generator_cls: type[GenerateJsonSchema]
    if use_openai_format:
        from pydantic_ai.tools import GenerateToolJsonSchema

        schema_generator_cls = GenerateToolJsonSchema
    else:
        schema_generator_cls = GenerateJsonSchema

    config = pydantic.ConfigDict(title=func.__name__ or "unknown")
    config_wrapper = ConfigWrapper(config)
    gen_schema = _generate_schema.GenerateSchema(config_wrapper)

    try:
        sig = inspect.signature(func)
    except ValueError:
        sig = inspect.signature(lambda: None)

    type_hints = _typing_extra.get_function_type_hints(func)
    try:
        from pydantic_ai._function_schema import function_schema

        # Create a wrapper function without excluded parameters
        if exclude_types:
            # Create a new signature without excluded parameters
            orig_sig = sig
            filtered_params = []
            for param in orig_sig.parameters.values():
                # Get parameter annotation
                if param.annotation is orig_sig.empty:
                    annotation = Any
                else:
                    annotation = type_hints.get(param.name, param.annotation)
                # Skip excluded types
                if not any(
                    _types_match(annotation, exclude_type)
                    for exclude_type in exclude_types
                ):
                    filtered_params.append(param)

            # Create new signature
            new_sig = orig_sig.replace(parameters=filtered_params)

            # Check if this is a bound method - they don't support signature modification
            if hasattr(func, "__func__") and hasattr(func, "__self__"):
                msg = (
                    f"Cannot filter parameters from bound method '{func.__qualname__}'. "
                    f"Bound methods don't support signature modification. "
                    f"Consider using a closure function instead:\n"
                )
                raise ValueError(msg)

            # Type ignore for dynamic signature modification
            func.__signature__ = new_sig  # type: ignore[attr-defined]
        # Use pydantic-ai's function_schema
        pydantic_ai_schema = function_schema(func, schema_generator_cls)
        # Convert to our format - now we can use the rich JSON schema directly
        json_schema = pydantic_ai_schema.json_schema
        # Create ToolParameters directly from the rich JSON schema
        parameters: ToolParameters = {
            "type": "object",
            "properties": json_schema.get("properties", {}),
        }

        if "required" in json_schema:
            parameters["required"] = json_schema["required"]
        # Copy over any extra fields like $defs
        for key, value in json_schema.items():
            if key not in {"type", "properties", "required"}:
                parameters[key] = value
        required_fields = json_schema.get("required", [])

    except ImportError:
        # Fallback to original approach if pydantic-ai not available

        # Parse docstring for parameter descriptions
        import docstring_parser

        docstring = docstring_parser.parse(func.__doc__ or "")
        param_descriptions = {
            p.arg_name: p.description for p in docstring.params if p.description
        }

        fields: dict[str, core_schema.TypedDictField] = {}
        fallback_required_fields: list[str] = []

        # Process parameters
        for name, param in sig.parameters.items():
            # Skip self parameter
            if name == "self" and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                continue

            # Skip *args and **kwargs
            if param.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue

            # Get parameter annotation
            if param.annotation is sig.empty:
                annotation = Any
            else:
                annotation = type_hints.get(name, param.annotation)

            # Skip excluded types
            if any(
                _types_match(annotation, exclude_type) for exclude_type in exclude_types
            ):
                continue

            # Create field info
            required = param.default is inspect.Parameter.empty
            if required:
                field_info = FieldInfo.from_annotation(annotation)  # pyright: ignore[reportArgumentType]
                fallback_required_fields.append(name)
            else:
                field_info = FieldInfo.from_annotated_attribute(annotation, param.default)  # pyright: ignore[reportArgumentType]

            # Add description from docstring if available
            if name in param_descriptions:
                field_info.description = param_descriptions[name]

            # Create typed dict field
            from pydantic._internal import _decorators

            td_field = gen_schema._generate_td_field_schema(
                name,
                field_info,
                _decorators.DecoratorInfos(),
                required=required,
            )
            fields[name] = td_field

        # Create typed dict schema
        core_config = config_wrapper.core_config(None)
        core_config["extra_fields_behavior"] = "forbid"

        schema_dict = core_schema.typed_dict_schema(
            fields,
            config=core_config,
        )

        # Generate JSON schema - this may fail for complex types
        try:
            json_schema = schema_generator_cls().generate(schema_dict)
            # Extract parameters
            fallback_parameters: ToolParameters = {
                "type": "object",
                "properties": json_schema.get("properties", {}),
            }

            if fallback_required_fields:
                fallback_parameters["required"] = fallback_required_fields
            parameters = fallback_parameters
        except Exception:  # noqa: BLE001
            # If JSON schema generation fails, fall back to original implementation
            return _create_schema_simple(
                func, name_override, description_override, exclude_types
            )

    # Handle return type
    function_type = determine_function_type(func)
    return_hint = type_hints.get("return", Any)

    if function_type in {"sync_generator", "async_generator"}:
        element_type = next(
            (t for t in get_args(return_hint) if t is not type(None)),
            Any,
        )
        returns_dct = {
            "type": "array",
            "items": resolve_type_annotation(element_type, is_parameter=False),
        }
    else:
        returns = resolve_type_annotation(return_hint, is_parameter=False)
        returns_dct = dict(returns)  # type: ignore[arg-type]

    # Get description
    import docstring_parser

    docstring = docstring_parser.parse(func.__doc__ or "")

    return FunctionSchema(
        name=name_override or getattr(func, "__name__", "unknown") or "unknown",
        description=description_override or docstring.short_description,
        parameters=parameters,
        required=required_fields,
        returns=returns_dct,
    )


def _create_schema_simple(
    func: Callable[..., Any],
    name_override: str | None,
    description_override: str | None,
    exclude_types: list[type],
) -> FunctionSchema:
    """Original schema creation implementation."""
    import docstring_parser

    # Parse function signature and docstring
    sig = inspect.signature(func)
    docstring = docstring_parser.parse(func.__doc__ or "")

    # Get clean type hints without extras
    try:
        hints = typing.get_type_hints(func, localns=locals())
    except NameError:
        msg = "Unable to resolve type hints for function %s, skipping"
        logger.warning(msg, getattr(func, "__name__", "unknown"))
        hints = {}

    parameters: ToolParameters = {"type": "object", "properties": {}}
    required: list[str] = []
    params = list(sig.parameters.items())
    skip_first = (
        inspect.isfunction(func)
        and not inspect.ismethod(func)
        and params
        and params[0][0] == "self"
    )

    for i, (name, param) in enumerate(sig.parameters.items()):
        # Skip the first parameter for bound methods
        if skip_first and i == 0:
            continue
        if param.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        # Skip parameters with excluded types
        param_type = hints.get(name, Any)
        if any(_types_match(param_type, exclude_type) for exclude_type in exclude_types):
            continue

        param_doc = next(
            (p.description for p in docstring.params if p.arg_name == name),
            None,
        )

        parameters["properties"][name] = resolve_type_annotation(
            param_type,
            description=param_doc,
            default=param.default,
            is_parameter=True,
        )

        if param.default is inspect.Parameter.empty:
            required.append(name)

    # Add required fields to parameters if any exist
    if required:
        parameters["required"] = required

    # Handle return type with is_parameter=False
    function_type = determine_function_type(func)
    return_hint = hints.get("return", Any)

    if function_type in {"sync_generator", "async_generator"}:
        element_type = next(
            (t for t in get_args(return_hint) if t is not type(None)),
            Any,
        )
        prop = resolve_type_annotation(element_type, is_parameter=False)
        returns_dct = {"type": "array", "items": prop}
    else:
        returns = resolve_type_annotation(return_hint, is_parameter=False)
        returns_dct = dict(returns)  # type: ignore[arg-type]

    return FunctionSchema(
        name=name_override or getattr(func, "__name__", "unknown") or "unknown",
        description=description_override or docstring.short_description,
        parameters=parameters,
        required=required,
        returns=returns_dct,
    )


if __name__ == "__main__":

    def get_weather(
        location: str,
        unit: Literal["C", "F"] = "C",
        detailed: bool = False,
    ) -> dict[str, str | float]:
        """Get the weather for a location.

        Args:
            location: City or address to get weather for
            unit: Temperature unit (Celsius or Fahrenheit)
            detailed: Include extended forecast
        """
        return {"temp": 22.5, "conditions": "sunny"}

    # Create schema and executable function
    schema = create_schema(get_weather)
    signature = schema.to_python_signature()
    print(signature)
    text = signature.format()
    print(text)
