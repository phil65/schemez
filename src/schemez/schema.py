"""Configuration models for Schemez."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_serializer
import upath

from schemez.generators import SchemaDataGenerator
from schemez.helpers import (
    iter_submodel_types as _iter_submodel_types,
    iter_submodels as _iter_submodels,
    model_to_python_code,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pydantic_ai import UserContent
    from upath.types import JoinablePathLike

    from schemez.helpers import PythonVersionStr


SourceType = Literal["pdf", "image"]

DEFAULT_SYSTEM_PROMPT = "You are a schema extractor for {name} BaseModels."
DEFAULT_USER_PROMPT = "Extract information from this document:"


class Schema(BaseModel):
    """Base class configuration models.

    Provides:
    - Common Pydantic settings
    - YAML serialization
    - Basic merge functionality
    """

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    @model_serializer(mode="wrap")
    def _serialize_with_subclass_fields_first(self, serializer: Any, info: Any) -> dict[str, Any]:
        """Serialize with subclass fields appearing before base class fields."""
        data = serializer(self)

        # Get fields from most derived class to base class
        field_order: list[str] = []
        for cls in type(self).__mro__:
            if cls is BaseModel or not issubclass(cls, BaseModel):
                continue
            # Get fields defined in this specific class
            cls_fields = getattr(cls, "__annotations__", {})
            for field_name in cls_fields:
                if field_name not in field_order and field_name in data:
                    field_order.append(field_name)

        # Reorder the dictionary
        return {k: data[k] for k in field_order if k in data}

    def model_dump_markdown(
        self,
        *,
        template: str | None = None,
        header_level: int = 1,
        include_defaults: bool = True,
        include_examples: bool = True,
        include_constraints: bool = True,
        include_values: bool = True,
        display_mode: Literal["table", "python_code", "yaml"] = "table",
        exclude_none: bool = True,
        exclude_defaults_yaml: bool = False,
        exclude_unset: bool = False,
        indent: int = 2,
        header_style: Literal["default", "pymdownx"] = "default",
        serialization_mode: Literal["json", "python"] = "json",
    ) -> str:
        """Dump model instance to Markdown documentation.

        Args:
            template: Custom Jinja2 template string (uses default if None)
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

        Returns:
            Markdown string documenting the model
        """
        from schemez.markdown import instance_to_markdown

        return instance_to_markdown(
            self,
            template=template,
            header_level=header_level,
            include_defaults=include_defaults,
            include_examples=include_examples,
            include_constraints=include_constraints,
            include_values=include_values,
            display_mode=display_mode,
            exclude_none=exclude_none,
            exclude_defaults_yaml=exclude_defaults_yaml,
            exclude_unset=exclude_unset,
            indent=indent,
            header_style=header_style,
            serialization_mode=serialization_mode,
        )

    @classmethod
    def dump_markdown_schema(
        cls,
        *,
        template: str | None = None,
        header_level: int = 1,
        include_defaults: bool = True,
        include_examples: bool = True,
        include_constraints: bool = True,
        display_mode: Literal["table", "python_code", "yaml"] = "table",
        mode: Literal["minimal", "maximal", "default"] = "default",
        exclude_none: bool = True,
        exclude_defaults: bool = False,
        exclude_unset: bool = False,
        indent: int = 2,
        header_style: Literal["default", "pymdownx"] = "default",
        serialization_mode: Literal["json", "python"] = "json",
    ) -> str:
        """Dump model class schema to Markdown documentation.

        Args:
            template: Custom Jinja2 template string (uses default if None)
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

        Returns:
            Markdown string documenting the model schema
        """
        from schemez.markdown import model_to_markdown

        return model_to_markdown(
            cls,
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
        )

    def merge(self, other: Self) -> Self:
        """Merge with another instance by overlaying its non-None values."""
        from schemez.helpers import merge_models

        return merge_models(self, other)

    def iter_submodels(self, *, recursive: bool = True) -> Iterator[tuple[str, BaseModel]]:
        """Iterate through all nested BaseModel instances in fields.

        Supports fields of type:
        - BaseModel (direct instance)
        - list[BaseModel]
        - dict[str, BaseModel]

        Args:
            recursive: If True, also iterate through submodels of submodels

        Yields:
            Tuples of (path, submodel) where path is like "field", "field[0]", "field['key']"
        """
        return _iter_submodels(self, recursive=recursive)

    @classmethod
    def iter_submodel_types(
        cls, *, recursive: bool = True, include_union_members: bool = True
    ) -> Iterator[tuple[str, type[BaseModel]]]:
        """Iterate through all nested BaseModel types in field annotations.

        Supports field types:
        - BaseModel (direct type)
        - list[BaseModel]
        - dict[str, BaseModel]
        - BaseModel | OtherModel (unions, when include_union_members=True)

        Args:
            recursive: If True, also iterate through submodel types of submodels
            include_union_members: If True, yield each union member separately

        Yields:
            Tuples of (path, model_type) where path is like "field", "field[]", "field{}"
        """
        return _iter_submodel_types(
            cls, recursive=recursive, include_union_members=include_union_members
        )

    @classmethod
    def from_yaml(cls, content: str, inherit_path: JoinablePathLike | None = None) -> Self:
        """Create from YAML string."""
        import yamling

        data = yamling.load_yaml(content, resolve_inherit=inherit_path or False)
        return cls.model_validate(data)

    @classmethod
    def for_function(cls, func: Callable[..., Any], *, name: str | None = None) -> type[Schema]:
        """Create a schema model from a function's signature.

        Args:
            func: The function to create a schema from
            name: Optional name for the model

        Returns:
            A new schema model class based on the function parameters
        """
        from schemez.convert import get_function_model

        return get_function_model(func, name=name)

    @classmethod
    def from_json_schema(cls, json_schema: dict[str, Any]) -> type[Schema]:
        """Create a schema model from a JSON schema.

        Args:
            json_schema: The JSON schema to create a schema from

        Returns:
            A new schema model class based on the JSON schema
        """
        from schemez.schema_to_type import json_schema_to_pydantic_class

        return json_schema_to_pydantic_class(
            json_schema, class_name="GeneratedSchema", base_class=cls
        )

    @classmethod
    async def from_llm(
        cls,
        *prompts: JoinablePathLike | str | UserContent,
        model: str = "google-gla:gemini-2.0-flash",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
    ) -> Self:
        """Create a schema model from a text snippet using AI.

        Args:
            prompts: The text to create a schema from
            model: The AI model to use for schema extraction
            system_prompt: The system prompt to use for schema extraction
            user_prompt: The user prompt to use for schema extraction

        Returns:
            A new schema model class based on the document
        """
        from llmling_agent import Agent

        prompt = system_prompt.format(name=cls.__name__)
        agent = Agent(model=model, system_prompt=prompt, output_type=cls)
        chat_message = await agent.run(user_prompt, prompts)
        return chat_message.content

    @classmethod
    def for_class_ctor(cls, target_cls: type) -> type[Schema]:
        """Create a schema model from a class constructor.

        Args:
            target_cls: The class whose constructor to convert

        Returns:
            A new schema model class based on the constructor parameters
        """
        from schemez.convert import get_ctor_basemodel

        return get_ctor_basemodel(target_cls)

    def model_dump_yaml(
        self,
        exclude_none: bool = True,
        exclude_defaults: bool = False,
        exclude_unset: bool = False,
        indent: int = 2,
        default_flow_style: bool | None = None,
        allow_unicode: bool = True,
        comments: bool = False,
        sort_keys: bool = True,
        mode: Literal["json", "python"] = "python",
    ) -> str:
        """Dump configuration to YAML string.

        Args:
            exclude_none: Exclude fields with None values
            exclude_defaults: Exclude fields with default values
            exclude_unset: Exclude fields that are not set
            indent: Indentation level for YAML output
            default_flow_style: Default flow style for YAML output
            allow_unicode: Allow unicode characters in YAML output
            comments: Include descriptions as comments in the YAML output
            sort_keys: Sort keys in the YAML output
            mode: Output mode, either "json" or "python"

        Returns:
            YAML string representation of the model
        """
        import yamling

        from schemez.commented_yaml import process_yaml_lines

        data = self.model_dump(
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

        schema = self.model_json_schema()
        yaml_lines = base_yaml.strip().split("\n")
        commented_lines = process_yaml_lines(yaml_lines, schema, as_listitem=False, wrapped_in=None)

        return "\n".join(commented_lines)

    def save(self, path: JoinablePathLike, overwrite: bool = False) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the configuration to
            overwrite: Whether to overwrite an existing file

        Raises:
            OSError: If file cannot be written
            ValueError: If path is invalid
        """
        yaml_str = self.model_dump_yaml()
        try:
            file_path = upath.UPath(path)
            if file_path.exists() and not overwrite:
                msg = f"File already exists: {path}"
                raise FileExistsError(msg)  # noqa: TRY301
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(yaml_str)
        except Exception as exc:
            msg = f"Failed to save configuration to {path}"
            raise ValueError(msg) from exc

    @classmethod
    def to_python_code(
        cls,
        *,
        class_name: str | None = None,
        target_python_version: PythonVersionStr | None = None,
        model_type: str = "pydantic.BaseModel",
    ) -> str:
        """Convert this model to Python code asynchronously.

        Args:
            class_name: Optional custom class name for the generated code
            target_python_version: Target Python version for code generation
            model_type: Type of model to generate

        Returns:
            Generated Python code as string
        """
        return model_to_python_code(
            cls,
            class_name=class_name,
            target_python_version=target_python_version,
            model_type=model_type,
        )

    @classmethod
    def generate_test_data(
        cls,
        *,
        seed: int = 0,
        mode: Literal["minimal", "maximal", "default"] = "default",
        validate: bool = True,
    ) -> Self:
        """Generate test data that conforms to this schema.

        Args:
            seed: Seed for deterministic generation (default: 0)
            mode: Generation mode:
                - "minimal": Only required fields, minimum values
                - "maximal": All fields, maximum reasonable values
                - "default": Balanced generation (default)
            validate: Whether to validate the generated data (default: True)

        Returns:
            An instance of this schema populated with generated test data

        Example:
            ```python
            class PersonSchema(Schema):
                name: str
                age: int = 25
                email: str | None = None

            # Generate test data
            person = PersonSchema.generate_test_data(seed=42)
            # Result: PersonSchema(name="abc", age=42, email=None)

            # Generate minimal data (required fields only)
            minimal = PersonSchema.generate_test_data(mode="minimal")
            # Result: PersonSchema(name="a", age=0, email=None)

            # Generate maximal data (all fields populated)
            maximal = PersonSchema.generate_test_data(mode="maximal")
            # Result: PersonSchema(name="abcdefghij", age=1000, email="user0@example.com")
            ```
        """
        json_schema = cls.model_json_schema()
        generator = SchemaDataGenerator(json_schema, seed=seed)

        if mode == "minimal":
            data = generator.generate_minimal()
        elif mode == "maximal":
            data = generator.generate_maximal()
        else:  # default
            data = generator.generate()

        if validate:
            return cls.model_validate(data)
        return cls.model_construct(**data)  # type: ignore[return-value]


class ConfigSchema(Schema):
    @classmethod
    def required_python_packages(cls) -> list[str]:
        """Can get overriden in order to specify required packages."""
        return []


if __name__ == "__main__":
    from pydantic import Field

    class Inner(Schema):
        """A nested model for demonstration."""

        a: int = 0
        """Inner class field"""

    class Outer(Schema):
        """An outer model with nested schema."""

        b: str = Field(default="", examples=["hello"])
        """Outer string field."""
        inner: Inner | None = Field(default=None, examples=[{"a": 100}])
        """Outer nested field"""

    result = Outer.generate_test_data(mode="maximal").model_dump_yaml(comments=True)
    print(result)
