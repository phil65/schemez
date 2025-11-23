"""Configuration models for Schemez."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict
import upath

from schemez.generators import SchemaDataGenerator
from schemez.helpers import model_to_python_code


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling_agent.models.content import BaseContent
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

    def merge(self, other: Self) -> Self:
        """Merge with another instance by overlaying its non-None values."""
        from schemez.helpers import merge_models

        return merge_models(self, other)

    @classmethod
    def from_yaml(
        cls, content: str, inherit_path: JoinablePathLike | None = None
    ) -> Self:
        """Create from YAML string."""
        import yamling

        data = yamling.load_yaml(content, resolve_inherit=inherit_path or False)
        return cls.model_validate(data)

    @classmethod
    def for_function(
        cls, func: Callable[..., Any], *, name: str | None = None
    ) -> type[Schema]:
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
    async def from_vision_llm(
        cls,
        file_content: bytes,
        source_type: SourceType = "pdf",
        model: str = "google-gla:gemini-2.0-flash",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
    ) -> Self:
        """Create a schema model from a document using AI.

        Args:
            file_content: The document content to create a schema from
            source_type: The type of the document
            model: The AI model to use for schema extraction
            system_prompt: The system prompt to use for schema extraction
            user_prompt: The user prompt to use for schema extraction

        Returns:
            A new schema model class based on the document
        """
        from llmling_agent import Agent, ImageBase64Content, PDFBase64Content

        if source_type == "pdf":
            content: BaseContent = PDFBase64Content.from_bytes(file_content)
        else:
            content = ImageBase64Content.from_bytes(file_content)
        prompt = system_prompt.format(name=cls.__name__)
        agent = Agent(model=model, system_prompt=prompt, output_type=cls)
        chat_message = await agent.run(user_prompt, content)
        return chat_message.content

    @classmethod
    async def from_llm(
        cls,
        text: str,
        model: str = "google-gla:gemini-2.0-flash",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
    ) -> Self:
        """Create a schema model from a text snippet using AI.

        Args:
            text: The text to create a schema from
            model: The AI model to use for schema extraction
            system_prompt: The system prompt to use for schema extraction
            user_prompt: The user prompt to use for schema extraction

        Returns:
            A new schema model class based on the document
        """
        from llmling_agent import Agent

        prompt = system_prompt.format(name=cls.__name__)
        agent = Agent(model=model, system_prompt=prompt, output_type=cls)
        chat_message = await agent.run(user_prompt, text)
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
        comments: bool = False,
        sort_keys: bool = True,
        mode: Literal["json", "python"] = "python",
    ) -> str:
        """Dump configuration to YAML string.

        Args:
            exclude_none: Exclude fields with None values
            exclude_defaults: Exclude fields with default values
            exclude_unset: Exclude fields that are not set
            comments: Include descriptions as comments in the YAML output
            sort_keys: Sort keys in the YAML output
            mode: Output mode, either "json" or "python"

        Returns:
            YAML string representation of the model
        """
        import re

        import yamling

        data = self.model_dump(
            exclude_none=exclude_none,
            exclude_defaults=exclude_defaults,
            exclude_unset=exclude_unset,
            mode=mode,
        )
        base_yaml = yamling.dump_yaml(data, sort_keys=sort_keys)
        if not comments:
            return base_yaml

        schema = self.model_json_schema()

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
                        return schema_obj["$defs"][ref_path]
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
                            elif (
                                option.get("type") == "object" and "properties" in option
                            ):
                                return option
                return None

            for i, segment in enumerate(path):
                # Handle properties
                if "properties" in current and segment in current["properties"]:
                    current = current["properties"][segment]

                    # If this is the last segment, return the field directly
                    if i == len(path) - 1:
                        return current

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

        def process_yaml_lines(yaml_lines: list[str]) -> list[str]:
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
                if (
                    ":" in stripped
                    and not stripped.startswith("#")
                    and not stripped.startswith("-")
                ):
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

        # Process the YAML
        yaml_lines = base_yaml.strip().split("\n")
        commented_lines = process_yaml_lines(yaml_lines)

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


if __name__ == "__main__":
    from pydantic import Field

    class Inner(Schema):
        a: int = 0
        """Inner class field"""

    class Outer(Schema):
        b: str = Field(default="", examples=["hello"])
        """Outer string field."""
        inner: Inner | None = Field(default=None, examples=[{"a": 100}])
        """Outer nested field"""

    result = Outer.generate_test_data(mode="maximal").model_dump_yaml(comments=True)
    print(result)
