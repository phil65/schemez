"""Configuration models for Schemez."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict
import upath

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
    ) -> str:
        """Dump configuration to YAML string."""
        import yamling

        text = self.model_dump(
            exclude_none=exclude_none,
            exclude_defaults=exclude_defaults,
            exclude_unset=exclude_unset,
        )
        return yamling.dump_yaml(text)

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
    model = Schema.from_json_schema(schema)
    import devtools

    devtools.debug(model.__fields__)
