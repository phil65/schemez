"""Configuration models for Schemez."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal, Self

import anyenv
from pydantic import BaseModel, ConfigDict
import upath


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling_agent.agent.agent import AgentType
    from llmling_agent.models.content import BaseContent


StrPath = str | os.PathLike[str]
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
    def from_yaml(cls, content: str, inherit_path: StrPath | None = None) -> Self:
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
    def from_vision_llm_sync(
        cls,
        file_content: bytes,
        source_type: SourceType = "pdf",
        model: str = "google-gla:gemini-2.0-flash",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
        provider: AgentType = "pydantic_ai",
    ) -> Self:
        """Create a schema model from a document using AI.

        Args:
            file_content: The document content to create a schema from
            source_type: The type of the document
            model: The AI model to use for schema extraction
            system_prompt: The system prompt to use for schema extraction
            user_prompt: The user prompt to use for schema extraction
            provider: The provider to use for schema extraction

        Returns:
            A new schema model class based on the document
        """
        from llmling_agent import Agent, ImageBase64Content, PDFBase64Content

        if source_type == "pdf":
            content: BaseContent = PDFBase64Content.from_bytes(file_content)
        else:
            content = ImageBase64Content.from_bytes(file_content)
        agent = Agent[None](  # type:ignore[var-annotated]
            model=model,
            system_prompt=system_prompt.format(name=cls.__name__),
            provider=provider,
        ).to_structured(cls)
        chat_message = anyenv.run_sync(agent.run(user_prompt, content))
        return chat_message.content

    @classmethod
    async def from_vision_llm(
        cls,
        file_content: bytes,
        source_type: SourceType = "pdf",
        model: str = "google-gla:gemini-2.0-flash",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
        provider: AgentType = "pydantic_ai",
    ) -> Self:
        """Create a schema model from a document using AI.

        Args:
            file_content: The document content to create a schema from
            source_type: The type of the document
            model: The AI model to use for schema extraction
            system_prompt: The system prompt to use for schema extraction
            user_prompt: The user prompt to use for schema extraction
            provider: The provider to use for schema extraction

        Returns:
            A new schema model class based on the document
        """
        from llmling_agent import Agent, ImageBase64Content, PDFBase64Content

        if source_type == "pdf":
            content: BaseContent = PDFBase64Content.from_bytes(file_content)
        else:
            content = ImageBase64Content.from_bytes(file_content)
        agent = Agent[None](  # type:ignore[var-annotated]
            model=model,
            system_prompt=system_prompt.format(name=cls.__name__),
            provider=provider,
        ).to_structured(cls)
        chat_message = await agent.run(user_prompt, content)
        return chat_message.content

    @classmethod
    def from_llm_sync(
        cls,
        text: str,
        model: str = "google-gla:gemini-2.0-flash",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
        provider: AgentType = "pydantic_ai",
    ) -> Self:
        """Create a schema model from a text snippet using AI.

        Args:
            text: The text to create a schema from
            model: The AI model to use for schema extraction
            system_prompt: The system prompt to use for schema extraction
            user_prompt: The user prompt to use for schema extraction
            provider: The provider to use for schema extraction

        Returns:
            A new schema model class based on the document
        """
        from llmling_agent import Agent

        agent = Agent[None](  # type:ignore[var-annotated]
            model=model,
            system_prompt=system_prompt.format(name=cls.__name__),
            provider=provider,
        ).to_structured(cls)
        chat_message = anyenv.run_sync(agent.run(user_prompt, text))
        return chat_message.content

    @classmethod
    async def from_llm(
        cls,
        text: str,
        model: str = "google-gla:gemini-2.0-flash",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
        provider: AgentType = "pydantic_ai",
    ) -> Self:
        """Create a schema model from a text snippet using AI.

        Args:
            text: The text to create a schema from
            model: The AI model to use for schema extraction
            system_prompt: The system prompt to use for schema extraction
            user_prompt: The user prompt to use for schema extraction
            provider: The provider to use for schema extraction

        Returns:
            A new schema model class based on the document
        """
        from llmling_agent import Agent

        agent = Agent[None](  # type:ignore[var-annotated]
            model=model,
            system_prompt=system_prompt.format(name=cls.__name__),
            provider=provider,
        ).to_structured(cls)
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

    def model_dump_yaml(self) -> str:
        """Dump configuration to YAML string."""
        import yamling

        return yamling.dump_yaml(self.model_dump(exclude_none=True))

    def save(self, path: StrPath, overwrite: bool = False) -> None:
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
