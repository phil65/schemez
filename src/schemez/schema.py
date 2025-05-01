"""Configuration models for Schemez."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, ConfigDict
import upath


if TYPE_CHECKING:
    from collections.abc import Callable


StrPath = str | os.PathLike[str]


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
