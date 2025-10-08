"""Type definitions for tool executor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from pydantic import BaseModel


class ToolHandler(Protocol):
    """Protocol for user-provided tool handler."""

    async def __call__(self, method_name: str, input_props: BaseModel) -> str:
        """Process a tool call.

        Args:
            method_name: Name of the tool being called (e.g., "get_weather")
            input_props: Validated input model instance

        Returns:
            String result from the tool execution

        Raises:
            Exception: If tool execution fails
        """
        ...
