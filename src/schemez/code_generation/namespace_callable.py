"""Namespace callable wrapper for tools."""

from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING, Any

from schemez.helpers import get_object_name
from schemez.introspection import get_signature


if TYPE_CHECKING:
    from collections.abc import Callable

    from schemez.code_generation.tool_code_generator import ToolCodeGenerator


@dataclass
class NamespaceCallable:
    """Wrapper for tool functions with proper repr and call interface."""

    callable: Callable[..., Any]
    """The callable function to execute."""

    name_override: str | None = None
    """Override name for the callable, defaults to callable.__name__."""

    def __post_init__(self) -> None:
        """Set function attributes for introspection."""
        self.__name__ = self.name_override or get_object_name(self.callable, "unknown")
        self.__doc__ = self.callable.__doc__ or ""

    @property
    def name(self) -> str:
        """Get the effective name of the callable."""
        return self.name_override or get_object_name(self.callable, "unknown")

    @classmethod
    def from_generator(cls, generator: ToolCodeGenerator) -> NamespaceCallable:
        """Create a NamespaceCallable from a ToolCodeGenerator.

        Args:
            generator: The generator to wrap

        Returns:
            NamespaceCallable instance
        """
        if generator.callable is None:
            msg = f"Callable required for NamespaceCallable {generator.name!r}. "
            raise ValueError(msg)
        return cls(generator.callable, generator.name_override)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the wrapped callable.

        Results (also falsy ones) and exceptions pass through unchanged; only `None`
        becomes a short confirmation text.
        """
        result = self.callable(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return "Operation completed successfully" if result is None else result

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return f"NamespaceCallable(name='{self.name}')"

    def __str__(self) -> str:
        """Return readable string representation."""
        return f"<tool: {self.name}>"

    @property
    def signature(self) -> str:
        """Get function signature for debugging."""
        try:
            sig = get_signature(self.callable)
        except ValueError, TypeError:
            return f"{self.name}(...)"
        else:
            return f"{self.name}{sig}"
