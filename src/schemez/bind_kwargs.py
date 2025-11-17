from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable


class BoundFunction[T]:
    """A function with pre-bound parameters.

    This class wraps a function and binds some parameters to fixed values,
    while updating the signature and docstring to reflect only the remaining
    parameters that can still be provided when calling.
    """

    def __init__(self, func: Callable[..., T], **bound_kwargs: Any):
        """Initialize with a function and parameters to bind.

        Args:
            func: The function to wrap
            **bound_kwargs: Parameters to bind to fixed values

        Raises:
            ValueError: If any parameter name is not in the function signature
        """
        self.func = func
        self.bound_kwargs = bound_kwargs
        self.__name__ = func.__name__
        self.__module__ = func.__module__
        self.__qualname__ = func.__qualname__
        self.__doc__ = remove_kwargs_from_docstring(func.__doc__, self.bound_kwargs)
        self.__annotations__ = self._update_annotations(
            getattr(func, "__annotations__", {})
        )
        self.__signature__ = self._update_signature()

        # Verify all bound kwargs are valid parameters
        sig = inspect.signature(func)
        for param in bound_kwargs:
            if param not in sig.parameters:
                msg = f"Parameter {param!r} not found in signature of {func.__name__}"
                raise ValueError(msg)

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Call the function with the bound parameters.

        Args:
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The return value from the wrapped function
        """
        # Combine bound parameters with provided parameters
        all_kwargs = {**self.bound_kwargs, **kwargs}
        return self.func(*args, **all_kwargs)

    def _update_signature(self) -> inspect.Signature:
        """Create a new signature excluding bound parameters.

        Returns:
            Updated signature without bound parameters
        """
        sig = inspect.signature(self.func)
        parameters = [
            param
            for name, param in sig.parameters.items()
            if name not in self.bound_kwargs
        ]
        return sig.replace(parameters=parameters)

    def _update_annotations(self, annotations: dict[str, Any]) -> dict[str, Any]:
        """Remove bound parameters from annotations.

        Args:
            annotations: Original function annotations

        Returns:
            Updated annotations dictionary
        """
        return {
            name: ann
            for name, ann in annotations.items()
            if name not in self.bound_kwargs and name != "return"
        }


def remove_kwargs_from_docstring(
    docstring: str | None, kwargs: dict[str, Any]
) -> str | None:
    """Update docstring to remove bound parameters.

    Args:
        docstring: Original function docstring
        kwargs: kwargs to remove from docstring

    Returns:
        Updated docstring with bound parameters removed
    """
    if not docstring:
        return docstring

    lines = docstring.splitlines()
    new_lines = []

    # Find the Args section and modify it
    in_args_section = False
    skip_lines = False
    current_param = None

    for line in lines:
        # Check if entering Args section
        if "Args:" in line:
            in_args_section = True
            new_lines.append(line)
            continue
        if in_args_section and line.strip() and not line.startswith(" "):
            in_args_section = False
        if in_args_section and ":" in line:
            # Get parameter name from the line
            param_name = line.strip().split(":", 1)[0].strip()
            if param_name in kwargs:
                skip_lines = True
                current_param = param_name
            else:
                skip_lines = False
                current_param = None
        if in_args_section and current_param and line.strip() and ":" in line.lstrip():
            new_param = line.strip().split(":", 1)[0].strip()
            if new_param != current_param:
                skip_lines = False
                current_param = None

        # Add the line if not skipping
        if not skip_lines:
            new_lines.append(line)

    return "\n".join(new_lines)


if __name__ == "__main__":
    import asyncio

    async def search_db(
        query: str,
        k: int = 5,
        filters: dict[str, list[str]] | None = None,
        min_score: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Search the database for relevant information.

        Args:
            query: Search query text
            k: Number of results to return
            filters: Filters to apply to search
                     2nd line
            min_score: Minimum relevance score

        Returns:
            List of search results

        Example:
            >>> await search_db("quantum computing", k=3)
        """
        print(f"query={query}, k={k}, filters={filters}, min_score={min_score}")
        return [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.8}]

    # Create a bound version
    simple_search = BoundFunction(search_db, k=3, min_score=0.8)
    print("Original function:")
    print(f"Signature: {inspect.signature(search_db)}")
    print(f"Docstring:\n{search_db.__doc__}")

    # Print bound function info
    print("\nBound function:")
    print(f"Signature: {inspect.signature(simple_search)}")
    print(f"Docstring:\n{simple_search.__doc__}")

    # Run both functions to compare
    async def run_test() -> None:
        print("\nCalling original function:")
        result1 = await search_db("quantum computing")
        print(f"Result: {result1}")

        print("\nCalling bound function:")
        result2 = await simple_search("quantum computing")
        print(f"Result: {result2}")

    asyncio.run(run_test())
