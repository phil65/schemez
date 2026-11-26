import asyncio
from collections.abc import AsyncIterator, Callable  # noqa: TC003
import concurrent.futures
from typing import TYPE_CHECKING, Any, TypeVar, assert_never, overload

from schemez.functionschema import create_schema


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from schemez.functionschema import FunctionSchema
    from schemez.functionschema.helpers import FunctionType

T_co = TypeVar("T_co", covariant=True)


def _run_coroutine(make_coro: Callable[[], Any]) -> Any:
    """Run a coroutine to completion from sync code, also inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coro())
    # A running loop can't be re-entered: use a fresh loop on a worker thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(make_coro())).result()


class ExecutableFunction[T_co]:
    """Wrapper for executing functions with different calling patterns."""

    def __init__(
        self,
        schema: FunctionSchema,
        func: (
            Callable[..., T_co]
            | Callable[..., Generator[T_co]]
            | Callable[..., AsyncGenerator[T_co]]
            | Callable[..., AsyncIterator[T_co]]
        ),
    ) -> None:
        """Initialize with schema and function.

        Args:
            schema: OpenAI function schema
            func: The actual function to execute
        """
        from schemez.functionschema import determine_function_type

        self.schema = schema
        self.func = func
        self.function_type: FunctionType = determine_function_type(self.func)

    def run(self, *args: Any, **kwargs: Any) -> T_co | list[T_co]:
        """Run the function synchronously.

        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Either a single result or list of results for generators

        Inside a running event loop, coroutines run on a worker thread's own loop
        (blocking the caller); use `arun` from async code.
        """
        func: Callable[..., Any] = self.func
        match self.function_type:
            case "sync":
                return func(*args, **kwargs)
            case "async":
                return _run_coroutine(lambda: func(*args, **kwargs))
            case "sync_generator":
                return list(func(*args, **kwargs))
            case "async_generator":
                return _run_coroutine(lambda: self._collect_async_gen(*args, **kwargs))
            case _ as unreachable:
                assert_never(unreachable)

    async def _collect_async_gen(self, *args: Any, **kwargs: Any) -> list[T_co]:
        """Collect async generator results into a list.

        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            List of collected results
        """
        func: Callable[..., Any] = self.func
        return [x async for x in func(*args, **kwargs)]

    async def arun(self, *args: Any, **kwargs: Any) -> T_co | list[T_co]:
        """Run the function asynchronously.

        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Function result or list of results for generators

        Raises:
            ValueError: If the function type is unknown
        """
        func: Callable[..., Any] = self.func
        match self.function_type:
            case "sync":
                return func(*args, **kwargs)
            case "async":
                return await func(*args, **kwargs)
            case "sync_generator":
                return list(func(*args, **kwargs))
            case "async_generator":
                return [x async for x in func(*args, **kwargs)]
            case _:
                msg = f"Unknown function type: {self.function_type}"
                raise ValueError(msg)

    async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[T_co]:
        """Stream results from the function.

        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Yields:
            Individual results as they become available

        Raises:
            ValueError: If the function type is unknown
        """
        func: Callable[..., Any] = self.func
        match self.function_type:
            case "sync_generator":
                for x in func(*args, **kwargs):
                    yield x
            case "async_generator":
                async for x in func(*args, **kwargs):
                    yield x
            case "sync":
                yield func(*args, **kwargs)
            case "async":
                yield await func(*args, **kwargs)
            case _ as unreachable:
                assert_never(unreachable)


@overload
def create_executable[T_co](
    func: Callable[..., Generator[T_co]],
) -> ExecutableFunction[T_co]: ...


@overload
def create_executable[T_co](
    func: Callable[..., AsyncGenerator[T_co]],
) -> ExecutableFunction[T_co]: ...


@overload
def create_executable[T_co](
    func: Callable[..., T_co],
) -> ExecutableFunction[T_co]: ...


def create_executable[T_co](
    func: (
        Callable[..., T_co] | Callable[..., Generator[T_co]] | Callable[..., AsyncGenerator[T_co]]
    ),
) -> ExecutableFunction[T_co]:
    """Create an executable function wrapper with schema.

    Args:
        func: Function to wrap

    Returns:
        Executable wrapper with schema
    """
    schema = create_schema(func)
    return ExecutableFunction[T_co](schema, func)


if __name__ == "__main__":
    from typing import Literal

    def get_weather(
        location: str,
        unit: Literal["C", "F"] = "C",
        detailed: bool = False,
    ) -> dict[str, str | float]:
        return {"temp": 22.5, "conditions": "sunny"}

    exe = create_executable(get_weather)
    # Execute the function
    result = exe.run("London", unit="C")
    print("\nFunction Result:")
    print(result)
