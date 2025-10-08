from __future__ import annotations

from collections.abc import AsyncGenerator, Generator  # noqa: TC003

from py2openai.executable import create_executable


def test_sync_generator_execution() -> None:
    """Test sync generator functions are properly schematized."""

    def gen(n: int) -> Generator[int]:
        yield from range(n)

    # Test execution
    exe = create_executable(gen)
    assert exe.run(3) == [0, 1, 2]


def test_async_generator_sync_execution() -> None:
    """Test synchronous execution of async generators."""

    async def agen(n: int) -> AsyncGenerator[str]:
        """Generate n strings."""
        for i in range(n):
            yield str(i)

    exe = create_executable(agen)
    # Test that we can run an async generator synchronously
    assert exe.run(3) == ["0", "1", "2"]


async def test_async_generator_execution() -> None:
    """Test async generator functions are properly schematized."""

    async def agen(n: int) -> AsyncGenerator[str]:
        for i in range(n):
            yield str(i)

    # Test execution
    exe = create_executable(agen)
    assert await exe.arun(3) == ["0", "1", "2"]

    collected = [item async for item in exe.astream(3)]
    assert collected == ["0", "1", "2"]
