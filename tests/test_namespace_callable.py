"""NamespaceCallable passes results and errors through unchanged."""

from __future__ import annotations

import pytest

from schemez.code_generation.namespace_callable import NamespaceCallable


@pytest.mark.parametrize("value", [0, [], "", False, {}])
async def test_falsy_results_are_returned_as_is(value: object) -> None:
    def tool() -> object:
        return value

    assert await NamespaceCallable(tool)() == value
    assert type(await NamespaceCallable(tool)()) is type(value)


async def test_none_result_gets_placeholder() -> None:
    def tool() -> None:
        return None

    assert await NamespaceCallable(tool)() == "Operation completed successfully"


async def test_exceptions_propagate() -> None:
    async def boom() -> int:
        raise ValueError("db down")

    with pytest.raises(ValueError, match="db down"):
        await NamespaceCallable(boom)()


async def test_awaitable_result_is_awaited() -> None:
    async def inner() -> int:
        return 5

    def tool():  # returns a coroutine without being a coroutine function
        return inner()

    assert await NamespaceCallable(tool)() == 5  # noqa: PLR2004
