"""marvin_impl.create_dataclass class cache: per-thread placeholders, keyed with $defs."""

from __future__ import annotations

import threading
from typing import Any, ForwardRef

import pytest

from schemez.schema_to_type import marvin_impl


SCHEMA = {"type": "object", "title": "CacheProbe", "properties": {"a": {"type": "integer"}}}


def test_failed_build_leaves_no_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = {**SCHEMA, "title": "FailOnce"}
    original = marvin_impl.schema_to_type

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(marvin_impl, "schema_to_type", boom)
    with pytest.raises(RuntimeError):
        marvin_impl.create_dataclass(schema)
    monkeypatch.setattr(marvin_impl, "schema_to_type", original)
    cls = marvin_impl.create_dataclass(schema)
    assert isinstance(cls, type)


def test_ref_schemas_are_part_of_the_key() -> None:
    schema = {"type": "object", "title": "RefProbe", "properties": {"x": {"$ref": "#/$defs/X"}}}
    a = marvin_impl.create_dataclass(schema, schemas={"$defs": {"X": {"type": "integer"}}})
    b = marvin_impl.create_dataclass(schema, schemas={"$defs": {"X": {"type": "string"}}})
    assert a is not b


def test_other_thread_does_not_see_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = {**SCHEMA, "title": "ThreadProbe"}
    original = marvin_impl.schema_to_type
    entered, release = threading.Event(), threading.Event()

    def slow(*args: Any, **kwargs: Any) -> Any:
        if threading.current_thread().name == "slow-builder":
            entered.set()
            release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(marvin_impl, "schema_to_type", slow)
    worker = threading.Thread(
        target=marvin_impl.create_dataclass, args=(schema,), name="slow-builder"
    )
    worker.start()
    assert entered.wait(5)
    try:
        result = marvin_impl.create_dataclass(schema)
    finally:
        release.set()
        worker.join(5)
    assert not isinstance(result, ForwardRef)
    assert isinstance(result, type)


def test_cache_is_bounded() -> None:
    for i in range(marvin_impl._MAX_CACHED_CLASSES + 10):
        marvin_impl.create_dataclass({**SCHEMA, "title": f"Bound{i}"})
    assert len(marvin_impl._classes) <= marvin_impl._MAX_CACHED_CLASSES
