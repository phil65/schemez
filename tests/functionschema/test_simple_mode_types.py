"""Simple-mode schemas for postponed annotations, unions, literals and recursion."""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field

from schemez import create_schema


@dataclasses.dataclass
class Cfg:
    retries: int
    tags: list[str]
    ratio: float = 0.5


class TD(TypedDict):
    n: int
    label: NotRequired[str]


class Color(enum.IntEnum):
    RED = 1
    GREEN = 2


class Tree(BaseModel):
    name: str
    kids: list[Tree] = []


@dataclasses.dataclass
class Node:
    value: int
    next: Node | None = None


class Inner(BaseModel):
    depth: int = Field(3, description="How deep to go")


def _props(func: Any) -> dict[str, Any]:
    return create_schema(func).parameters["properties"]


def test_postponed_dataclass_and_typeddict_fields() -> None:
    def f(cfg: Cfg, td: TD) -> None:
        pass

    props = _props(f)
    assert props["cfg"]["properties"] == {
        "retries": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "ratio": {"type": "number", "default": 0.5},
    }
    assert props["cfg"]["required"] == ["retries", "tags"]
    assert props["td"]["properties"] == {"n": {"type": "integer"}, "label": {"type": "string"}}
    assert props["td"]["required"] == ["n"]


def test_union_becomes_any_of() -> None:
    def f(a: int | str, b: str | None = None, c: int | None = 1) -> None:
        pass

    props = _props(f)
    assert props["a"] == {"anyOf": [{"type": "integer"}, {"type": "string"}]}
    # An optional parameter with a None default stays a plain type (it may be omitted).
    assert props["b"] == {"type": "string"}
    assert props["c"] == {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 1}


def test_literal_and_enum_types_follow_values() -> None:
    def f(
        b: Literal[1, 2],
        c: Color,
        s: Literal["x", "y"],
        mixed: Literal[1, "a"],
        flag: Literal[True] = True,
    ) -> None:
        pass

    props = _props(f)
    assert props["b"] == {"type": "integer", "enum": [1, 2]}
    assert props["c"] == {"type": "integer", "enum": [1, 2]}
    assert props["s"] == {"type": "string", "enum": ["x", "y"]}
    assert props["mixed"] == {
        "anyOf": [{"type": "integer", "enum": [1]}, {"type": "string", "enum": ["a"]}]
    }
    assert props["flag"] == {"type": "boolean", "enum": [True], "default": True}


def test_tuple_any_and_nested_field_metadata() -> None:
    def f(d: tuple[int, str], v: tuple[int, ...], e: Any, inner: Inner) -> None:
        pass

    props = _props(f)
    assert props["d"] == {
        "type": "array",
        "prefixItems": [{"type": "integer"}, {"type": "string"}],
        "minItems": 2,
        "maxItems": 2,
    }
    assert props["v"] == {"type": "array", "items": {"type": "integer"}}
    assert props["e"] == {}
    assert props["inner"]["properties"]["depth"] == {
        "type": "integer",
        "description": "How deep to go",
        "default": 3,
    }


def test_self_reference_uses_defs() -> None:
    def f(tree: Tree, node: Node) -> None:
        pass

    schema = create_schema(f)
    params = schema.parameters
    assert params["properties"]["tree"] == {"$ref": "#/$defs/Tree"}
    assert params["properties"]["node"] == {"$ref": "#/$defs/Node"}
    defs = params["$defs"]
    assert defs["Tree"]["properties"]["kids"] == {
        "type": "array",
        "items": {"$ref": "#/$defs/Tree"},
        "default": [],
    }
    assert defs["Node"]["properties"]["next"] == {"$ref": "#/$defs/Node"}


def test_recursive_return_type_carries_defs() -> None:
    def f() -> Tree:
        raise NotImplementedError

    returns = create_schema(f).returns
    assert returns["$ref"] == "#/$defs/Tree"
    assert "Tree" in returns["$defs"]


def test_parameter_model_from_any_of() -> None:
    def f(a: int | str, b: Literal[1, 2], e: Any) -> None:
        pass

    fields = create_schema(f).create_parameter_model().model_fields
    assert fields["a"].annotation == int | str
    assert fields["b"].annotation == Literal[1, 2]
    assert fields["e"].annotation is Any


def test_unresolvable_annotation_keeps_the_others() -> None:
    def f(name: str, other: NotImported, count: int = 1) -> None:  # noqa: F821  # ty: ignore[unresolved-reference]
        pass

    props = _props(f)
    assert props["name"] == {"type": "string"}
    assert props["other"] == {}
    assert props["count"] == {"type": "integer", "default": 1}
