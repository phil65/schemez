"""json_schema_to_base_model on common JSON schema shapes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ValidationError
import pytest

from schemez import create_schema
from schemez.schema_to_type.simple_impl import json_schema_to_base_model
from tests.functionschema.test_simple_mode_types import Tree  # noqa: TC001


def test_type_list_and_numeric_enum() -> None:
    model = json_schema_to_base_model({
        "properties": {
            "name": {"type": ["string", "null"], "description": "The name"},
            "level": {"enum": [1, 2]},
            "color": {"type": "string", "enum": ["red", "blue"]},
        },
        "required": ["name", "level"],
    })
    fields = model.model_fields
    assert fields["name"].annotation == str | None
    assert fields["name"].description == "The name"
    assert fields["level"].annotation == Literal[1, 2]
    assert model(name=None, level=2).level == 2  # noqa: PLR2004
    with pytest.raises(ValidationError):
        model(name="x", level=3)
    assert model(name="x", level=1, color="red").color.value == "red"


def test_any_of_and_refs() -> None:
    model = json_schema_to_base_model({
        "properties": {
            "value": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
            "pet": {"$ref": "#/$defs/Pet"},
        },
        "required": ["value", "pet"],
        "$defs": {"Pet": {"type": "object", "properties": {"name": {"type": "string"}}}},
    })
    obj = model(value=3, pet={"name": "Rex"})
    assert obj.value == 3  # noqa: PLR2004
    assert obj.pet.name == "Rex"
    with pytest.raises(ValidationError):
        model(value=[1], pet={"name": "Rex"})


def test_recursive_ref_from_simple_mode() -> None:
    def f(tree: Tree, extra: Any = None) -> None:
        pass

    model = json_schema_to_base_model(dict(create_schema(f).parameters))
    assert model.model_fields["extra"].annotation is Any
    obj = model(tree={"name": "a", "kids": [{"name": "b"}]})
    assert obj.tree.kids[0].name == "b"


def test_route_code_only_names_builtins() -> None:
    from schemez.code_generation.route_helpers import generate_func_code

    model = json_schema_to_base_model({
        "properties": {"level": {"enum": [1, 2]}, "name": {"type": "string"}},
        "required": ["level", "name"],
    })
    code = generate_func_code(model.model_fields)
    assert "level: Any, name: str" in code
