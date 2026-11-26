"""Callables whose annotations use names imported only under `TYPE_CHECKING` (PEP 649).

Without `from __future__ import annotations`, evaluating such annotations raises
`NameError`. schemez resolves what it can and lets unresolvable names degrade to `Any`.
"""

import inspect
from typing import TYPE_CHECKING, Literal

import pytest

from schemez import Schema, create_schema
from schemez.bind_kwargs import BoundFunction
from schemez.convert import get_function_model
from schemez.functionschema.schema_generators import create_constructor_schema
from schemez.introspection import get_resolved_hints, get_signature


if TYPE_CHECKING:
    from decimal import Decimal
    from fractions import Fraction

MODES: list[Literal["simple", "openai", "jsonschema"]] = ["simple", "openai", "jsonschema"]


def with_late_import(amount: Decimal, note: str = "x") -> Decimal:
    """Use a name that is imported only after the definition.

    Args:
        amount: The amount.
        note: A note.
    """
    return amount


def with_missing_name(value: Fraction, values: list[Fraction], n: int = 1) -> Fraction | None:
    """Use a name that never exists at runtime.

    Args:
        value: The value.
        values: More values.
        n: A count.
    """
    return None


class Context:
    """Context type to exclude."""


class Tools:
    def tool(self, ctx: Context, value: Fraction, n: int = 1) -> str:
        """Bound method with an excluded parameter and a missing name.

        Args:
            ctx: Excluded.
            value: The value.
            n: A count.
        """
        return str(n)


class Ctor:
    def __init__(self, value: Fraction, n: int = 1) -> None:
        self.n = n


# Imported after the definitions: resolvable when the schema is created.
from decimal import Decimal  # noqa: E402


def test_annotations_raise_without_help() -> None:
    with pytest.raises(NameError):
        inspect.signature(with_missing_name)


@pytest.mark.parametrize("mode", MODES)
def test_resolvable_at_call_time(mode: Literal["simple", "openai", "jsonschema"]) -> None:
    schema = create_schema(with_late_import, mode=mode)
    amount = schema.parameters["properties"]["amount"]
    assert amount.get("type") == "number" or {"type": "number"} in amount.get("anyOf", [])
    assert amount["description"] == "The amount."
    assert schema.required == ["amount"]
    assert get_resolved_hints(with_late_import)["amount"] is Decimal


@pytest.mark.parametrize("mode", MODES)
def test_missing_name_degrades_to_any(mode: Literal["simple", "openai", "jsonschema"]) -> None:
    schema = create_schema(with_missing_name, mode=mode)
    props = schema.parameters["properties"]
    assert "type" not in props["value"]  # unconstrained (Any)
    assert props["value"]["description"] == "The value."
    assert props["values"]["type"] == "array"  # list[Fraction] -> list[Any]
    assert props["n"]["type"] == "integer"
    assert schema.required == ["value", "values"]


def test_missing_name_signature_and_hints() -> None:
    sig = get_signature(with_missing_name)
    assert str(sig) == ("(value: Any, values: list[typing.Any], n: int = 1) -> Any | None")
    hints = get_resolved_hints(with_missing_name)
    assert hints["n"] is int


@pytest.mark.parametrize("mode", MODES)
def test_bound_method_with_exclusion_and_missing_name(
    mode: Literal["simple", "openai", "jsonschema"],
) -> None:
    method = Tools().tool
    schema = create_schema(method, mode=mode, exclude_types=[Context])
    assert set(schema.parameters["properties"]) == {"value", "n"}
    assert "__signature__" not in vars(Tools.tool)  # user callable not mutated


def test_function_model_and_constructor() -> None:
    model = get_function_model(with_missing_name)
    assert set(model.model_fields) == {"value", "values", "n"}
    assert model(value=1, values=[2]).n == 1  # ty: ignore[unknown-argument]
    assert Schema.for_function(with_late_import).model_fields["amount"].annotation is Decimal
    schema = create_constructor_schema(Ctor)
    assert set(schema.parameters["properties"]) == {"value", "n"}


def test_bound_function() -> None:
    bound = BoundFunction(with_missing_name, n=2)
    assert list(inspect.signature(bound).parameters) == ["value", "values"]
    assert "n" not in bound.__annotations__
