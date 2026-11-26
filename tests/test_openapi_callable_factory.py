"""Python types from OpenAPI schemas."""

from typing import Any, get_args

from schemez.openapi.callable_factory import OpenAPICallableFactory


async def _handler(*args: Any) -> dict[str, Any]:
    return {}


def test_additional_properties_true_maps_to_any_values() -> None:
    factory = OpenAPICallableFactory({}, _handler)
    schema = {"type": "object", "additionalProperties": True}
    assert factory.get_type_for_schema(schema) == dict[str, Any]


def test_additional_properties_schema_types_the_values() -> None:
    factory = OpenAPICallableFactory({}, _handler)
    schema = {"type": "object", "additionalProperties": {"type": "integer"}}
    assert get_args(factory.get_type_for_schema(schema).__value__) == (str, int)
