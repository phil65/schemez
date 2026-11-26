"""Configuration models for Schemez."""

from enum import Enum
import functools
import operator
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, create_model


TYPE_MAPPING: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def json_schema_to_base_model[TModel: BaseModel = BaseModel](
    schema: dict[str, Any],
    model_cls: type[TModel] = BaseModel,  # ty: ignore[invalid-parameter-default]
) -> type[TModel]:
    """Create a pydantic model from a JSON schema object.

    Handles nested objects and arrays, `enum` (an Enum for strings, `Literal`
    otherwise), `anyOf` / `oneOf` and `"type": [...]` (unions, `"null"` is None),
    and `$ref`s into `$defs` / `definitions` (also recursive ones).
    """
    builder = _ModelBuilder(schema)
    model = builder.build(schema, schema.get("title", "DynamicModel"), model_cls)
    for created in builder.created:
        created.model_rebuild(_types_namespace=builder.namespace)
    return model


class _ModelBuilder:
    def __init__(self, root: dict[str, Any]) -> None:
        self.defs: dict[str, Any] = {**root.get("definitions", {}), **root.get("$defs", {})}
        self.namespace: dict[str, Any] = {}  # def name -> model (or in progress: None)
        self.created: list[type[BaseModel]] = []

    def build[M: BaseModel](
        self,
        schema: dict[str, Any],
        name: str,
        base: type[M] = BaseModel,  # ty: ignore[invalid-parameter-default]
    ) -> type[M]:
        required = schema.get("required", [])
        fields: dict[str, Any] = {}
        for field_name, props in schema.get("properties", {}).items():
            field_type = self.type_for(props, field_name)
            if props.get("nullable", False):
                field_type = Optional[field_type]  # noqa: UP045
            default = props.get("default", ...) if field_name in required else props.get("default")
            description = props.get("description", props.get("title", ""))
            fields[field_name] = (field_type, Field(default, description=description))
        model = create_model(name, **fields, __base__=base)
        self.created.append(model)
        return model

    def type_for(self, props: Any, field_name: str) -> Any:  # noqa: PLR0911
        """Python type of a property schema."""
        if not isinstance(props, dict):
            return Any
        if isinstance(ref := props.get("$ref"), str):
            return self.ref_type(ref)
        if variants := props.get("anyOf") or props.get("oneOf"):
            return _union([self.type_for(v, field_name) for v in variants])
        if enum_values := props.get("enum"):
            if all(isinstance(v, str) for v in enum_values):
                enum_name = f"{field_name.capitalize()}Enum"
                return Enum(enum_name, {v: v for v in enum_values})
            return Literal[tuple(enum_values)]
        json_type = props.get("type")
        if isinstance(json_type, list):
            return _union([self.type_for({**props, "type": t}, field_name) for t in json_type])
        if json_type == "null":
            return type(None)
        if json_type == "object" and "properties" in props:
            return self.build(props, props.get("title", field_name.capitalize()))
        if json_type == "array" and isinstance(props.get("items"), dict):
            item_type = self.type_for(props["items"], field_name)
            return list[item_type]
        return TYPE_MAPPING.get(json_type, Any)

    def ref_type(self, ref: str) -> Any:
        for prefix in ("#/$defs/", "#/definitions/"):
            if ref.startswith(prefix):
                name = ref.removeprefix(prefix)
                break
        else:
            return Any
        if name in self.namespace:  # built, or in progress (recursive ref)
            return self.namespace[name] or name
        target = self.defs.get(name)
        if not isinstance(target, dict):
            return Any
        if target.get("type", "object") != "object" or "properties" not in target:
            return self.type_for(target, name)
        self.namespace[name] = None
        model = self.build(target, name)
        self.namespace[name] = model
        return model


def _union(members: list[Any]) -> Any:
    unique = list(dict.fromkeys(members))
    if Any in unique:
        return Any
    return functools.reduce(operator.or_, unique)


if __name__ == "__main__":
    schema = {
        "$id": "https://example.com/person.schema.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Person",
        "type": "object",
        "properties": {
            "firstName": {"type": "string", "description": "The person's first name."},
            "lastName": {"type": "string", "description": "The person's last name."},
            "age": {
                "description": "Age in years, must be equal to or greater than zero.",
                "type": "integer",
                "minimum": 0,
            },
        },
    }
    model = json_schema_to_base_model(schema)
    import devtools

    devtools.debug(model.model_fields)
