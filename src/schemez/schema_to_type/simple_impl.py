"""Configuration models for Schemez."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field, create_model


if TYPE_CHECKING:
    from pydantic.fields import FieldInfo


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
    model_cls: type[TModel] = BaseModel,  # type: ignore[assignment]
) -> type[TModel]:

    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])
    model_fields = {}

    def process_field(
        field_name: str, field_props: dict[str, Any]
    ) -> tuple[Any, FieldInfo]:
        """Recursively processes a field and returns its type and Field instance."""
        json_type = field_props.get("type", "string")
        enum_values = field_props.get("enum")

        # Handle Enums
        if enum_values:
            enum_name: str = f"{field_name.capitalize()}Enum"
            field_type: Any = Enum(enum_name, {v: v for v in enum_values})  # type: ignore[misc]
        # Handle Nested Objects
        elif json_type == "object" and "properties" in field_props:
            # Recursively create submodel
            field_type = json_schema_to_base_model(field_props)  # type: ignore[misc, assignment]
        # Handle Arrays with Nested Objects
        elif json_type == "array" and "items" in field_props:
            item_props = field_props["items"]
            if item_props.get("type") == "object":
                item_type: Any = json_schema_to_base_model(item_props)  # pyright: ignore[reportRedeclaration]
            else:
                item_type = TYPE_MAPPING.get(item_props.get("type"), Any)  # pyright: ignore[reportAssignmentType]
            field_type = list[item_type]  # type: ignore[assignment, misc]
        else:
            field_type = TYPE_MAPPING.get(json_type, Any)  # type: ignore[assignment, misc]

        # Handle default values and optionality
        default_value = field_props.get("default", ...)
        nullable = field_props.get("nullable", False)
        description = field_props.get("title", "")

        if nullable:
            field_type = Optional[field_type]  # type: ignore[assignment, misc] # noqa: UP045

        if field_name not in required_fields:
            default_value = field_props.get("default")

        return field_type, Field(default_value, description=description)  # pyright: ignore[reportReturnType]

    # Process each field
    for field_name, field_props in properties.items():
        model_fields[field_name] = process_field(field_name, field_props)

    return create_model(  # type: ignore[call-overload, no-any-return]
        schema.get("title", "DynamicModel"), **model_fields, __base__=model_cls
    )


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

    devtools.debug(model.__fields__)
