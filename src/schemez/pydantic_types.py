"""Custom field types with 'field_type' metadata for UI rendering hints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field


if TYPE_CHECKING:
    from pydantic import BaseModel


ModelIdentifier = Annotated[
    str,
    Field(
        json_schema_extra={"field_type": "model_identifier"},
        pattern=r"^[a-zA-Z0-9\-]+(/[a-zA-Z0-9\-]+)*(:[\w\-\.]+)?$",
        examples=["openai:gpt-o1-mini", "anthropic/claude-3-opus"],
        description="Identifier for an AI model, optionally including provider.",
    ),
]

ModelTemperature = Annotated[
    float,
    Field(
        json_schema_extra={"field_type": "temperature", "step": 0.1},
        ge=0.0,
        le=2.0,
        description=(
            "Controls randomness in model responses.\n"
            "Lower values are more deterministic, higher values more creative"
        ),
        examples=[0.0, 0.7, 1.0],
    ),
]

MimeType = Annotated[
    str,
    Field(
        json_schema_extra={"field_type": "mime_type"},
        pattern=r"^[a-z]+/[a-z0-9\-+.]+$",
        examples=["text/plain", "application/pdf", "image/jpeg", "application/json"],
        description="Standard MIME type identifying file formats and content types",
    ),
]


def get_field_type(model: type[BaseModel], field_name: str) -> dict[str, Any]:
    """Extract field_type metadata from a model field."""
    field_info = model.model_fields[field_name]
    metadata = {}
    if field_info.json_schema_extra and isinstance(field_info.json_schema_extra, dict):
        metadata.update(field_info.json_schema_extra)

    return metadata
