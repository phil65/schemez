"""Models for schema fields and definitions."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, create_model

from schemez import Schema, helpers


class SchemaField(Schema):
    """Field definition for inline response types.

    Defines a single field in an inline response definition, including:
    - Data type specification
    - Optional description
    - Validation constraints

    Used by InlineSchemaDef to structure response fields.
    """

    type: str
    """Data type of the response field"""

    description: str | None = None
    """Optional description of what this field represents"""

    constraints: dict[str, Any] = Field(default_factory=dict)
    """Optional validation constraints for the field"""


class BaseSchemaDef(Schema):
    """Base class for response definitions."""

    type: str = Field(init=False)

    description: str | None = None
    """A description for this response definition."""


class InlineSchemaDef(BaseSchemaDef):
    """Inline definition of schema.

    Allows defining response types directly in the configuration using:
    - Field definitions with types and descriptions
    - Optional validation constraints
    - Custom field descriptions

    Example:
        schemas:
          BasicResult:
            type: inline
            fields:
              success: {type: bool, description: "Operation success"}
              message: {type: str, description: "Result details"}
    """

    type: Literal["inline"] = Field("inline", init=False)
    """Inline response definition."""

    fields: dict[str, SchemaField]
    """A dictionary containing all fields."""

    def get_schema(self) -> type[Schema]:  # type: ignore
        """Create Pydantic model from inline definition."""
        fields = {}
        for name, field in self.fields.items():
            python_type = helpers.resolve_type_string(field.type)
            if not python_type:
                msg = f"Unsupported field type: {field.type}"
                raise ValueError(msg)

            field_info = Field(description=field.description, **(field.constraints))
            fields[name] = (python_type, field_info)

        cls_name = self.description or "ResponseType"
        return create_model(cls_name, **fields, __base__=Schema, __doc__=self.description)  # type: ignore[call-overload]


class ImportedSchemaDef(BaseSchemaDef):
    """Response definition that imports an existing Pydantic model.

    Allows using externally defined Pydantic models as response types.
    Benefits:
    - Reuse existing model definitions
    - Full Python type support
    - Complex validation logic
    - IDE support for imported types

    Example:
        responses:
          AnalysisResult:
            type: import
            import_path: myapp.models.AnalysisResult
    """

    type: Literal["import"] = Field("import", init=False)
    """Import-path based response definition."""

    import_path: str
    """The path to the pydantic model to use as the response type."""

    # mypy is confused about "type"
    # TODO: convert BaseModel to Schema?
    def get_schema(self) -> type[BaseModel]:  # type: ignore
        """Import and return the model class."""
        try:
            model_class = helpers.import_class(self.import_path)
            if not issubclass(model_class, BaseModel):
                msg = f"{self.import_path} must be a Pydantic model"
                raise TypeError(msg)  # noqa: TRY301
        except Exception as e:
            msg = f"Failed to import response type {self.import_path}"
            raise ValueError(msg) from e
        else:
            return model_class


SchemaDef = Annotated[InlineSchemaDef | ImportedSchemaDef, Field(discriminator="type")]
