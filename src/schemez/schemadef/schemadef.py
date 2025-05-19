"""Models for schema fields and definitions."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, create_model

from schemez import Schema, helpers


class SchemaField(Schema):
    """Field definition for inline response types.

    Defines a single field in an inline response definition, including:
    - Data type specification
    - Optional description
    - Validation constraints
    - Enum values (when type is 'enum')

    Used by InlineSchemaDef to structure response fields.
    """

    type: str
    """Data type of the response field"""

    description: str | None = None
    """Optional description of what this field represents"""

    values: list[Any] | None = None
    """Values for enum type fields"""

    # Common validation constraints
    default: Any | None = None
    """Default value for the field"""

    title: str | None = None
    """Title for the field in generated JSON Schema"""

    pattern: str | None = None
    """Regex pattern for string validation"""

    min_length: int | None = None
    """Minimum length for collections"""

    max_length: int | None = None
    """Maximum length for collections"""

    gt: float | None = None
    """Greater than (exclusive) validation for numbers"""

    ge: float | None = None
    """Greater than or equal (inclusive) validation for numbers"""

    lt: float | None = None
    """Less than (exclusive) validation for numbers"""

    le: float | None = None
    """Less than or equal (inclusive) validation for numbers"""

    multiple_of: float | None = None
    """Number must be a multiple of this value"""

    literal_value: Any | None = None
    """Value for Literal type constraint, makes field accept only this specific value"""

    examples: list[Any] | None = None
    """Examples for this field in JSON Schema"""

    optional: bool = False
    """Whether this field is optional (None value allowed)"""

    json_schema_extra: dict[str, Any] | None = None
    """Additional JSON Schema information"""

    field_config: dict[str, Any] | None = None
    """Configuration for Pydantic model fields"""

    # Extensibility for future or custom constraints
    constraints: dict[str, Any] = Field(default_factory=dict)
    """Additional constraints not covered by explicit fields"""


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

    def get_schema(self) -> type[BaseModel]:  # type: ignore
        """Create Pydantic model from inline definition."""
        fields = {}
        for name, field in self.fields.items():
            # Initialize constraint dictionary
            field_constraints = {}

            # Handle enum type
            if field.type == "enum":
                if not field.values:
                    msg = f"Field '{name}' has type 'enum' but no values defined"
                    raise ValueError(msg)

                # Create dynamic Enum class
                enum_name = f"{name.capitalize()}Enum"

                # Create enum members dictionary
                enum_members = {}
                for i, value in enumerate(field.values):
                    if isinstance(value, str) and value.isidentifier():
                        # If value is a valid Python identifier, use it as is
                        key = value
                    else:
                        # Otherwise, create a synthetic name
                        key = f"VALUE_{i}"
                    enum_members[key] = value

                # Create the enum class
                enum_class = Enum(enum_name, enum_members)
                python_type: Any = enum_class

                # Handle enum default value specially
                if field.default is not None:
                    # Store default value as the enum value string
                    # Pydantic v2 will convert it to the enum instance
                    if field.default in list(field.values):
                        field_constraints["default"] = field.default
                    else:
                        msg = (
                            f"Default value {field.default!r} not found "
                            f"in enum values for field {name!r}"
                        )
                        raise ValueError(msg)
            else:
                python_type = helpers.resolve_type_string(field.type)
                if not python_type:
                    msg = f"Unsupported field type: {field.type}"
                    raise ValueError(msg)

            # Handle literal constraint if provided
            if field.literal_value is not None:
                from typing import Literal as LiteralType

                python_type = LiteralType[field.literal_value]

            # Handle optional fields (allowing None)
            if field.optional:
                python_type = python_type | None  # type: ignore

            # Add standard Pydantic constraints
            # Collect all constraint values
            for constraint in [
                "default",
                "title",
                "min_length",
                "max_length",
                "pattern",
                "min_length",
                "max_length",
                "gt",
                "ge",
                "lt",
                "le",
                "multiple_of",
            ]:
                value = getattr(field, constraint, None)
                if value is not None:
                    field_constraints[constraint] = value

            # Handle examples separately (Pydantic v2 way)
            if field.examples:
                if field.json_schema_extra is None:
                    field.json_schema_extra = {}
                field.json_schema_extra["examples"] = field.examples

            # Add json_schema_extra if provided
            if field.json_schema_extra:
                field_constraints["json_schema_extra"] = field.json_schema_extra

            # Add any additional constraints
            field_constraints.update(field.constraints)

            field_info = Field(description=field.description, **field_constraints)
            fields[name] = (python_type, field_info)

        cls_name = self.description or "ResponseType"
        return create_model(
            cls_name,
            **fields,
            __base__=BaseModel,
            __doc__=self.description,
        )  # type: ignore[call-overload]


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
