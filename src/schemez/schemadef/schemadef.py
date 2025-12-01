"""Models for schema fields and definitions."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, Field, create_model, field_validator

from schemez import Schema, helpers


if TYPE_CHECKING:
    from pydantic import ValidationInfo


FieldType = Literal[
    "str",
    "int",
    "bool",
    "float",
    "list",
    "dict",
    "set",
    "enum",
    "list[str]",
    "list[int]",
    "list[float]",
    "list[bool]",
    "dict[str, str]",
    "dict[str, int]",
    "dict[str, float]",
    "dict[str, bool]",
    "dict[str, Any]",
    "set[str]",
    "set[int]",
    "set[float]",
    "set[bool]",
    "tuple[str, ...]",
    "tuple[int, ...]",
    "Any",
]
ConstraintType = Literal[
    "alias",
    "alias_priority",
    "validation_alias",
    "serialization_alias",
    "title",
    "field_title_generator",
    "description",
    "examples",
    "exclude",
    "exclude_if",
    "discriminator",
    "deprecated",
    "json_schema_extra",
    "frozen",
    "validate_default",
    "repr",
    "init",
    "init_var",
    "kw_only",
    "pattern",
    "strict",
    "coerce_numbers_to_str",
    "gt",
    "ge",
    "lt",
    "le",
    "multiple_of",
    "allow_inf_nan",
    "max_digits",
    "decimal_places",
    "min_length",
    "max_length",
    "union_mode",
    "fail_fast",
]


class SchemaField(Schema):
    """Field definition for inline response types.

    Defines a single field in an inline response definition, including:
    - Data type specification
    - Optional description
    - Validation constraints
    - Enum values (when type is 'enum')
    - Field dependencies and relationships

    Used by InlineSchemaDef to structure response fields.
    """

    type: FieldType | str = Field(
        title="Field type", examples=["str", "int", "list[str]", "dict[str, Any]", "enum"]
    )
    """Data type of the response field"""

    description: str | None = Field(
        default=None,
        title="Field description",
        examples=["User name", "Port number", "Enable debug mode"],
    )
    """Optional description of what this field represents"""

    values: list[Any] | None = Field(
        default=None,
        title="Enum values",
        examples=[["active", "inactive", "pending"], ["small", "medium", "large"]],
    )
    """Values for enum type fields"""

    default: Any | None = Field(
        default=None,
        title="Default value",
        examples=["localhost", 8080, True, []],
    )
    """Default value for the field"""

    title: str | None = Field(
        default=None,
        title="Field title",
        examples=["Server Name", "Port Number", "Debug Mode"],
    )
    """Title for the field in generated JSON Schema"""

    pattern: str | None = Field(
        default=None,
        title="Regex pattern",
        examples=["^[A-Za-z]+$", "\\d{3}-\\d{3}-\\d{4}", "^[a-z0-9_]+$"],
    )
    """Regex pattern for string validation"""

    min_length: Annotated[int | None, Field(ge=0)] = Field(
        default=None,
        title="Minimum length",
        examples=[1, 3, 8],
    )
    """Minimum length for collections"""

    max_length: Annotated[int | None, Field(ge=0)] = Field(
        default=None, title="Maximum length", examples=[50, 255, 1000]
    )
    """Maximum length for collections"""

    gt: float | None = Field(default=None, title="Greater than", examples=[0, 1.0, 100])
    """Greater than (exclusive) validation for numbers"""

    ge: float | None = Field(
        default=None,
        title="Greater than or equal",
        examples=[0, 1, 18],
    )
    """Greater than or equal (inclusive) validation for numbers"""

    lt: float | None = Field(default=None, title="Less than", examples=[100, 65536, 1.0])
    """Less than (exclusive) validation for numbers"""

    le: float | None = Field(
        default=None,
        title="Less than or equal",
        examples=[100, 255, 99.9],
    )
    """Less than or equal (inclusive) validation for numbers"""

    multiple_of: Annotated[float | None, Field(gt=0)] = Field(
        default=None,
        title="Multiple of",
        examples=[2, 5, 10, 0.5],
    )
    """Number must be a multiple of this value"""

    literal_value: Any | None = Field(
        default=None,
        title="Literal value",
        examples=["production", 42, True],
    )
    """Value for Literal type constraint, makes field accept only this specific value"""

    examples: list[Any] | None = Field(
        default=None,
        title="Field examples",
        examples=[["localhost", "example.com"], [8080, 3000, 443], [True, False]],
    )
    """Examples for this field in JSON Schema"""

    optional: bool = Field(default=False, title="Optional field")
    """Whether this field is optional (None value allowed)"""

    json_schema_extra: dict[str, Any] | None = Field(
        default=None,
        title="JSON Schema extras",
        examples=[{"format": "email"}, {"contentMediaType": "application/json"}],
    )
    """Additional JSON Schema information"""

    field_config: dict[str, Any] | None = Field(
        default=None,
        title="Pydantic field config",
        examples=[{"alias": "userName"}, {"exclude": True}],
    )
    """Configuration for Pydantic model fields"""

    # Dependencies between fields
    dependent_required: dict[str, list[str]] = Field(
        default_factory=dict,
        title="Required field dependencies",
        examples=[
            {"payment_method": ["card_number", "expiry_date"]},
            {"auth_enabled": ["username", "password"]},
        ],
    )
    """Field dependencies - when this field exists, dependent fields are required"""

    dependent_schema: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        title="Schema dependencies",
        examples=[
            {"ssl_enabled": {"properties": {"ssl_cert": {"type": "string"}}}},
            {"debug": {"properties": {"log_level": {"enum": ["debug", "trace"]}}}},
        ],
    )
    """Schema dependencies - when this field exists, dependent fields must match schema"""

    # Extensibility for future or custom constraints
    constraints: dict[ConstraintType | str, Any] = Field(
        default_factory=dict,
        title="Constraints",
        examples=[
            {"min_length": 3, "max_length": 50},
            {"ge": 0, "le": 100},
            {"strict": True, "frozen": False},
        ],
    )
    """Additional Pydantic Field constraints not covered by explicit fields"""

    def add_required_dependency(self, field_name: str, required_fields: list[str]) -> None:
        """Add a dependency requiring other fields when this field exists.

        Args:
            field_name: The field that triggers the dependency
            required_fields: Fields that become required when field_name exists
        """
        if field_name not in self.dependent_required:
            self.dependent_required[field_name] = []
        self.dependent_required[field_name].extend([
            field for field in required_fields if field not in self.dependent_required[field_name]
        ])

    def add_schema_dependency(self, field_name: str, schema: dict[str, Any]) -> None:
        """Add a schema dependency when this field exists.

        Args:
            field_name: The field that triggers the dependency
            schema: JSON Schema to apply when field_name exists
        """
        self.dependent_schema[field_name] = schema

    @field_validator("max_length")
    @classmethod
    def validate_max_min_length(cls, v: int | None, info: ValidationInfo) -> int | None:
        if (
            v is not None
            and info.data.get("min_length") is not None
            and v < info.data.get("min_length")  # type:ignore[operator]
        ):
            msg = "max_length must be ≥ min_length"
            raise ValueError(msg)
        return v


class BaseSchemaDef(Schema):
    """Response type definition."""

    type: str = Field(init=False)

    description: str | None = Field(
        default=None,
        title="Schema description",
        examples=["User profile data", "API response format", "Configuration schema"],
    )
    """A description for this response definition."""


class InlineSchemaDef(BaseSchemaDef):
    """Inline definition of schema.

    Allows defining response types directly in the configuration using:
    - Field definitions with types and descriptions
    - Optional validation constraints
    - Custom field descriptions
    - Field dependencies and relationships

    Example:
        type: inline
        fields:
            success: {type: bool, description: "Operation success"}
            message: {type: str, description: "Result details"}
            payment_method: {type: str, description: "Payment method used"}
            card_number: {
                type: str,
                description: "Credit card number",
                dependent_required: {
                "payment_method": ["card_expiry", "card_cvc"]
                }
            }
    """

    type: Literal["inline"] = Field("inline", init=False)
    """Inline response definition."""

    fields: dict[str, SchemaField] = Field(
        title="Schema fields",
        examples=[
            {
                "name": {"type": "str", "description": "User name"},
                "age": {"type": "int", "description": "User age", "ge": 0},
            }
        ],
    )
    """A dictionary containing all fields."""

    def add_field_dependency(
        self, field_name: str, dependent_on: str, required_fields: list[str]
    ) -> None:
        """Add a dependency between fields.

        Args:
            field_name: The field where to add the dependency
            dependent_on: The field that triggers the dependency
            required_fields: Fields that become required when the dependency is triggered
        """
        if field_name not in self.fields:
            msg = f"Field '{field_name}' not found in schema"
            raise ValueError(msg)

        field = self.fields[field_name]
        field.add_required_dependency(dependent_on, required_fields)

    def add_schema_dependency(
        self, field_name: str, dependent_on: str, schema: dict[str, Any]
    ) -> None:
        """Add a schema dependency between fields.

        Args:
            field_name: The field where to add the dependency
            dependent_on: The field that triggers the dependency
            schema: JSON Schema to apply when dependent_on exists
        """
        if field_name not in self.fields:
            msg = f"Field '{field_name}' not found in schema"
            raise ValueError(msg)

        field = self.fields[field_name]
        field.add_schema_dependency(dependent_on, schema)

    def get_schema(self) -> type[BaseModel]:  # type: ignore[valid-type]
        """Create Pydantic model from inline definition."""
        fields = {}
        model_dependencies: dict[str, dict[str, Any]] = {}

        # First pass: collect all field information
        for name, field in self.fields.items():
            # Initialize constraint dictionary
            field_constraints: dict[str, Any] = {}

            # Handle enum type
            if field.type == "enum":
                if not field.values:
                    msg = f"Field {name!r} has type 'enum' but no values defined"
                    raise ValueError(msg)

                enum_name = f"{name.capitalize()}Enum"  # Create dynamic Enum class
                enum_members = {}  # Create enum members dictionary
                for i, value in enumerate(field.values):
                    # If value is valid Python identifier use as is, otherwise create synthetic name
                    key = value if isinstance(value, str) and value.isidentifier() else f"VALUE_{i}"
                    enum_members[key] = value

                enum_class = Enum(enum_name, enum_members)  # type: ignore[misc]
                python_type: Any = enum_class

                if field.default is not None:  # Handle enum default value specially
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

            if field.literal_value is not None:  # Handle literal constraint if provided
                python_type = Literal[field.literal_value]

            if field.optional:  # Handle optional fields (allowing None)
                python_type = python_type | None

            # Add standard Pydantic constraints. Collect all constraint values
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

            if field.examples:
                if field.json_schema_extra is None:
                    field.json_schema_extra = {}
                field.json_schema_extra["examples"] = field.examples

            if field.json_schema_extra:
                field_constraints["json_schema_extra"] = field.json_schema_extra

            if field.dependent_required or field.dependent_schema:
                if field.json_schema_extra is None:
                    field_constraints["json_schema_extra"] = {}

                json_extra = field_constraints.get("json_schema_extra", {})
                if field.dependent_required:
                    if "dependentRequired" not in json_extra:
                        json_extra["dependentRequired"] = {}
                    json_extra["dependentRequired"].update(field.dependent_required)

                if field.dependent_schema:
                    if "dependentSchemas" not in json_extra:
                        json_extra["dependentSchemas"] = {}
                    json_extra["dependentSchemas"].update(field.dependent_schema)

                field_constraints["json_schema_extra"] = json_extra

            field_constraints.update(field.constraints)  # Add any additional constraints
            field_info = Field(description=field.description, **field_constraints)
            fields[name] = (python_type, field_info)

            # Collect model-level dependencies for JSON Schema
            if field.dependent_required or field.dependent_schema:
                if not model_dependencies:
                    model_dependencies = {"json_schema_extra": {}}
                extra = model_dependencies["json_schema_extra"]
                if field.dependent_required:
                    if "dependentRequired" not in extra:
                        extra["dependentRequired"] = {}
                    extra["dependentRequired"].update(field.dependent_required)
                if field.dependent_schema:
                    if "dependentSchemas" not in extra:
                        extra["dependentSchemas"] = {}
                    extra["dependentSchemas"].update(field.dependent_schema)

        model = create_model(  # Create the model class
            self.description or "ResponseType",
            **fields,
            __base__=BaseModel,
            __doc__=self.description,
        )  # type: ignore[call-overload]

        # Add model-level JSON Schema extras for dependencies
        if model_dependencies:
            existing_extra = model.model_config.get("json_schema_extra")
            deps_extra = model_dependencies["json_schema_extra"]

            match existing_extra:
                case None:
                    model.model_config["json_schema_extra"] = deps_extra
                case dict() as schema_extra:
                    schema_extra.update(deps_extra)
                case Callable() as callable_func:

                    def wrapped_extra(*args: Any) -> None:
                        callable_func(*args)
                        schema = args[0]
                        schema.update(deps_extra)

                    model.model_config["json_schema_extra"] = wrapped_extra

        # Return the created model
        return model  # type: ignore[no-any-return]


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

    import_path: str = Field(
        title="Import path",
        examples=[
            "myapp.models.User",
            "schemas.api:ResponseModel",
            "models.core.BaseResponse",
        ],
    )
    """The path to the pydantic model to use as the response type."""

    # mypy is confused about "type"
    # TODO: convert BaseModel to Schema?
    def get_schema(self) -> type[BaseModel]:  # type: ignore[valid-type]
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
