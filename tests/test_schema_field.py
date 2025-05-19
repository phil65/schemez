"""Test SchemaField functionality comprehensively."""

from __future__ import annotations

from enum import Enum

from pydantic import ValidationError
import pytest

from schemez.schemadef.schemadef import InlineSchemaDef, SchemaField


def test_basic_types():
    """Test that basic types are resolved correctly."""
    schema_def = InlineSchemaDef(
        description="Test Schema with Basic Types",
        fields={
            "string_field": SchemaField(type="str", description="A string field"),
            "int_field": SchemaField(type="int", description="An integer field"),
            "float_field": SchemaField(type="float", description="A float field"),
            "bool_field": SchemaField(type="bool", description="A boolean field"),
            "dict_field": SchemaField(type="dict", description="A dictionary field"),
            "list_field": SchemaField(type="list", description="A list field"),
        },
    )

    model = schema_def.get_schema()

    # Check field types
    assert model.model_fields["string_field"].annotation is str
    assert model.model_fields["int_field"].annotation is int
    assert model.model_fields["float_field"].annotation is float
    assert model.model_fields["bool_field"].annotation is bool
    assert model.model_fields["dict_field"].annotation is dict
    assert model.model_fields["list_field"].annotation is list

    # Check descriptions
    assert model.model_fields["string_field"].description == "A string field"


def test_generic_types():
    """Test that generic types like list[str] are resolved correctly."""
    schema_def = InlineSchemaDef(
        description="Test Schema with Generic Types",
        fields={
            "str_list": SchemaField(type="list[str]", description="List of strings"),
            "int_list": SchemaField(type="list[int]", description="List of integers"),
            "str_dict": SchemaField(
                type="dict[str, int]", description="String to int mapping"
            ),
        },
    )

    model = schema_def.get_schema()

    # Create valid instances
    instance = model(
        str_list=["a", "b", "c"],  # type: ignore
        int_list=[1, 2, 3],  # type: ignore
        str_dict={"a": 1, "b": 2},  # type: ignore
    )

    assert instance.str_list == ["a", "b", "c"]  # type: ignore
    assert instance.int_list == [1, 2, 3]  # type: ignore
    assert instance.str_dict == {"a": 1, "b": 2}  # type: ignore

    # Test type validation
    with pytest.raises(ValidationError):
        model(str_list=[1, 2, 3])  # type: ignore

    with pytest.raises(ValidationError):
        model(int_list=["a", "b", "c"])  # type: ignore

    with pytest.raises(ValidationError):
        model(str_dict={1: "a"})  # type: ignore


def test_string_constraints():
    """Test string validation constraints."""
    schema_def = InlineSchemaDef(
        description="Test Schema with String Constraints",
        fields={
            "username": SchemaField(
                type="str",
                description="Username",
                min_length=3,
                max_length=20,
                pattern=r"^[a-z0-9_]+$",
            ),
        },
    )

    model = schema_def.get_schema()

    # Valid username
    assert model(username="user_123").username == "user_123"  # type: ignore

    # Too short
    with pytest.raises(ValidationError):
        model(username="ab")  # type: ignore

    # Too long
    with pytest.raises(ValidationError):
        model(username="a" * 21)  # type: ignore

    # Invalid pattern
    with pytest.raises(ValidationError):
        model(username="User-Name")  # type: ignore


def test_numeric_constraints():
    """Test numeric validation constraints."""
    schema_def = InlineSchemaDef(
        description="Test Schema with Numeric Constraints",
        fields={
            "age": SchemaField(
                type="int",
                description="Age",
                ge=18,
                lt=120,
            ),
            "score": SchemaField(
                type="float",
                description="Score",
                gt=0.0,
                le=100.0,
                multiple_of=0.5,
            ),
        },
    )

    model = schema_def.get_schema()

    # Valid values
    instance = model(age=18, score=99.5)  # type: ignore
    assert instance.age == 18  # type: ignore  # noqa: PLR2004
    assert instance.score == 99.5  # type: ignore  # noqa: PLR2004

    # Test ge constraint
    with pytest.raises(ValidationError):
        model(age=17, score=50.0)  # type: ignore

    # Test lt constraint
    with pytest.raises(ValidationError):
        model(age=120, score=50.0)  # type: ignore

    # Test gt constraint
    with pytest.raises(ValidationError):
        model(age=18, score=0.0)  # type: ignore

    # Test le constraint
    with pytest.raises(ValidationError):
        model(age=18, score=100.5)  # type: ignore

    # Test multiple_of constraint
    with pytest.raises(ValidationError):
        model(age=18, score=50.1)  # type: ignore


def test_collection_constraints():
    """Test collection validation constraints."""
    schema_def = InlineSchemaDef(
        description="Test Schema with Collection Constraints",
        fields={
            "tags": SchemaField(
                type="list[str]",
                description="Tags",
                min_length=1,
                max_length=5,
            ),
            "unique_tags": SchemaField(
                type="set[str]",
                description="Tags with unique values",
                min_length=1,
                max_length=5,
            ),
        },
    )

    model = schema_def.get_schema()

    # Valid - provide both required fields
    instance = model(
        tags=["one", "two"],  # type: ignore
        unique_tags={"one", "two"},  # type: ignore
    )
    assert instance.tags == ["one", "two"]  # type: ignore
    assert instance.unique_tags == {"one", "two"}  # type: ignore

    # Too few items - need to provide both fields
    with pytest.raises(ValidationError):
        model(
            tags=[],  # type: ignore
            unique_tags={"one", "two"},  # type: ignore
        )

    with pytest.raises(ValidationError):
        model(
            tags=["one", "two"],  # type: ignore
            unique_tags=set(),  # type: ignore
        )

    # Too many items
    with pytest.raises(ValidationError):
        model(
            tags=["one", "two", "three", "four", "five", "six"],  # type: ignore
            unique_tags={"one", "two"},  # type: ignore
        )

    with pytest.raises(ValidationError):
        model(
            tags=["one", "two"],  # type: ignore
            unique_tags={"one", "two", "three", "four", "five", "six"},  # type: ignore
        )

    # Non-unique items are allowed in list but automatically deduplicated in set
    instance = model(
        tags=["one", "one", "two"],  # type: ignore
        unique_tags={"one", "two"},  # type: ignore
    )
    assert instance.tags == ["one", "one", "two"]  # type: ignore

    # Attempting to pass non-unique items to a set field will automatically deduplicate
    instance = model(
        tags=["one", "two"],  # type: ignore
        unique_tags=["one", "one", "two"],  # type: ignore  # Will be converted to set
    )
    assert instance.unique_tags == {"one", "two"}  # type: ignore


def test_default_values():
    """Test default values for fields."""
    schema_def = InlineSchemaDef(
        description="Test Schema with Default Values",
        fields={
            "name": SchemaField(
                type="str",
                description="Name",
                default="Anonymous",
            ),
            "active": SchemaField(
                type="bool",
                description="Active status",
                default=True,
            ),
            "count": SchemaField(
                type="int",
                description="Count",
                default=0,
            ),
        },
    )

    model = schema_def.get_schema()

    # Test defaults when not provided
    instance = model()
    assert instance.name == "Anonymous"  # type: ignore
    assert instance.active is True  # type: ignore
    assert instance.count == 0  # type: ignore

    # Test overriding defaults
    instance = model(name="User", active=False, count=10)  # type: ignore
    assert instance.name == "User"  # type: ignore
    assert instance.active is False  # type: ignore
    assert instance.count == 10  # type: ignore  # noqa: PLR2004


def test_const_constraint():
    """Test const constraint."""
    schema_def = InlineSchemaDef(
        description="Test Schema with Const Constraint",
        fields={
            "version": SchemaField(
                type="str",
                description="API Version",
                literal_value="1.0",
            ),
        },
    )

    model = schema_def.get_schema()

    # Valid
    assert model(version="1.0").version == "1.0"  # type: ignore

    # Invalid
    with pytest.raises(ValidationError):
        model(version="2.0")  # type: ignore


def test_enum_type():
    """Test enum type field."""
    schema_def = InlineSchemaDef(
        description="Test Schema with Enum",
        fields={
            "status": SchemaField(
                type="enum",
                description="Status",
                values=["pending", "active", "completed"],
            ),
        },
    )

    model = schema_def.get_schema()

    # The field should use an Enum type
    status_field = model.model_fields["status"]
    assert issubclass(status_field.annotation, Enum)  # type: ignore

    # Valid values
    status = model(status="pending").status  # type: ignore
    assert isinstance(status, Enum)
    assert status.value == "pending"

    # Invalid value
    with pytest.raises(ValidationError):
        model(status="invalid")  # type: ignore


def test_enum_with_default():
    """Test enum with default value."""
    schema_def = InlineSchemaDef(
        description="Test Schema with Enum Default",
        fields={
            "role": SchemaField(
                type="enum",
                description="User Role",
                values=["admin", "user", "guest"],
                default="user",
            ),
        },
    )

    model = schema_def.get_schema()

    # Test default
    instance = model()
    assert instance.role == "user"  # type: ignore

    # Override default
    instance = model(role="admin")  # type: ignore
    assert instance.role.value == "admin"  # type: ignore


def test_nullable_constraint():
    """Test nullable constraint."""
    schema_def = InlineSchemaDef(
        description="Test Schema with Nullable Fields",
        fields={
            "name": SchemaField(
                type="str",
                description="Optional name",
                optional=True,
            ),
        },
    )

    model = schema_def.get_schema()

    # Test null value
    instance = model(name=None)  # type: ignore
    assert instance.name is None  # type: ignore


def test_additional_constraints():
    """Test additional constraints via the constraints dict."""
    schema_def = InlineSchemaDef(
        description="Test Schema with Additional Constraints",
        fields={
            "custom_field": SchemaField(
                type="str",
                description="Field with custom constraint",
                examples=["example1", "example2"],
            ),
        },
    )

    model = schema_def.get_schema()
    assert model(custom_field="test").custom_field == "test"  # type: ignore


def test_title_constraint():
    """Test title constraint for JSON Schema generation."""
    schema_def = InlineSchemaDef(
        description="Test Schema with Title",
        fields={
            "user_id": SchemaField(
                type="str",
                description="User ID",
                title="User Identifier",
            ),
        },
    )

    model = schema_def.get_schema()

    # Check that the title was set
    field = model.model_fields["user_id"]
    assert field.title == "User Identifier"


def test_complex_nested_structure():
    """Test complex nested structure with various constraints."""
    schema_def = InlineSchemaDef(
        description="Complex Nested Schema",
        fields={
            "user": SchemaField(
                type="dict[str, Any]",
                description="User information",
                json_schema_extra={
                    "required": ["name", "email"],
                },
            ),
            "settings": SchemaField(
                type="dict[str, bool]",
                description="User settings",
                default={},
            ),
            "metadata": SchemaField(
                type="dict",
                description="Additional metadata",
                optional=True,
            ),
        },
    )

    model = schema_def.get_schema()

    # Test valid complex structure
    instance = model(
        user={"name": "Test User", "email": "test@example.com", "age": 30},  # type: ignore
        settings={"notifications": True, "darkMode": False},  # type: ignore
        metadata={"created": "2023-01-01", "source": "API"},  # type: ignore
    )

    assert instance.user["name"] == "Test User"  # type: ignore
    assert instance.settings["notifications"] is True  # type: ignore
    assert instance.metadata["source"] == "API"  # type: ignore

    # Test with null metadata
    instance = model(
        user={"name": "Test User", "email": "test@example.com"},  # type: ignore
        settings={"notifications": True},  # type: ignore
        metadata=None,  # type: ignore
    )
    # Test optional field with None value
    assert instance.metadata is None  # type: ignore

    # Test with default empty settings
    instance = model(
        user={"name": "Test User", "email": "test@example.com"},  # type: ignore
        metadata=None,  # type: ignore
    )
    assert instance.settings == {}  # type: ignore


if __name__ == "__main__":
    test_basic_types()
    test_generic_types()
    test_string_constraints()
    test_numeric_constraints()
    test_collection_constraints()
    test_default_values()
    test_const_constraint()
    test_enum_type()
    test_enum_with_default()
