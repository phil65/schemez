"""Test enum support in SchemaField."""

from __future__ import annotations

from enum import Enum

from pydantic import ValidationError
import pytest

from schemez.schemadef.schemadef import InlineSchemaDef, SchemaField


def test_explicit_constraints():
    """Test that explicit constraints work correctly."""
    fields = {
        "name": SchemaField(
            type="str",
            min_length=3,
            max_length=50,
            pattern=r"^[a-zA-Z0-9_]+$",
        ),
        "age": SchemaField(type="int", description="User age", ge=18, lt=120),
        "score": SchemaField(
            type="float",
            description="Score value",
            ge=0.0,
            le=100.0,
            multiple_of=0.5,
        ),
        "tags": SchemaField(type="set[str]", min_length=1, max_length=10),
    }
    schema_def = InlineSchemaDef(description="Test Schema Constraints", fields=fields)
    model = schema_def.get_schema()
    valid_instance = model(
        name="user123",  # type: ignore
        age=30,  # type: ignore
        score=95.5,  # type: ignore
        tags=["developer", "python"],  # type: ignore
    )
    assert valid_instance.name == "user123"  # type: ignore
    assert valid_instance.age == 30  # type: ignore  # noqa: PLR2004
    assert valid_instance.score == 95.5  # type: ignore # noqa: PLR2004
    assert valid_instance.tags == {"developer", "python"}  # type: ignore

    # Test min_length constraint
    with pytest.raises(ValidationError):
        model(
            name="ab",  # Too short  # type: ignore
            age=30,  # type: ignore
            score=95.5,  # type: ignore
            tags=["developer"],  # type: ignore
        )

    # Test pattern constraint
    with pytest.raises(ValidationError):
        model(
            name="user-name",  # Invalid character # type: ignore
            age=30,  # type: ignore
            score=95.5,  # type: ignore
            tags=["developer"],  # type: ignore
        )

    # Test ge constraint
    with pytest.raises(ValidationError):
        model(
            name="user123",  # type: ignore
            age=17,  # Too young  # type: ignore
            score=95.5,  # type: ignore
            tags=["developer"],  # type: ignore
        )

    # Test multiple_of constraint
    with pytest.raises(ValidationError):
        model(
            name="user123",  # type: ignore
            age=30,  # type: ignore
            score=95.7,  # Not a multiple of 0.5  # type: ignore
            tags=["developer"],  # type: ignore
        )

    # Test that set automatically enforces uniqueness
    instance = model(
        name="user123",  # type: ignore
        age=30,  # type: ignore
        score=95.5,  # type: ignore
        # Duplicates get automatically removed
        tags=["developer", "developer", "python"],  # type: ignore
    )
    assert len(instance.tags) == 2  # type: ignore  # noqa: PLR2004
    assert "developer" in instance.tags  # type: ignore
    assert "python" in instance.tags  # type: ignore


def test_enum_type():
    """Test that enum type creates an Enum type."""
    fields = {
        "color": SchemaField(
            type="enum",
            description="Color selection",
            values=["red", "green", "blue"],
        )
    }
    schema_def = InlineSchemaDef(description="Test Schema", fields=fields)

    model = schema_def.get_schema()

    # Check that the model has the color field with an Enum type
    color_field = model.model_fields["color"]
    assert color_field.description == "Color selection"

    # The field should be an Enum type
    color_value = model(color="red").color  # type: ignore
    assert isinstance(color_value, Enum)
    assert color_value.value == "red"

    # Check all values work
    assert model(color="green").color.value == "green"  # type: ignore
    assert model(color="blue").color.value == "blue"  # type: ignore

    # Try with invalid value
    with pytest.raises(ValidationError):
        model(color="purple")  # type: ignore


def test_enum_with_numeric_values():
    """Test that enum type works with numeric values."""
    fields = {
        "priority": SchemaField(
            type="enum",
            description="Task priority",
            values=[1, 2, 3, 5, 8],
        )
    }
    schema_def = InlineSchemaDef(description="Test Schema", fields=fields)
    model = schema_def.get_schema()

    # Create a valid instance
    instance = model(priority=5)  # type: ignore
    assert instance.priority.value == 5  # type: ignore  # noqa: PLR2004
    # Try with invalid value
    with pytest.raises(ValidationError):
        model(priority=4)  # type: ignore


def test_mixed_enum_values():
    """Test that enum type works with mixed value types."""
    fields = {
        "value": SchemaField(
            type="enum",
            description="Mixed values",
            values=[1, "text", True],
        )
    }
    schema_def = InlineSchemaDef(description="Test Schema", fields=fields)
    model = schema_def.get_schema()

    # Create valid instances
    assert model(value=1).value.value == 1  # type: ignore
    assert model(value="text").value.value == "text"  # type: ignore
    assert model(value=True).value.value  # type: ignore

    # Try with invalid value
    with pytest.raises(ValidationError):
        model(value="invalid")  # type: ignore


def test_missing_enum_values():
    """Test that enum type raises error when values are missing."""
    fields = {"status": SchemaField(type="enum", description="Status with no vals")}
    schema_def = InlineSchemaDef(description="Test Schema", fields=fields)
    # Should raise an error when creating the schema
    with pytest.raises(ValueError, match="has type 'enum' but no values defined"):
        schema_def.get_schema()


def test_enum_with_default():
    """Test that enum type works with default value."""
    fields = {
        "status": SchemaField(
            type="enum",
            description="Status with default",
            values=["pending", "active", "completed"],
            default="pending",
        )
    }
    schema_def = InlineSchemaDef(description="Test Schema", fields=fields)

    model = schema_def.get_schema()

    # Test default value
    instance = model()
    assert instance.status == "pending"  # type: ignore
