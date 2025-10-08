"""Tests for create_constructor_schema function."""

from __future__ import annotations

import dataclasses
from typing import Literal

import pydantic

from py2openai.schema_generators import create_constructor_schema


def test_regular_class_constructor_schema() -> None:
    """Test schema generation for regular class constructor."""

    class User:
        """A user class for testing."""

        def __init__(
            self,
            name: str,
            age: int,
            email: str | None = None,
            role: Literal["admin", "user"] = "user",
        ) -> None:
            self.name = name
            self.age = age
            self.email = email
            self.role = role

    schema = create_constructor_schema(User)

    assert schema.name == "create_User"
    assert schema.description == "A user class for testing."
    assert set(schema.parameters["properties"]) == {"name", "age", "email", "role"}
    assert set(schema.required) == {"name", "age"}

    # Check specific property details
    props = schema.parameters["properties"]
    assert props["name"]["type"] == "string"
    assert props["age"]["type"] == "integer"
    assert props["email"]["type"] == "string"
    assert props["role"]["type"] == "string"
    assert props["role"]["enum"] == ["admin", "user"]  # type: ignore
    assert props["role"]["default"] == "user"  # type: ignore


def test_dataclass_constructor_schema() -> None:
    """Test schema generation for dataclass constructor."""

    @dataclasses.dataclass
    class Product:
        """A product class for testing."""

        name: str
        price: float
        in_stock: bool = True
        tags: list[str] = dataclasses.field(default_factory=list)

    schema = create_constructor_schema(Product)

    assert schema.name == "create_Product"
    assert schema.description == "A product class for testing."
    assert set(schema.parameters["properties"]) == {
        "name",
        "price",
        "in_stock",
        "tags",
    }
    assert set(schema.required) == {"name", "price"}

    # Check specific property details
    props = schema.parameters["properties"]
    assert props["name"]["type"] == "string"
    assert props["price"]["type"] == "number"
    assert props["in_stock"]["type"] == "boolean"
    assert props["in_stock"]["default"] is True  # type: ignore
    assert props["tags"]["type"] == "array"
    assert props["tags"]["items"]["type"] == "string"


def test_pydantic_model_constructor_schema() -> None:
    """Test schema generation for Pydantic BaseModel constructor."""

    class Order(pydantic.BaseModel):
        """An order class for testing."""

        id: str
        items: list[int]
        status: Literal["pending", "completed", "cancelled"] = "pending"
        notes: str | None = None
        model_config = pydantic.ConfigDict(frozen=True)

    schema = create_constructor_schema(Order)

    assert schema.name == "create_Order"
    assert schema.description == "An order class for testing."
    assert set(schema.parameters["properties"]) == {
        "id",
        "items",
        "status",
        "notes",
    }
    assert set(schema.required) == {"id", "items"}

    # Check specific property details
    props = schema.parameters["properties"]
    assert props["id"]["type"] == "string"
    assert props["items"]["type"] == "array"
    assert props["items"]["items"]["type"] == "integer"
    assert props["status"]["type"] == "string"
    assert props["status"]["enum"] == ["pending", "completed", "cancelled"]  # type: ignore
    assert props["status"]["default"] == "pending"  # type: ignore
    assert props["notes"]["type"] == "string"
