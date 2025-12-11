"""Test field ordering in Schema subclasses."""

from __future__ import annotations

import json

from pydantic import Field

from schemez.schema import Schema


class TestFieldOrdering:
    """Test that subclass fields appear before base class fields."""

    def test_basic_inheritance(self):
        """Test basic two-level inheritance."""

        class Base(Schema):
            base_a: str = "a"
            base_b: int = 1

        class Sub(Base):
            sub_x: str = "x"
            sub_y: int = 2

        instance = Sub()
        keys = list(instance.model_dump().keys())

        # Subclass fields should come first
        assert keys == ["sub_x", "sub_y", "base_a", "base_b"]

    def test_deep_inheritance(self):
        """Test three-level inheritance."""

        class GrandParent(Schema):
            gp_field1: str = "gp1"
            gp_field2: int = 1

        class Parent(GrandParent):
            p_field1: str = "p1"
            p_field2: int = 2

        class Child(Parent):
            c_field1: str = "c1"
            c_field2: int = 3

        instance = Child()
        keys = list(instance.model_dump().keys())

        # Most derived class fields first, then parent, then grandparent
        expected = ["c_field1", "c_field2", "p_field1", "p_field2", "gp_field1", "gp_field2"]
        assert keys == expected

    def test_json_serialization(self):
        """Test that JSON serialization also maintains order."""

        class Base(Schema):
            base_field: str = "base"

        class Sub(Base):
            sub_field: str = "sub"

        instance = Sub()
        json_str = instance.model_dump_json()
        parsed = json.loads(json_str)
        keys = list(parsed.keys())

        assert keys == ["sub_field", "base_field"]

    def test_yaml_serialization(self):
        """Test that YAML serialization maintains order."""

        class Base(Schema):
            base_field: str = "base"

        class Sub(Base):
            sub_field: str = "sub"

        instance = Sub()
        # Must use sort_keys=False to preserve order
        yaml_str = instance.model_dump_yaml(sort_keys=False)

        lines = [line for line in yaml_str.split("\n") if line and not line.startswith("#")]
        field_names = [line.split(":")[0].strip() for line in lines if ":" in line]

        assert field_names == ["sub_field", "base_field"]

    def test_with_optional_fields(self):
        """Test ordering with optional fields and None values."""

        class Base(Schema):
            base_required: str
            base_optional: str | None = None

        class Sub(Base):
            sub_required: str
            sub_optional: int | None = None

        instance = Sub(base_required="req1", sub_required="req2")
        keys = list(instance.model_dump().keys())

        expected = ["sub_required", "sub_optional", "base_required", "base_optional"]
        assert keys == expected

    def test_exclude_none(self):
        """Test ordering when excluding None values."""

        class Base(Schema):
            base_field: str = "base"
            base_none: str | None = None

        class Sub(Base):
            sub_field: str = "sub"
            sub_none: str | None = None

        instance = Sub()
        dumped = instance.model_dump(exclude_none=True)
        keys = list(dumped.keys())

        # Only non-None fields, still in subclass-first order
        assert keys == ["sub_field", "base_field"]

    def test_with_nested_schemas(self):
        """Test ordering with nested schema objects."""

        class Inner(Schema):
            inner_a: str = "a"

        class Base(Schema):
            base_field: str = "base"

        class Sub(Base):
            sub_field: str = "sub"
            nested: Inner = Inner()

        instance = Sub()
        keys = list(instance.model_dump().keys())

        assert keys == ["sub_field", "nested", "base_field"]

    def test_mode_parameter(self):
        """Test that mode parameter doesn't affect ordering."""

        class Base(Schema):
            base_field: str = "base"

        class Sub(Base):
            sub_field: str = "sub"

        instance = Sub()

        json_keys = list(instance.model_dump(mode="json").keys())
        python_keys = list(instance.model_dump(mode="python").keys())

        expected = ["sub_field", "base_field"]

        assert json_keys == expected
        assert python_keys == expected

    def test_exclude_unset(self):
        """Test ordering with exclude_unset parameter."""

        class Base(Schema):
            base_field: str = "base"
            base_unset: str | None = None

        class Sub(Base):
            sub_field: str = "sub"
            sub_unset: str | None = None

        # Explicitly set only some fields to test exclude_unset
        instance = Sub(sub_field="custom_sub", base_field="custom_base")
        dumped = instance.model_dump(exclude_unset=True)
        keys = list(dumped.keys())

        # Only set fields, in subclass-first order
        assert keys == ["sub_field", "base_field"]

    def test_with_field_alias(self):
        """Test ordering with field aliases."""

        class Base(Schema):
            base_field: str = Field(default="base", alias="base_alias")

        class Sub(Base):
            sub_field: str = Field(default="sub", alias="sub_alias")

        instance = Sub()
        keys = list(instance.model_dump(by_alias=False).keys())
        assert keys == ["sub_field", "base_field"]

    def test_multiple_inheritance_same_level(self):
        """Test ordering with multiple inheritance at the same level."""

        class Mixin1(Schema):
            mixin1_field: str = "m1"

        class Mixin2(Schema):
            mixin2_field: str = "m2"

        class Combined(Mixin1, Mixin2):
            combined_field: str = "combined"

        instance = Combined()
        keys = list(instance.model_dump().keys())
        # Combined class fields first, then mixins in MRO order
        assert keys[0] == "combined_field"
        assert "mixin1_field" in keys
        assert "mixin2_field" in keys

    def test_preserve_values(self):
        """Test that values are preserved during reordering."""

        class Base(Schema):
            base_a: str = "base_a_default"
            base_b: int = 10

        class Sub(Base):
            sub_x: str = "sub_x_default"
            sub_y: int = 20

        instance = Sub(base_a="custom_a", sub_x="custom_x", sub_y=99)
        dumped = instance.model_dump()
        assert list(dumped.keys()) == ["sub_x", "sub_y", "base_a", "base_b"]
        assert dumped["sub_x"] == "custom_x"
        assert dumped["sub_y"] == 99  # noqa: PLR2004
        assert dumped["base_a"] == "custom_a"
        assert dumped["base_b"] == 10  # noqa: PLR2004
