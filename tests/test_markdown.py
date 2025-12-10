"""Tests for markdown conversion functionality."""

from __future__ import annotations

from pydantic import Field

from schemez import Schema
from schemez.markdown import (
    DEFAULT_TEMPLATE,
    instance_to_markdown,
    model_to_markdown,
    schema_to_markdown_context,
)


class Inner(Schema):
    """A nested model."""

    value: int = Field(default=0, description="Inner value")


class Outer(Schema):
    """An outer model with nested schema."""

    name: str = Field(min_length=1, max_length=50, description="Name field")
    count: int = Field(default=0, ge=0, description="A counter")
    inner: Inner | None = Field(default=None, examples=[{"value": 42}])


def test_schema_to_markdown_context_basic():
    """Test context extraction from schema."""
    context = schema_to_markdown_context(Outer)

    assert context["header_level"] == 1
    assert context["root"]["name"] == "Outer"
    assert context["root"]["description"] == "An outer model with nested schema."
    assert len(context["root"]["fields"]) == 3  # noqa: PLR2004
    assert "Inner" in context["nested_models"]


def test_schema_to_markdown_context_fields():
    """Test field extraction in context."""
    context = schema_to_markdown_context(Outer)
    fields = {f["name"]: f for f in context["root"]["fields"]}

    assert fields["name"]["required"] is True
    assert fields["name"]["type"] == "str"
    assert fields["name"]["description"] == "Name field"

    assert fields["count"]["required"] is False
    assert fields["count"]["default"] == "0"

    assert fields["inner"]["required"] is False
    assert "Inner" in fields["inner"]["type"]


def test_schema_to_markdown_context_constraints():
    """Test constraint extraction."""
    context = schema_to_markdown_context(Outer)

    assert "name" in context["root"]["constraints"]
    assert "min_length=`1`" in context["root"]["constraints"]["name"]
    assert "max_length=`50`" in context["root"]["constraints"]["name"]

    assert "count" in context["root"]["constraints"]
    assert "min=`0`" in context["root"]["constraints"]["count"]


def test_schema_to_markdown_context_examples():
    """Test example extraction."""
    context = schema_to_markdown_context(Outer)

    assert "inner" in context["root"]["examples"]
    assert {"value": 42} in context["root"]["examples"]["inner"]


def test_model_to_markdown_basic():
    """Test basic markdown generation."""
    md = model_to_markdown(Outer)

    assert "# Outer" in md
    assert "An outer model with nested schema." in md
    assert "| Name | Type | Required | Default | Description |" in md
    assert "`name`" in md
    assert "`count`" in md
    assert "`inner`" in md


def test_model_to_markdown_nested_models():
    """Test nested model section in markdown."""
    md = model_to_markdown(Outer)

    assert "# Nested Models" in md
    assert "## Inner" in md
    assert "A nested model." in md


def test_model_to_markdown_constraints_section():
    """Test constraints section in markdown."""
    md = model_to_markdown(Outer)

    assert "## Constraints" in md
    assert "min_length=`1`" in md
    assert "max_length=`50`" in md


def test_model_to_markdown_examples_section():
    """Test examples section in markdown."""
    md = model_to_markdown(Outer)

    assert "## Examples" in md


def test_model_to_markdown_header_level():
    """Test custom header level."""
    md = model_to_markdown(Outer, header_level=2)

    assert "## Outer" in md
    assert "### Inner" in md


def test_model_to_markdown_exclude_constraints():
    """Test excluding constraints section."""
    md = model_to_markdown(Outer, include_constraints=False)

    assert "Constraints" not in md


def test_model_to_markdown_exclude_examples():
    """Test excluding examples section."""
    md = model_to_markdown(Outer, include_examples=False)

    assert "Examples" not in md


def test_instance_to_markdown_basic():
    """Test instance markdown generation."""
    instance = Outer(name="test", count=5, inner=Inner(value=10))
    md = instance_to_markdown(instance)

    assert "# Outer" in md
    assert "# Current Values" in md
    assert "`name`" in md
    assert '"test"' in md


def test_instance_to_markdown_without_values():
    """Test instance markdown without values section."""
    instance = Outer(name="test")
    md = instance_to_markdown(instance, include_values=False)

    assert "Current Values" not in md


def test_schema_dump_markdown_schema_method():
    """Test class method on Schema."""
    md = Outer.dump_markdown_schema()

    assert "# Outer" in md
    assert "An outer model with nested schema." in md


def test_schema_model_dump_markdown_method():
    """Test instance method on Schema."""
    instance = Outer(name="test")
    md = instance.model_dump_markdown()

    assert "# Outer" in md
    assert "# Current Values" in md


def test_custom_template():
    """Test using a custom template."""
    custom_template = "# {{ root.name }}\n\nField count: {{ root.fields | length }}\n"
    md = model_to_markdown(Outer, template=custom_template)

    assert "# Outer" in md
    assert "Field count: 3" in md


def test_required_field_marker():
    """Test required field marker in table."""
    md = model_to_markdown(Outer)

    # name is required, should have checkmark
    lines = md.split("\n")
    name_line = next(line for line in lines if "`name`" in line)
    assert "✓" in name_line


def test_optional_field_no_marker():
    """Test optional field has no marker."""
    md = model_to_markdown(Outer)

    lines = md.split("\n")
    count_line = next(line for line in lines if "`count`" in line)
    # count has a default so not required - no checkmark after the type column
    assert count_line.count("✓") == 0


def test_no_excessive_blank_lines():
    """Test that output doesn't have excessive blank lines."""
    md = model_to_markdown(Outer)

    assert "\n\n\n" not in md


def test_default_template_is_valid():
    """Test that default template can be parsed."""
    from jinja2 import Environment

    env = Environment(autoescape=False)
    # Should not raise
    env.from_string(DEFAULT_TEMPLATE)


def test_model_to_markdown_python_code_mode():
    """Test python_code display mode."""
    md = model_to_markdown(Outer, display_mode="python_code")

    assert md.startswith("```")
    assert "class Outer(Schema):" in md
    assert '"""An outer model with nested schema."""' in md
    assert "name: str = Field" in md


def test_model_to_markdown_yaml_mode():
    """Test yaml display mode."""
    md = model_to_markdown(Outer, display_mode="yaml", seed=42)

    assert md.startswith("```")
    assert ".yaml#L" in md
    assert "name:" in md
    assert "count:" in md
    # Should have comments from schema
    assert "# Name field" in md or "# A counter" in md


def test_instance_to_markdown_python_code_mode():
    """Test python_code display mode for instances."""
    instance = Outer(name="test", count=5)
    md = instance_to_markdown(instance, display_mode="python_code")

    assert md.startswith("```")
    assert "class Outer(Schema):" in md


def test_instance_to_markdown_yaml_mode():
    """Test yaml display mode for instances."""
    instance = Outer(name="test", count=5, inner=Inner(value=10))
    md = instance_to_markdown(instance, display_mode="yaml")

    assert md.startswith("```")
    assert ".yaml#L" in md
    assert 'name: "test"' in md or "name: test" in md
    assert "count: 5" in md
    assert "inner:" in md
    assert "value: 10" in md


def test_schema_class_methods_with_display_mode():
    """Test Schema class methods support display_mode parameter."""
    # Test class method
    md_python = Outer.dump_markdown_schema(display_mode="python_code")
    assert "class Outer(Schema):" in md_python

    md_yaml = Outer.dump_markdown_schema(display_mode="yaml", seed=42)
    assert "name:" in md_yaml

    # Test instance method
    instance = Outer(name="test")
    md_python = instance.model_dump_markdown(display_mode="python_code")
    assert "class Outer(Schema):" in md_python

    md_yaml = instance.model_dump_markdown(display_mode="yaml")
    assert 'name: "test"' in md_yaml or "name: test" in md_yaml


def test_header_style_default():
    """Test default header style."""
    md = model_to_markdown(Outer, display_mode="python_code", header_style="default")

    # Default style uses path#L1-N format
    assert "#L1-" in md
    assert "tests.test_markdown.Outer" in md


def test_header_style_pymdownx():
    """Test pymdownx header style."""
    md = model_to_markdown(Outer, display_mode="python_code", header_style="pymdownx")

    # Pymdownx style uses title and linenums
    assert 'title="Outer"' in md
    assert 'linenums="1"' in md
    assert "```Outer" in md or "```py" in md


def test_header_style_yaml_mode():
    """Test header styles with yaml display mode."""
    # Default style
    md_default = model_to_markdown(Outer, display_mode="yaml", header_style="default", seed=42)
    assert ".yaml#L1-" in md_default

    # Pymdownx style
    md_pymdownx = model_to_markdown(Outer, display_mode="yaml", header_style="pymdownx", seed=42)
    assert 'title="Outer (YAML)"' in md_pymdownx
    assert "```yaml" in md_pymdownx


def test_header_style_instance():
    """Test header styles with instance markdown."""
    instance = Outer(name="test", count=5)

    # Default style
    md_default = instance_to_markdown(instance, display_mode="python_code", header_style="default")
    assert "#L1-" in md_default

    # Pymdownx style
    md_pymdownx = instance_to_markdown(
        instance, display_mode="python_code", header_style="pymdownx"
    )
    assert 'title="Outer"' in md_pymdownx
    assert 'linenums="1"' in md_pymdownx
