"""Test models_to_markdown_docs helper function."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
import pytest

from schemez.helpers import models_to_markdown_docs


class SimpleModel(BaseModel):
    """A simple test model for demonstration."""

    name: str = Field(description="The name of the item")
    count: int = Field(default=1, description="How many items")
    active: bool = Field(default=True, description="Whether the item is active")


class ComplexModel(BaseModel):
    """A more complex model with various field types."""

    title: str = Field(description="The title")
    tags: list[str] = Field(default_factory=list, description="List of tags")
    metadata: dict[str, str] = Field(default_factory=dict, description="Metadata mapping")
    status: Literal["draft", "published", "archived"] = Field(
        default="draft", description="Publication status"
    )


def test_single_model_markdown():
    """Test generating markdown for a single model."""
    result = models_to_markdown_docs(SimpleModel)

    assert "## SimpleModel" in result
    assert "A simple test model for demonstration." in result
    assert "```yaml" in result
    assert "```" in result
    # Should contain YAML with comments
    assert "name:" in result
    assert "# The name of the item" in result


def test_multiple_models_markdown():
    """Test generating markdown for multiple models."""
    result = models_to_markdown_docs(SimpleModel, ComplexModel)

    assert "## SimpleModel" in result
    assert "## ComplexModel" in result
    assert result.count("```yaml") == 2  # noqa: PLR2004
    assert result.count("```") == 4  # noqa: PLR2004  # 2 opening, 2 closing

    # Check both models are documented
    assert "A simple test model for demonstration." in result
    assert "A more complex model with various field types." in result


def test_custom_header_level():
    """Test custom header level."""
    result = models_to_markdown_docs(SimpleModel, header_level=3)

    assert "### SimpleModel" in result
    assert result.startswith("### SimpleModel")


def test_minimal_expand_mode():
    """Test minimal expand mode."""
    result = models_to_markdown_docs(SimpleModel, expand_mode="minimal")

    assert "## SimpleModel" in result
    assert "```yaml" in result
    # In minimal mode, should have basic required fields
    assert "name:" in result


def test_maximal_expand_mode():
    """Test maximal expand mode."""
    result = models_to_markdown_docs(ComplexModel, expand_mode="maximal")

    assert "## ComplexModel" in result
    assert "```yaml" in result
    # In maximal mode, should populate all fields including optional ones
    assert "tags:" in result
    assert "metadata:" in result


def test_model_without_docstring():
    """Test model without docstring."""

    class NoDocModel(BaseModel):
        field: str = Field(description="A field")

    result = models_to_markdown_docs(NoDocModel)

    assert "## NoDocModel" in result
    assert "```yaml" in result
    # Should still generate YAML even without docstring
    assert "field:" in result


def test_yaml_options():
    """Test various YAML formatting options."""
    result = models_to_markdown_docs(SimpleModel, exclude_defaults=True, sort_keys=False, indent=4)

    assert "## SimpleModel" in result
    assert "```yaml" in result
    # Should still contain the required field
    assert "name:" in result


@pytest.mark.parametrize("expand_mode", ["minimal", "maximal", "default"])
def test_all_expand_modes(expand_mode):
    """Test all expand modes work without errors."""
    result = models_to_markdown_docs(SimpleModel, expand_mode=expand_mode)

    assert "## SimpleModel" in result
    assert "```yaml" in result
    assert "name:" in result
