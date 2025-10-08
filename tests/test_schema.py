from __future__ import annotations

from pydantic import BaseModel
import pytest

from schemez.helpers import model_to_python_code
from schemez.schema import Schema


class SimpleModel(BaseModel):
    name: str
    age: int
    active: bool = True


@pytest.mark.asyncio
async def test_model_to_python_code():
    """Test converting a BaseModel to Python code."""
    try:
        code = await model_to_python_code(SimpleModel)

        # Basic checks for generated code
        assert "class SimpleModel" in code
        assert "name: str" in code
        assert "age: int" in code
        assert "active: bool" in code
        assert "from pydantic import BaseModel" in code

    except RuntimeError as e:
        if "datamodel-codegen not available" in str(e):
            pytest.skip("datamodel-codegen not installed")
        raise


@pytest.mark.asyncio
async def test_schema_to_python_code_method():
    """Test the to_python_code method on Schema instances."""

    class TestSchema(Schema):
        title: str
        count: int = 0

    schema_instance = TestSchema(title="test", count=42)

    try:
        code = await schema_instance.to_python_code(class_name="GeneratedSchema")

        # Check custom class name is used
        assert "class GeneratedSchema" in code
        assert "title: str" in code
        assert "count: int" in code

    except RuntimeError as e:
        if "datamodel-codegen not available" in str(e):
            pytest.skip("datamodel-codegen not installed")
        raise
