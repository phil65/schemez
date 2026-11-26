from pydantic import BaseModel

from schemez.helpers import model_to_python_code
from schemez.schema import Schema


class SimpleModel(BaseModel):
    name: str
    age: int
    active: bool = True


async def test_model_to_python_code():
    """Test converting a BaseModel to Python code."""
    code = model_to_python_code(SimpleModel)
    assert "class SimpleModel" in code
    assert "name: str" in code
    assert "age: int" in code
    assert "active: bool" in code
    assert "from pydantic import BaseModel" in code


async def test_schema_to_python_code_method():
    """Test the to_python_code method on Schema instances."""

    class TestSchema(Schema):
        title: str
        count: int = 0

    schema_instance = TestSchema(title="test", count=42)
    code = schema_instance.to_python_code(class_name="GeneratedSchema")
    assert "class GeneratedSchema" in code
    assert "title: str" in code
    assert "count: int" in code


def test_get_function_model_respects_name_and_ctor_name():
    """Custom model name is respected by for_function and used by for_class_ctor."""

    def greet(who: str) -> str:
        return f"hi {who}"

    class Person:
        def __init__(self, full_name: str, age: int = 0) -> None:
            self.full_name = full_name
            self.age = age

    assert Schema.for_function(greet).__name__ == "greetParams"
    assert Schema.for_function(greet, name="GreetInput").__name__ == "GreetInput"
    assert Schema.for_class_ctor(Person).__name__ == "Person"
