# Schemez

[![PyPI License](https://img.shields.io/pypi/l/schemez.svg)](https://pypi.org/project/schemez/)
[![Package status](https://img.shields.io/pypi/status/schemez.svg)](https://pypi.org/project/schemez/)
[![Monthly downloads](https://img.shields.io/pypi/dm/schemez.svg)](https://pypi.org/project/schemez/)
[![Distribution format](https://img.shields.io/pypi/format/schemez.svg)](https://pypi.org/project/schemez/)
[![Wheel availability](https://img.shields.io/pypi/wheel/schemez.svg)](https://pypi.org/project/schemez/)
[![Python version](https://img.shields.io/pypi/pyversions/schemez.svg)](https://pypi.org/project/schemez/)
[![Implementation](https://img.shields.io/pypi/implementation/schemez.svg)](https://pypi.org/project/schemez/)
[![Releases](https://img.shields.io/github/downloads/phil65/schemez/total.svg)](https://github.com/phil65/schemez/releases)
[![Github Contributors](https://img.shields.io/github/contributors/phil65/schemez)](https://github.com/phil65/schemez/graphs/contributors)
[![Github Discussions](https://img.shields.io/github/discussions/phil65/schemez)](https://github.com/phil65/schemez/discussions)
[![Github Forks](https://img.shields.io/github/forks/phil65/schemez)](https://github.com/phil65/schemez/forks)
[![Github Issues](https://img.shields.io/github/issues/phil65/schemez)](https://github.com/phil65/schemez/issues)
[![Github Issues](https://img.shields.io/github/issues-pr/phil65/schemez)](https://github.com/phil65/schemez/pulls)
[![Github Watchers](https://img.shields.io/github/watchers/phil65/schemez)](https://github.com/phil65/schemez/watchers)
[![Github Stars](https://img.shields.io/github/stars/phil65/schemez)](https://github.com/phil65/schemez/stars)
[![Github Repository size](https://img.shields.io/github/repo-size/phil65/schemez)](https://github.com/phil65/schemez)
[![Github last commit](https://img.shields.io/github/last-commit/phil65/schemez)](https://github.com/phil65/schemez/commits)
[![Github release date](https://img.shields.io/github/release-date/phil65/schemez)](https://github.com/phil65/schemez/releases)
[![Github language count](https://img.shields.io/github/languages/count/phil65/schemez)](https://github.com/phil65/schemez)
[![Github commits this month](https://img.shields.io/github/commit-activity/m/phil65/schemez)](https://github.com/phil65/schemez)
[![Package status](https://codecov.io/gh/phil65/schemez/branch/main/graph/badge.svg)](https://codecov.io/gh/phil65/schemez/)
[![PyUp](https://pyup.io/repos/github/phil65/schemez/shield.svg)](https://pyup.io/repos/github/phil65/schemez/)

[Read the documentation!](https://phil65.github.io/schemez/)



# OpenAI Function Schema Generator

Convert Python functions to OpenAI-compatible function schemas automatically.

## Installation

```bash
pip install schemez
```

## Basic Usage

```python
from schemez import create_schema
from typing import Literal

def get_weather(
    location: str,
    unit: Literal["C", "F"] = "C",
    detailed: bool = False,
) -> dict[str, str | float]:
    """Get the weather for a location.

    Args:
        location: City or address to get weather for
        unit: Temperature unit (Celsius or Fahrenheit)
        detailed: Include extended forecast
    """
    return {"temp": 22.5, "conditions": "sunny"}

# Create schema
schema = create_schema(get_weather)

# The schema.model_dump_openai() returns a TypedDict with the complete OpenAI tool definition:
# OpenAIFunctionTool = TypedDict({
#     "type": Literal["function"],
#     "function": OpenAIFunctionDefinition
# })

# Use with OpenAI
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in London?"}],
    tools=[schema.model_dump_openai()],  # Schema includes the type: "function" wrapper
    tool_choice="auto"
)
```

> **Note**: This library supports the OpenAI API v1 format (openai>=1.0.0). For older
> versions of the OpenAI package that use the legacy functions API, you'll need to
> unwrap the function definition using `schema.model_dump_openai()["function"]`.
```

## Supported Types

### Basic Types
```python
def func(
    text: str,              # -> "type": "string"
    number: int,            # -> "type": "integer"
    amount: float,          # -> "type": "number"
    enabled: bool,          # -> "type": "boolean"
    anything: Any,          # -> "type": "string"
) -> None: ...
```

### Container Types
```python
def func(
    items: list[str],                    # -> "type": "array", "items": {"type": "string"}
    numbers: set[int],                   # -> same as list
    mapping: dict[str, Any],            # -> "type": "object", "additionalProperties": true
    nested: list[dict[str, int]],       # -> nested array/object types
    sequence: Sequence[str],            # -> "type": "array"
    collection: Collection[int],        # -> "type": "array"
) -> None: ...
```

### Enums and Literals
```python
class Color(Enum):
    RED = "red"
    BLUE = "blue"

def func(
    color: Color,                       # -> "type": "string", "enum": ["red", "blue"]
    mode: Literal["fast", "slow"],      # -> "type": "string", "enum": ["fast", "slow"]
) -> None: ...
```

### Optional and Union Types
```python
def func(
    opt1: str | None,                   # -> "type": "string"
    opt2: int | None,                   # -> "type": "integer"
    union: str | int,                   # -> "type": "string" (first type)
) -> None: ...
```

### Custom Types
```python
@dataclass
class User:
    name: str
    age: int

def func(
    user: User,                         # -> "type": "object"
    data: JsonDict,                     # -> "type": "object"
) -> None: ...
```

### Type Aliases
```python
JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None
JsonDict = dict[str, JsonValue]

def func(
    data: JsonDict,                     # -> "type": "object"
    values: list[JsonValue],            # -> "type": "array"
) -> None: ...
```

### Recursive Types
```python
def func(
    tree: dict[str, "dict[str, Any] | str"],  # -> "type": "object"
    nested: dict[str, list["dict[str, Any]"]], # -> "type": "object"
) -> None: ...
```

## Generated Schema Example

```python
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or address to get weather for"
                },
                "unit": {
                    "type": "string",
                    "enum": ["C", "F"],
                    "description": "Temperature unit (Celsius or Fahrenheit)",
                    "default": "C"
                },
                "detailed": {
                    "type": "boolean",
                    "description": "Include extended forecast",
                    "default": false
                }
            },
            "required": ["location"]
        }
    }
}
```

## Schema Generators

### Module Schemas

You can generate schemas for all public functions in a module using `create_schemas_from_module`:

```python
from schemez import create_schemas_from_module
import math

# Generate schemas for all public functions
schemas = create_schemas_from_module(math)

# Generate schemas for specific functions only
schemas = create_schemas_from_module(math, include_functions=['sin', 'cos'])

# Import module by string name
schemas = create_schemas_from_module('math')
```

### Class Schemas

Generate schemas for all public methods in a class using `create_schemas_from_class`:

```python
from schemez import create_schemas_from_class

class Calculator:
    def add(self, x: int, y: int) -> int:
        """Add two numbers.

        Args:
            x: First number
            y: Second number

        Returns:
            Sum of x and y
        """
        return x + y

    @classmethod
    def multiply(cls, x: int, y: int) -> int:
        """Multiply two numbers.

        Args:
            x: First number
            y: Second number

        Returns:
            Product of x and y
        """
        return x * y

    @staticmethod
    def divide(x: float, y: float) -> float:
        """Divide two numbers.

        Args:
            x: Numerator
            y: Denominator

        Returns:
            Result of x divided by y
        """
        return x / y

# Generate schemas for all public methods
schemas = create_schemas_from_class(Calculator)

# Access individual method schemas
add_schema = schemas['Calculator.add']
multiply_schema = schemas['Calculator.multiply']
divide_schema = schemas['Calculator.divide']
```

The schema generators support:

- Regular functions
- Regular instance methods (bound and unbound)
- Class methods
- Static methods
- Decorated functions / methods
- Async functions / methods
- Property methods
- Basically all stdlib typing features as well as many stdlib types
- Method docstrings for descriptions
- Default values
- Return type hints


## Diferences to pydantic schema generation

While Pydantics schema generation preserves detailed type information, `schema.model_dump_openai()`
simplifies types to match OpenAI's function calling format. Most special types
(datetime, UUID, Path, etc.) are handled similarly by both (we only strip unused information), but we handle enums
differently: Instead of preserving enum class information, we extract just the values
as a string enum. Union types and Optionals are also handled differently - we typically
pick the first type to keep the schema simple and practical for AI interaction.
This ensures compatibility with OpenAI's function calling API while maintaining enough
type information for the AI to understand the function signature.
