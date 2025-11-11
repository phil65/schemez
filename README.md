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

A powerful toolkit for Python function schema generation and code generation. Extract schemas from functions, generate OpenAI-compatible tools, create HTTP clients, and set up FastAPI routes - all from your function signatures.

## Installation

```bash
pip install schemez
```

## Quick Start

```python
from schemez import create_schema

def get_weather(location: str, unit: str = "C") -> dict:
    """Get weather for a location."""
    return {"temp": 22, "location": location}

# Create schema from function
schema = create_schema(get_weather)
print(schema.name)          # "get_weather"
print(schema.description)   # "Get weather for a location."
print(schema.parameters)    # Full parameter schema
```

## FunctionSchema - The Core

The `FunctionSchema` class is the heart of schemez, providing a rich representation of Python functions with powerful methods for introspection and code generation.

### Schema Creation

```python
from schemez import create_schema
from typing import Literal

def search_users(
    query: str,
    limit: int = 10,
    status: Literal["active", "inactive"] = "active",
    include_details: bool = False
) -> list[dict]:
    """Search for users with filters.
    
    Args:
        query: Search query string
        limit: Maximum number of results
        status: User status filter
        include_details: Include detailed user information
    """
    return []

# Create schema
schema = create_schema(search_users)
```

### Key Methods

#### schema output
```python
# Get OpenAI-compatible tool definition
openai_tool = schema.model_dump_openai()
# Returns: {"type": "function", "function": {...}}
```

#### Code Generation
```python
# Generate Python function signature
signature = schema.to_python_signature()
# Returns: "(*, query: str, limit: int = 10, status: Literal['active', 'inactive'] = 'active', include_details: bool = False) -> list[dict]"

# Generate Pydantic model for return type
model_code = schema.to_pydantic_model_code("SearchResponse")
# Returns Python code string for a Pydantic model
```

#### Schema Inspection
```python
# Access schema components
print(schema.name)                    # Function name
print(schema.description)             # Function docstring
print(schema.parameters)              # Parameter schema dict
print(schema.returns)                 # Return type schema
print(schema.get_annotations())       # Python type annotations
```

### Bulk Schema Generation

```python
from schemez.functionschema import (
    create_schemas_from_module,
    create_schemas_from_class,
    create_schemas_from_callables
)

# From module
import math
schemas = create_schemas_from_module(math, include_functions=['sin', 'cos'])

# From class
class Calculator:
    def add(self, x: int, y: int) -> int:
        """Add two numbers."""
        return x + y
    
    def multiply(self, x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y

schemas = create_schemas_from_class(Calculator)

# From callable list
functions = [get_weather, search_users]
schemas = create_schemas_from_callables({"weather": get_weather, "search": search_users})
```

## Code Generation - Powerful Automation

Transform your schemas into executable code for different contexts: HTTP clients, FastAPI routes, and Python execution environments.

### HTTP Client Generation

Generate complete HTTP client code from schemas:

```python
from schemez.code_generation import ToolsetCodeGenerator

# Create from functions
functions = [get_weather, search_users]
generator = ToolsetCodeGenerator.from_callables(functions)

# Generate HTTP client code (multiple modes available)
client_code = generator.generate_code(
    mode="models",  # Rich Pydantic models for validation
    base_url="https://api.example.com",
    path_prefix="/v1/tools"
)

# Available modes:
# - "models": Rich Pydantic models with validation
# - "simple": Clean natural function signatures  
# - "stubs": Function stubs for LLM consumption

# Generated code includes:
# - Pydantic input models for each function
# - Async HTTP wrapper functions
# - Type-safe parameter validation
# - Complete module with imports and exports
```

Example generated client:
```python
"""Generated HTTP client tools."""

from __future__ import annotations
from pydantic import BaseModel
import httpx

class GetWeatherInput(BaseModel):
    location: str
    unit: str | None = 'C'

async def get_weather(input: GetWeatherInput) -> str:
    """Get weather for a location."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.example.com/v1/tools/get_weather",
            params=input.model_dump(),
            timeout=30.0
        )
        response.raise_for_status()
        return response.text

__all__ = ['GetWeatherInput', 'get_weather', ...]
```

### FastAPI Route Setup

Automatically create FastAPI routes from your functions:

```python
from fastapi import FastAPI
from schemez.code_generation import ToolsetCodeGenerator

app = FastAPI()

# Create generator with actual callables for execution
generator = ToolsetCodeGenerator.from_callables([get_weather, search_users])

# Add all routes automatically
generator.add_all_routes(app, path_prefix="/api/tools")

# Creates routes:
# GET /api/tools/get_weather
# GET /api/tools/search_users
```

### Python Execution Environment

Create sandboxed execution environments with tool functions:

```python
# Generate different modes of client code
models_code = generator.generate_code(mode="models")    # With Pydantic models
simple_code = generator.generate_code(mode="simple")    # Clean signatures
stubs_code = generator.generate_code(mode="stubs")      # For LLM consumption

# For advanced use, get structured components
structured = generator.generate_structured_code()
print(structured.models)          # Just the Pydantic models
print(structured.clean_methods)   # Just the clean method signatures
print(structured.exports)         # List of exported names

# Use any mode as needed
exec(simple_code, globals())
result = await get_weather(location="London", unit="F")
```

### Schema-Only Code Generation

Generate code without needing actual function implementations:

```python
from schemez import create_schema
from schemez.code_generation import ToolsetCodeGenerator

# Create schemas from function signatures only
schema1 = create_schema(get_weather)
schema2 = create_schema(search_users)

# Generate client code from schemas
generator = ToolsetCodeGenerator.from_schemas([schema1, schema2])
client_code = generator.generate_code(mode="models")

# Works for client generation, signatures, models
# Routes require actual callables for execution
```

### Tool Documentation

Generate comprehensive documentation for your tools:

```python
# Generate tool descriptions with signatures and docstrings
documentation = generator.generate_tool_description()

# Generate different formats for different needs
stubs = generator.generate_code(mode="stubs")        # Clean stubs for LLMs
simple = generator.generate_code(mode="simple")      # Natural signatures
models = generator.generate_code(mode="models")      # Rich validation

# Includes:
# - Function signatures with type hints  
# - Docstrings and parameter descriptions
# - Usage examples and constraints
# - Available return type models
```

## Code Generation Modes

Schemez offers three distinct generation modes to suit different use cases:

### Models Mode (`mode="models"`)
**Best for**: Production APIs, complex validation, detailed type safety

```python
client_code = generator.generate_code(mode="models")
```

Generated code includes:
- Full Pydantic input models with validation constraints
- Rich type information (min/max values, string patterns, etc.)
- HTTP client functions that accept model instances
- Comprehensive error handling and type checking

Example generated interface:
```python
class SearchUsersInput(BaseModel):
    query: str = Field(min_length=1, description="Search query")
    limit: int = Field(ge=1, le=100, default=10)
    
async def search_users(input: SearchUsersInput) -> str:
    # HTTP client implementation
```

### Simple Mode (`mode="simple"`)  
**Best for**: LLM integration, natural code execution, clean interfaces

```python
client_code = generator.generate_code(mode="simple")
```

Generated code features:
- Clean, natural function signatures
- Direct parameter passing (no model wrappers)
- Minimal boilerplate for easy LLM consumption
- Intuitive function calls

Example generated interface:
```python
async def search_users(*, query: str, limit: int = 10, status: str = "active") -> str:
    # HTTP client implementation with automatic parameter handling
```

### Stubs Mode (`mode="stubs"`)
**Best for**: Documentation, LLM prompts, API exploration

```python
stubs_code = generator.generate_code(mode="stubs")
```

Generated code provides:
- Function signatures with type hints
- Complete docstrings and parameter descriptions  
- Input models for reference
- No implementation details (just `...` bodies)

Example generated interface:
```python
class SearchUsersInput(BaseModel):
    query: str
    limit: int = 10
    
async def search_users(input: SearchUsersInput) -> list[dict]:
    """Search for users with filters.
    
    Args:
        input: Function parameters
        
    Returns:
        List of user dictionaries
    """
    ...
```

## Advanced Features

### Type Support

Schemez handles complex Python types:

```python
from typing import Literal, Optional, Union
from enum import Enum
from dataclasses import dataclass

class Status(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

@dataclass
class User:
    name: str
    email: str

def complex_function(
    users: list[User],                    # -> Nested object arrays
    status: Status,                       # -> Enum values
    mode: Literal["fast", "detailed"],    # -> String literals
    metadata: dict[str, Any],             # -> Objects with any properties
    optional: Optional[str] = None,       # -> Optional parameters
) -> Union[dict, list]:                   # -> Union return types
    """Handle complex types."""
    pass

schema = create_schema(complex_function)
```

### Configuration

Fine-tune schema generation:

```python
# Exclude specific parameter types (e.g., context objects)
schema = create_schema(my_function, exclude_types=[Context, Session])

# Override names and descriptions
schema = create_schema(
    my_function,
    name_override="custom_name",
    description_override="Custom description"
)
```

### Error Handling

Robust error handling throughout:

```python
from schemez.code_generation import ToolCodeGenerator

# Schema-only generator (no execution capability)
generator = ToolCodeGenerator.from_schema(schema)

try:
    # This will fail with clear error message
    generator.add_route_to_app(app)
except ValueError as e:
    print(e)  # "Callable required for route generation for tool 'my_function'"
```

## Use Cases

- **AI Tool Integration**: Convert functions to OpenAI-compatible tools
- **API Client Generation**: Create type-safe HTTP clients from schemas  
- **FastAPI Automation**: Auto-generate routes with validation
- **Documentation**: Generate comprehensive API docs
- **Testing**: Create mock implementations and test data
- **Code Analysis**: Extract and analyze function signatures
- **Dynamic Execution**: Build sandboxed Python environments

## Differences from Pydantic

While Pydantic focuses on detailed type preservation, schemez optimizes for practical AI interaction:

- **Simplified unions**: Takes first type instead of complex anyOf schemas
- **Enum flattening**: Extracts enum values as simple string arrays
- **AI-optimized**: Generates schemas that work well with LLM function calling
- **Code generation focus**: Built for generating executable code, not just validation

## Contributing

Contributions welcome! This library consolidates schema and type utilities from multiple projects into a unified toolkit.
