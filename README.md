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

Schemez is a Pydantic toolkit for turning Python code into schemas and schemas into code.
It builds OpenAI-compatible tool schemas from function signatures, creates Pydantic models
from JSON Schema, generates HTTP/MCP client code and FastAPI routes for tools, and gives your
config models YAML, markdown and test-data helpers.

## Installation

```bash
pip install schemez
```

Optional extras enable individual features:

| Extra | Enables |
| --- | --- |
| `yaml` | `Schema.from_yaml()`, `Schema.model_dump_yaml()` |
| `codegen` | `Schema.from_json_schema()`, `FunctionSchema.to_parameter_model_code()`, `jsonschema_to_code()`, `openapi_to_code()` |
| `tool_execution` | FastAPI routes and the HTTP tool server |
| `storage` | `ModelStore` (async SQLite storage) |
| `markdown` | Jinja templates for markdown docs |
| `ai` | `Schema.from_llm()`, `create_schema(..., mode="openai")`, `schemez.openapi` |

```bash
pip install "schemez[yaml,codegen]"
```

## Quick start

```python
from typing import Literal

from schemez import create_schema


def search_users(
    query: str,
    limit: int = 10,
    status: Literal["active", "inactive"] = "active",
) -> list[dict]:
    """Search for users.

    Args:
        query: Search query string
        limit: Maximum number of results
        status: User status filter
    """
    return []


schema = create_schema(search_users)
schema.name  # "search_users"
schema.description  # "Search for users."
schema.parameters["properties"]["status"]
# {'type': 'string', 'description': 'User status filter',
#  'enum': ['active', 'inactive'], 'default': 'active'}

schema.model_dump_openai()  # {"type": "function", "function": {...}}
schema.to_python_signature()
# (*, query: str, limit: int = 10, status: Literal['active', 'inactive'] = 'active') -> list[typing.Any]
```

## Features

### Function schemas

`create_schema()` returns a `FunctionSchema` (name, description, parameters, required, returns)
built from type hints and docstrings (Google, NumPy and Sphinx styles).

- `model_dump_openai()`: OpenAI function-calling tool definition
- `to_python_signature()`, `get_annotations()`: reconstruct the signature
- `create_parameter_model()`: Pydantic model for the parameters
- `to_parameter_model_code()` / `to_return_model_code()`: Python source for those models (`codegen` extra)
- `create_schema(func, exclude_types=[Context])` drops parameters of the given types (e.g. injected
  context objects); `name_override` / `description_override` replace name and docstring.
- `mode="simple"` (default) produces compact, LLM-friendly schemas: unions collapse to their first
  type, enums become plain value lists. `mode="jsonschema"` keeps Pydantic's full JSON Schema
  (`anyOf`, `$defs`), `mode="openai"` uses pydantic-ai's tool schema generator.

Bulk helpers: `create_schemas_from_callables({"name": func, ...})`, `create_schemas_from_class(cls)`,
`create_schemas_from_module(module, include_functions=[...])` and `create_constructor_schema(cls)`.

`create_executable(func)` wraps a function together with its schema and runs sync, async and
(async) generator functions uniformly via `run()`, `arun()` and `astream()`.

### `Schema` base model

`Schema` is a `pydantic.BaseModel` with `extra="forbid"` and attribute docstrings as field
descriptions, plus helpers:

```python
from schemez import Schema


class DatabaseConfig(Schema):
    host: str = "localhost"
    """Database host."""

    port: int = 5432
    """Database port."""


class AppConfig(Schema):
    name: str
    database: DatabaseConfig = DatabaseConfig()


config = AppConfig.from_yaml("name: my-app\ndatabase:\n  port: 6543\n")
print(config.model_dump_yaml(comments=True))
# database:
#   host: localhost  # Database host.
#   port: 6543  # Database port.
# name: my-app

config.model_dump_markdown()  # markdown table documenting the model and its values
AppConfig.generate_test_data(mode="minimal")  # valid instance generated from the schema
```

More class methods: `Schema.for_function(func)` and `Schema.for_class_ctor(cls)` build models from
signatures, `Schema.from_json_schema(json_schema)` builds one from JSON Schema (`codegen` extra),
`to_python_code()` emits model source code, and `merge()` / `iter_submodels()` work on instances.

### JSON Schema and OpenAPI to code

`jsonschema_to_code()` and `openapi_to_code()` generate Pydantic model source code (via
datamodel-code-generator). `SchemaDef` (`InlineSchemaDef`, `ImportedSchemaDef`) lets you declare
models in YAML/JSON config and build them with `get_schema()`.

### Tool code generation

`ToolsetCodeGenerator` turns a set of callables (or `FunctionSchema`s) into client code, LLM stubs
or FastAPI routes:

```python
from fastapi import FastAPI

from schemez import ToolsetCodeGenerator


def get_weather(location: str, unit: str = "C") -> dict:
    """Get weather for a location."""
    return {"temp": 22, "location": location}


generator = ToolsetCodeGenerator.from_callables([get_weather, search_users])

# Async httpx client functions calling a tool server
client_code = generator.generate_code(base_url="https://api.example.com", path_prefix="/tools")
# Signatures + docstrings only, for LLM prompts
stubs = generator.generate_code(output_type="stubs")
# Functions taking a generated Pydantic input model instead of keyword arguments
model_based = generator.generate_code(args_format="model")
# Human-readable overview of the available functions
print(generator.generate_tool_description())

# Expose the callables as FastAPI routes: GET /tools/get_weather, GET /tools/search_users
app = FastAPI()
generator.add_all_routes(app, path_prefix="/tools")
```

`ToolsetCodeGenerator.from_schemas([...])` works without implementations (code generation only),
`generate_mcp_code(server_configs)` emits FastMCP client code, and `ToolCodeGenerator` handles a
single tool. `schemez.tool_executor.HttpToolExecutor` serves tools over HTTP.

### Model storage

```python
from pydantic import BaseModel

from schemez import ModelStore


class Note(BaseModel):
    title: str
    tags: list[str] = []


async def main() -> None:
    async with ModelStore(Note, "notes.db") as store:
        note_id = await store.insert(Note(title="hello", tags=["a"]))
        await store.get(note_id)  # Note(title='hello', tags=['a'])
        await store.query(title="hello")

    # Reopen later: the model type is rebuilt from the stored JSON schema
    async with await ModelStore.open("notes.db") as store:
        await store.all()
```

### Markdown documentation

`models_to_markdown_docs()` renders documentation for several models; the `jinjarope` entry point
registers the markdown helpers as Jinja filters.

## Documentation

Full API reference: https://phil65.github.io/schemez/

## License

MIT
