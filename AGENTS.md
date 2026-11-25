# AGENTS.md

## Overview

`schemez` is a Pydantic toolkit: a `Schema` base model with YAML/markdown/codegen helpers,
conversion between Python callables, JSON Schema and OpenAI function-calling schemas,
dynamic model creation from JSON Schema / YAML definitions, and code generation for tool
clients/servers. It is a dependency of several other projects (agentpool, llmling, docler, ...),
so treat everything exported from `src/schemez/__init__.py` as public API and avoid breaking changes.

## Architecture

All code lives in `src/schemez/` (uv_build, `src` layout). `__init__.py` re-exports the public API.

- `schema.py`: `Schema` (base `BaseModel`, `extra="forbid"`, attribute docstrings, subclass-fields-first
  serialization) with `from_yaml`, `model_dump_yaml`, `model_dump_markdown`, `for_function`,
  `from_json_schema`, `for_class_ctor`, `from_llm` (needs `agentpool`), `to_python_code`,
  `generate_test_data`, `merge`, `iter_submodels`. `ConfigSchema` adds `required_python_packages`.
- `functionschema/`: callable -> schema.
  - `functionschema.py`: `FunctionSchema` and `create_schema(func, mode="simple"|"openai"|"jsonschema")`.
    `simple` is the custom implementation; the others go through pydantic / pydantic-ai internals.
  - `helpers.py` (type -> JSON schema property mapping, `determine_function_type`), `typedefs.py`
    (TypedDicts: `OpenAIFunctionTool`, `Property`, ...), `schema_generators.py`
    (`create_schemas_from_callables/_module/_class`, `create_constructor_schema`),
    `executable.py` (`ExecutableFunction` / `create_executable`: run sync/async/generator functions uniformly).
- `schema_to_type/`: JSON Schema -> Python types. `main_impl.py` (`json_schema_to_pydantic_class`, via
  datamodel-code-generator), `simple_impl.py` (`json_schema_to_base_model`, lightweight),
  `marvin_impl.py` (`jsonschema_to_type`, dataclass-based).
- `schemadef/schemadef.py`: declarative model definitions (`SchemaField`, `InlineSchemaDef`,
  `ImportedSchemaDef`), `get_schema()` builds a model via `pydantic.create_model`.
- `convert.py`: `get_function_model` / `get_ctor_basemodel` (callables/classes -> `Schema` subclasses).
- `helpers.py`: codegen (`json_schema_to_pydantic_code`, `openapi_to_code`, `jsonschema_to_code`,
  `model_to_python_code`), `import_callable`/`import_class`, `merge_models`, `iter_submodel(_type)s`,
  `models_to_markdown_docs`.
- `markdown.py`: model/instance/union -> markdown docs; `setup_jinjarope_filters` is registered as the
  `jinjarope.environment` entry point in `pyproject.toml`.
- `code_generation/`: `ToolCodeGenerator` (one callable/schema -> signature, models, FastAPI route),
  `ToolsetCodeGenerator` (many tools -> client code, MCP client code, namespaces),
  `route_helpers.py`, `namespace_callable.py`.
- `tool_executor/executor.py`: `HttpToolExecutor` generates and serves HTTP tool endpoints (FastAPI + uvicorn).
- `openapi/`: OpenAPI spec loading (`loader.py`), `$ref` resolution (`resolver.py`), callables from
  operations (`callable_factory.py`).
- `storage/model_store.py`: `ModelStore[T]`, async SQLite persistence of models (aiosqlite).
- Smaller modules: `code.py` (`PythonCode`/`JSONCode`/`TOMLCode`/`YAMLCode` validated code strings),
  `commented_yaml.py` (YAML with field descriptions as comments), `generators.py` (`SchemaDataGenerator`,
  test data from JSON schema), `docstrings.py` (griffe-based docstring parsing), `bind_kwargs.py`,
  `pydantic_types.py`, `log.py`.

## Where to look

| Task | Location |
| --- | --- |
| Base model behavior, YAML dump/load | `src/schemez/schema.py` |
| Function -> OpenAI/JSON schema | `src/schemez/functionschema/functionschema.py`, `functionschema/helpers.py` |
| Python type -> JSON schema mapping | `src/schemez/functionschema/helpers.py` |
| JSON Schema -> pydantic model | `src/schemez/schema_to_type/` |
| YAML-defined models | `src/schemez/schemadef/schemadef.py` |
| Model -> markdown | `src/schemez/markdown.py` |
| Tool client/server code generation | `src/schemez/code_generation/`, `src/schemez/tool_executor/` |
| OpenAPI handling | `src/schemez/openapi/` |
| SQLite model storage | `src/schemez/storage/model_store.py` |
| Tests | `tests/` (function schema tests in `tests/functionschema/`, fixtures in `tests/resources/`) |
| Example | `examples/http_tool_example.py` |

## Commands

```bash
uv sync --all-extras                 # setup (default groups: dev, lint, docs)
uv run pytest                        # tests (asyncio_mode = "auto")
uv run ruff check . && uv run ruff format --check .
uv run ty check                      # type check (src/ only, see [tool.ty.src])
uv run deptry src                    # dependency declaration check
uv run mkdocs-mknodes build          # docs (CI: .github/workflows/documentation.yml)
uv build
uv run duty lint | lint_check | test | build | serve | update   # duties.py shortcuts
```

CI (`.github/workflows/build.yml`) runs ruff check, ruff format --check, `ty check` and pytest with
coverage on Python 3.13/3.14 x Linux/macOS/Windows (3.14 is `continue-on-error`).

## Conventions

- Python >= 3.13, `from __future__ import annotations` everywhere, PEP 695 generics.
- Ruff with a large rule set (see `[tool.ruff.lint]`), line length 100, preview formatting; Markdown code
  blocks (README) are formatted too.
- ty must pass with zero warnings. Use targeted `# ty: ignore[rule]` only where dynamic model/type
  creation defeats the checker; no blanket ignores. mypy is no longer used.
- Google-style docstrings. Tests are plain pytest functions, async tests need no marker.
- Optional dependencies are imported lazily inside functions (`yamling`, `agentpool`, `fastapi`,
  `uvicorn`, `aiosqlite`, `datamodel_code_generator`, `jinja2`, `pydantic_ai`).

## Specs

| Spec | URL | Targets | Code | Update check |
|---|---|---|---|---|
| JSON Schema | https://json-schema.org/draft/2020-12 | draft 2020-12 (`$schema` written by `schema_to_type/*_impl.py`) | `src/schemez/jsonschema_utils.py`, `src/schemez/schema_to_type/`, `src/schemez/schemadef/` | https://json-schema.org/specification |
| OpenAI function calling (tool schema) | https://platform.openai.com/docs/guides/function-calling | unpinned (tracks latest) | `src/schemez/functionschema/` (`model_dump_openai`, `OpenAIFunctionTool`) | same URL |
| OpenAPI | https://spec.openapis.org/oas/v3.1.0.html | 3.0 and 3.1 documents (3.0 `nullable` handled in `jsonschema_utils.py`) | `src/schemez/openapi/` | https://spec.openapis.org/oas/latest.html |

## Pitfalls

- Copier-managed project (`.copier-answers.yml`, template `copier-phil65`): don't hand-edit template-owned
  files (`.github/workflows/*`, `.pre-commit-config.yaml`, `duties.py`, most of `pyproject.toml`).
  Add dependencies below the `# Only add below (Copier)` markers.
- Extras: `ai` (agentpool, anyenv, httpx2, pydantic-ai; needed for `Schema.from_llm`, `create_schema(mode="openai")`,
  `schemez.openapi`), `codegen` (datamodel-code-generator; `json_schema_to_pydantic_class`, `*_to_code`),
  `yaml` (yamling; `Schema.from_yaml`/`model_dump_yaml`), `markdown` (jinja2), `storage` (aiosqlite),
  `tool_execution` (fastapi, uvicorn). `ai` pulls in agentpool, which itself depends on schemez.
- Pre-commit hooks run ruff, tombi, ty and the full pytest suite on every commit.
- `uv.lock` is not tracked.
- Docs are the mkdocs-mknodes default site: `README.md` is the index page, API pages are generated from
  docstrings; `docs/` is intentionally empty.
