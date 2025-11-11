"""Schemez: main package.

Pydantic shim for config stuff.
"""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("schemez")
__title__ = "Schemez"

__author__ = "Philipp Temminghoff"
__author_email__ = "philipptemminghoff@googlemail.com"
__copyright__ = "Copyright (c) 2025 Philipp Temminghoff"
__license__ = "MIT"
__url__ = "https://github.com/phil65/schemez"

from schemez.schema import Schema
from schemez.code import PythonCode, JSONCode, TOMLCode, YAMLCode
from schemez.schemadef.schemadef import (
    SchemaDef,
    SchemaField,
    ImportedSchemaDef,
    InlineSchemaDef,
)
from schemez.pydantic_types import ModelIdentifier, ModelTemperature, MimeType

from schemez.functionschema.executable import create_executable, ExecutableFunction
from schemez.functionschema import FunctionType, create_schema, FunctionSchema
from schemez.functionschema.schema_generators import (
    create_schemas_from_callables,
    create_schemas_from_module,
    create_schemas_from_class,
    create_constructor_schema,
)
from schemez.functionschema.typedefs import (
    OpenAIFunctionDefinition,
    OpenAIFunctionTool,
    Property,
)
from schemez.code_generation import ToolCodeGenerator, ToolsetCodeGenerator

__version__ = version("schemez")


__all__ = [
    "ExecutableFunction",
    "FunctionSchema",
    "FunctionType",
    "ImportedSchemaDef",
    "InlineSchemaDef",
    "JSONCode",
    "MimeType",
    "ModelIdentifier",
    "ModelTemperature",
    "OpenAIFunctionDefinition",
    "OpenAIFunctionTool",
    "Property",
    "PythonCode",
    "Schema",
    "SchemaDef",
    "SchemaField",
    "TOMLCode",
    "ToolCodeGenerator",
    "ToolsetCodeGenerator",
    "YAMLCode",
    "__version__",
    "create_constructor_schema",
    "create_executable",
    "create_schema",
    "create_schemas_from_callables",
    "create_schemas_from_class",
    "create_schemas_from_module",
]
