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

__version__ = version("schemez")

__all__ = [
    "ImportedSchemaDef",
    "InlineSchemaDef",
    "JSONCode",
    "MimeType",
    "ModelIdentifier",
    "ModelTemperature",
    "PythonCode",
    "Schema",
    "SchemaDef",
    "SchemaField",
    "TOMLCode",
    "YAMLCode",
]
