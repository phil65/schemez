from __future__ import annotations

import ast
from typing import Literal

from pydantic import field_validator

from schemez.schema import Schema


CodeLanguage = Literal["python", "yaml", "json", "toml"]


class BaseCode(Schema):
    """Base class for syntax-validated code."""

    code: str
    """The source code."""

    @field_validator("code")
    @classmethod
    def validate_syntax(cls, code: str) -> str:
        """Override in subclasses."""
        return code


class YAMLCode(BaseCode):
    """YAML with syntax validation."""

    @field_validator("code")
    @classmethod
    def validate_syntax(cls, code: str) -> str:
        import yamling

        try:
            yamling.load(code, mode="yaml")
        except yamling.ParsingError as e:
            msg = f"Invalid YAML syntax: {e}"
            raise ValueError(msg) from e
        else:
            return code


class JSONCode(BaseCode):
    """JSON with syntax validation."""

    @field_validator("code")
    @classmethod
    def validate_syntax(cls, code: str) -> str:
        import yamling

        try:
            yamling.load(code, mode="json")
        except yamling.ParsingError as e:
            msg = f"Invalid JSON syntax: {e}"
            raise ValueError(msg) from e
        else:
            return code


class TOMLCode(BaseCode):
    """TOML with syntax validation."""

    @field_validator("code")
    @classmethod
    def validate_syntax(cls, code: str) -> str:
        import yamling

        try:
            yamling.load(code, mode="toml")
        except yamling.ParsingError as e:
            msg = f"Invalid TOML syntax: {e}"
            raise ValueError(msg) from e
        else:
            return code


class PythonCode(BaseCode):
    """Python with syntax validation."""

    @field_validator("code")
    @classmethod
    def validate_syntax(cls, code: str) -> str:
        try:
            ast.parse(code)
        except SyntaxError as e:
            msg = f"Invalid Python syntax: {e}"
            raise ValueError(msg) from e
        else:
            return code
