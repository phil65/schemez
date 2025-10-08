"""Tests for OpenAI function schema creation."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator  # noqa: TC003
import dataclasses
from datetime import date, datetime, time, timedelta, timezone  # noqa: TC003
import decimal  # noqa: TC003
import enum
import ipaddress  # noqa: TC003
from pathlib import Path  # noqa: TC003
import re  # noqa: TC003
import typing as t
from typing import Annotated, Any, Literal
from uuid import UUID  # noqa: TC003

import pytest

from py2openai.functionschema import (
    FunctionType,
    create_schema,
)


JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None
JsonDict = dict[str, JsonValue]


class Color(enum.Enum):
    """Test enum."""

    RED = "red"
    BLUE = "blue"


@dataclasses.dataclass
class User:
    """Test dataclass."""

    name: str
    age: int


def test_basic_types() -> None:
    """Test basic type annotations supported by OpenAI."""

    def func(
        s: str,
        i: int,
        f: float,
        b: bool,
    ) -> str:
        """Test function with basic types.

        Args:
            s: A string
            i: An integer
            f: A float
            b: A boolean
        """
        return s

    schema = create_schema(func)
    assert schema.parameters["properties"] == {
        "s": {"type": "string", "description": "A string"},
        "i": {"type": "integer", "description": "An integer"},
        "f": {"type": "number", "description": "A float"},
        "b": {"type": "boolean", "description": "A boolean"},
    }
    assert schema.returns == {"type": "string"}


def test_container_types() -> None:
    """Test container type annotations supported by OpenAI."""

    def func(
        ls: list[str],
        dct: dict[str, Any],
    ) -> dict[str, list[int]]:
        """Test function with container types.

        Args:
            ls: A list of strings
            dct: A dictionary
        """
        return {"nums": [1, 2, 3]}

    schema = create_schema(func)
    props = schema.parameters["properties"]

    assert props["ls"] == {
        "type": "array",
        "items": {"type": "string"},
        "description": "A list of strings",
    }
    assert props["dct"] == {
        "type": "object",
        "additionalProperties": True,
        "description": "A dictionary",
    }
    assert schema.returns == {"type": "object"}


def test_optional_union_types() -> None:
    """Test Optional and Union type annotations."""

    def func(
        o: int | None,
        u: str | int,
        lit: Literal["a", "b", "c"],
        ou: str | int | None,
    ) -> None:
        """Test function with Optional/Union types.

        Args:
            o: An optional
            u: A union
            lit: A literal
            ou: An optional union
        """

    schema = create_schema(func)
    props = schema.parameters["properties"]

    assert props["o"] == {"type": "integer", "description": "An optional"}
    assert props["u"] == {
        "type": "string",  # first type in union
        "description": "A union",
    }
    assert props["lit"] == {
        "type": "string",
        "enum": ["a", "b", "c"],
        "description": "A literal",
    }
    assert props["ou"] == {
        "type": "string",  # first type in union after removing None
        "description": "An optional union",
    }


def test_custom_types() -> None:
    """Test custom type annotations."""

    def func(
        e: Color,
        dc: User,
    ) -> None:
        """Test function with custom types.

        Args:
            e: An enum
            dc: A dataclass
        """

    schema = create_schema(func)
    props = schema.parameters["properties"]

    assert props["e"] == {
        "type": "string",
        "enum": ["red", "blue"],
        "description": "An enum",
    }
    assert props["dc"] == {"type": "object", "description": "A dataclass"}


def test_typing_variants() -> None:
    """Test different typing module variants."""

    def func(
        # List variants
        builtin_list: list[int],
        typing_list: t.List[int],  # noqa: UP006
        # Dict variants
        builtin_dict: dict[str, int],
        typing_dict: t.Dict[str, int],  # noqa: UP006
        # Optional variants
        optional_new: int | None,
        optional_old: int | None,
        # Union variants
        union_new: int | str,
        union_old: int | str,
        # Sequence variants
        sequence: t.Sequence[int],
    ) -> None:
        """Test function with typing variants.

        Args:
            builtin_list: Python 3.9+ list
            typing_list: Traditional typing.List
            builtin_dict: Python 3.9+ dict
            typing_dict: Traditional typing.Dict
            optional_new: Python 3.10+ Optional
            optional_old: Traditional Optional
            union_new: Python 3.10+ Union
            union_old: Traditional Union
            sequence: Generic sequence
        """

    schema = create_schema(func)
    props = schema.parameters["properties"]

    # List variants should produce array
    assert props["builtin_list"] == {
        "type": "array",
        "items": {"type": "integer"},
        "description": "Python 3.9+ list",
    }
    assert props["typing_list"] == {
        "type": "array",
        "items": {"type": "integer"},
        "description": "Traditional typing.List",
    }
    assert props["sequence"] == {
        "type": "array",
        "items": {"type": "integer"},
        "description": "Generic sequence",
    }

    # Dict variants should produce object
    assert props["builtin_dict"] == {
        "type": "object",
        "additionalProperties": True,
        "description": "Python 3.9+ dict",
    }
    assert props["typing_dict"] == {
        "type": "object",
        "additionalProperties": True,
        "description": "Traditional typing.Dict",
    }

    # Optional variants should handle None
    assert props["optional_new"] == {
        "type": "integer",
        "description": "Python 3.10+ Optional",
    }
    assert props["optional_old"] == {
        "type": "integer",
        "description": "Traditional Optional",
    }

    # Union variants should take first type
    assert props["union_new"] == {
        "type": "integer",
        "description": "Python 3.10+ Union",
    }
    assert props["union_old"] == {
        "type": "integer",
        "description": "Traditional Union",
    }


def test_default_values() -> None:
    """Test function with default values."""

    def func(
        req: int,
        opt: str = "default",
        flag: bool = False,
        lst: list[int] | None = None,
    ) -> None:
        """Test function with defaults.

        Args:
            req: Required parameter
            opt: Optional parameter
            flag: Boolean flag
            lst: Optional list
        """

    schema = create_schema(func)
    assert set(schema.required) == {"req"}
    assert "opt" not in schema.required
    assert "flag" not in schema.required
    assert "lst" not in schema.required


def test_invalid_input() -> None:
    """Test invalid inputs."""
    with pytest.raises(TypeError, match="Expected callable"):
        create_schema("not a function")  # type: ignore

    with pytest.raises(TypeError, match="Expected callable"):
        create_schema(None)  # type: ignore


def test_sync_generator() -> None:
    """Test sync generator functions are properly schematized."""

    def gen(n: int) -> Generator[int]:
        """Generate n numbers.

        Args:
            n: Number of items
        """
        yield from range(n)

    schema = create_schema(gen)
    assert schema.function_type == FunctionType.SYNC_GENERATOR
    assert schema.returns == {
        "type": "array",
        "items": {"type": "integer"},
    }


async def test_async_generator() -> None:
    """Test async generator functions are properly schematized."""

    async def agen(n: int) -> AsyncGenerator[str]:
        """Generate n strings.

        Args:
            n: Number of items
        """
        for i in range(n):
            yield str(i)

    schema = create_schema(agen)
    assert schema.function_type == FunctionType.ASYNC_GENERATOR
    assert schema.returns == {
        "type": "array",
        "items": {"type": "string"},
    }


def test_docstring_parsing() -> None:
    """Test docstring parsing for descriptions."""

    def func(x: int, y: str) -> None:
        """Main description.

        Detailed description that should be ignored.

        Args:
            x: X parameter
            y: Y parameter
        """

    schema = create_schema(func)
    assert schema.description == "Main description."
    assert schema.parameters["properties"]["x"]["description"] == "X parameter"
    assert schema.parameters["properties"]["y"]["description"] == "Y parameter"


def test_openai_schema_format() -> None:
    """Test that schema follows OpenAI function format."""

    def func(x: int, y: str = "default") -> str:
        """Calculate something.

        Args:
            x: The number
            y: Optional string
        """
        return str(x)

    schema = create_schema(func)

    # Check all required OpenAI schema fields are present
    assert isinstance(schema.name, str)
    assert isinstance(schema.description, str | type(None))
    assert isinstance(schema.parameters, dict)
    assert "type" in schema.parameters
    assert "properties" in schema.parameters
    assert isinstance(schema.required, list)
    assert all(isinstance(r, str) for r in schema.required)


def test_openai_schema_serialization() -> None:
    """Test schema can be serialized as OpenAI expects."""

    def func(x: int) -> str:
        """Test function."""
        return str(x)

    schema = create_schema(func)
    data = schema.model_dump(exclude_none=True)

    # Check OpenAI expected structure
    assert set(data.keys()) >= {"name", "parameters", "required"}
    assert data["parameters"]["type"] == "object"
    assert isinstance(data["parameters"]["properties"], dict)


def test_nested_complex_types() -> None:
    """Test handling of complex nested type structures."""

    def func(
        nested_dict: dict[str, list[set[int]]],
        complex_union: list[dict[str, int | str]],
        nested_list: list[dict[str, set[str]]] | None = None,
    ) -> dict[str, list[Any]]:
        """Test complex types.

        Args:
            nested_dict: Dictionary with nested collections
            nested_list: List of dictionaries with sets
            complex_union: List of dicts with union values
        """
        return {"results": []}

    schema = create_schema(func)
    props = schema.parameters["properties"]

    # Check nested dictionary structure
    assert "nested_dict" in props
    assert props["nested_dict"]["type"] == "object"

    # Check optional nested list
    assert "nested_list" in props
    assert "type" in props["nested_list"]

    # Check list with union type values
    assert "complex_union" in props
    assert props["complex_union"]["type"] == "array"


def test_annotated_types() -> None:
    """Test handling of annotated types - should use base types."""

    def func(
        x: Annotated[int, "The number"],
        y: Annotated[str, "The string"] = "default",
    ) -> Annotated[dict[str, Any], "Result"]:
        """Test annotated parameters.

        Args:
            x: The number
            y: The string
        """
        return {"x": x, "y": y}

    schema = create_schema(func)
    props = schema.parameters["properties"]

    # Should only use base types, ignoring Annotated metadata
    assert props["x"]["type"] == "integer"
    assert props["y"]["type"] == "string"
    assert props["y"].get("default") == "default"

    # Descriptions should come from docstrings, not Annotated
    assert props["x"]["description"] == "The number"
    assert props["y"]["description"] == "The string"


def test_empty_and_edge_cases() -> None:
    """Test edge cases in function definitions."""

    def no_params() -> None:
        """Function with no parameters."""

    def any_param(x: Any) -> Any:
        """Function with Any type."""
        return x

    def variadic(*args: int, **kwargs: str) -> None:
        """Function with *args and **kwargs."""

    # Test empty parameter list
    empty_schema = create_schema(no_params)
    assert not empty_schema.parameters["properties"]

    # Test Any type
    any_schema = create_schema(any_param)
    assert "type" in any_schema.parameters["properties"]["x"]

    # Test variadic args (should be ignored)
    var_schema = create_schema(variadic)
    assert not var_schema.parameters["properties"]


def test_literal_types() -> None:
    """Test handling of literal types."""

    def func(
        mode: Literal["read", "write"],
        flag: Literal[True, False] | None = None,
    ) -> None:
        """Test literal types.

        Args:
            mode: Operation mode
            flag: Boolean flag
        """

    schema = create_schema(func)
    props = schema.parameters["properties"]

    assert props["mode"]["type"] == "string"
    assert props["mode"]["enum"] == ["read", "write"]
    assert props["flag"].get("default") is None


async def test_async_generators() -> None:
    """Test async generator function schemas."""

    async def stream(
        text: str,
        chunk_size: int = 100,
    ) -> AsyncGenerator[str]:
        """Stream text in chunks.

        Args:
            text: Input text
            chunk_size: Size of chunks
        """
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]

    schema = create_schema(stream)
    props = schema.parameters["properties"]

    assert props["text"]["type"] == "string"
    assert props["chunk_size"]["type"] == "integer"
    assert props["chunk_size"].get("default") == 100  # noqa: PLR2004
    assert schema.returns["type"] == "array"
    assert schema.returns["items"]["type"] == "string"


def test_union_type_variations() -> None:
    """Test various union type combinations."""

    def func(
        basic_union: int | str,
        multi_union: int | str | float,  # noqa: PYI041
        optional_union: (int | str) | None,
        nested_union: list[int | str],
    ) -> dict[str, int | str]:
        """Test union variations.

        Args:
            basic_union: Simple union
            multi_union: Union with multiple types
            optional_union: Optional union
            nested_union: Union in container
        """
        return {"result": 1}

    schema = create_schema(func)
    props = schema.parameters["properties"]

    # Check basic union (should use first type)
    assert props["basic_union"]["type"] == "integer"

    # Check multi-type union (should use first type)
    assert props["multi_union"]["type"] == "integer"

    # Check optional union
    assert props["optional_union"]["type"] == "integer"

    # Check nested union in container
    assert props["nested_union"]["type"] == "array"
    assert props["nested_union"]["items"]["type"] == "integer"


def test_recursive_types() -> None:
    """Test handling of recursive type definitions."""

    def func(
        tree: dict[str, dict[str, Any] | str],
        nested: dict[str, list[dict[str, Any]]],
    ) -> dict[str, dict[str, Any] | None]:
        """Handle recursive structures.

        Args:
            tree: A tree-like structure
            nested: Nested dictionaries in lists
        """
        return {"key": None}

    schema = create_schema(func)
    props = schema.parameters["properties"]

    assert props["tree"]["type"] == "object"
    assert props["nested"]["type"] == "object"
    assert schema.returns["type"] == "object"


def test_type_aliases() -> None:
    """Test handling of type aliases."""

    def func(
        data: JsonDict,
        values: list[JsonValue],
    ) -> JsonDict:
        """Handle type aliases.

        Args:
            data: JSON-like dictionary
            values: List of JSON values
        """
        return data

    schema = create_schema(func)
    props = schema.parameters["properties"]

    assert props["data"]["type"] == "object"
    assert props["values"]["type"] == "array"
    assert schema.returns["type"] == "object"


def test_datetime_types() -> None:
    """Test datetime type handling."""

    def func(
        dt: datetime,
        d: date,
        t: time,
        optional_dt: datetime | None = None,
    ) -> dict[str, datetime]:
        """Test datetime types.

        Args:
            dt: A datetime
            d: A date
            t: A time
            optional_dt: Optional datetime
        """
        return {"result": dt}

    schema = create_schema(func)
    props = schema.parameters["properties"]

    assert props["dt"] == {
        "type": "string",
        "format": "date-time",
        "description": "A datetime (ISO 8601 format)",
    }
    assert props["d"] == {
        "type": "string",
        "format": "date",
        "description": "A date (ISO 8601 format)",
    }
    assert props["t"] == {
        "type": "string",
        "format": "time",
        "description": "A time (ISO 8601 format)",
    }
    assert props["optional_dt"]["type"] == "string"
    assert props["optional_dt"]["format"] == "date-time"

    assert schema.returns == {
        "type": "object",
    }


def test_extended_basic_types() -> None:
    """Test extended basic type annotations."""

    def func(
        dec: decimal.Decimal,
        comp: complex,
        b: bytes,
        p: Path,
        td: timedelta,
        tz: timezone,
        uid: UUID,
        pat: re.Pattern[str],
        r: range,
    ) -> None:
        """Test extended basic types.

        Args:
            dec: A decimal number
            comp: A complex number
            b: Some bytes
            p: A path
            td: A time duration
            tz: A timezone
            uid: A UUID
            pat: A regex pattern
            r: A range
        """

    schema = create_schema(func)
    props = schema.parameters["properties"]

    assert props["dec"] == {
        "type": "number",
        "description": "A decimal number",
    }
    assert props["comp"] == {
        "type": "object",
        "properties": {
            "real": {"type": "number"},
            "imag": {"type": "number"},
        },
        "description": "A complex number",
    }
    assert props["b"] == {
        "type": "string",
        "description": "Some bytes (base64 encoded)",
    }
    assert props["p"] == {
        "type": "string",
        "description": "A path",
    }
    assert props["td"] == {
        "type": "string",
        "description": "A time duration (ISO 8601 duration)",
    }
    assert props["tz"] == {
        "type": "string",
        "description": "A timezone (IANA timezone name)",
    }
    assert props["uid"] == {
        "type": "string",
        "description": "A UUID",
    }
    assert props["pat"] == {
        "type": "string",
        "description": "A regex pattern",
    }
    assert props["r"] == {
        "type": "object",
        "description": "A range",
    }


def test_extended_string_formats() -> None:
    """Test string format types."""

    def func(
        ip4: ipaddress.IPv4Address,
        ip6: ipaddress.IPv6Address,
        uid: UUID,
        data: bytes,
    ) -> None:
        """Test string formats.

        Args:
            ip4: IPv4 address
            ip6: IPv6 address
            uid: Unique identifier
            data: Binary data
        """

    schema = create_schema(func)
    props = schema.parameters["properties"]

    assert props["ip4"] == {
        "type": "string",
        "description": "IPv4 address",
    }
    assert props["ip6"] == {
        "type": "string",
        "description": "IPv6 address",
    }
    assert props["uid"] == {
        "type": "string",
        "description": "Unique identifier",
    }
    assert props["data"] == {
        "type": "string",
        "description": "Binary data (base64 encoded)",
    }


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
