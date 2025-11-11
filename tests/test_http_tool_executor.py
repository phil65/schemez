"""Tests for HttpToolExecutor."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

import pytest

from schemez.tool_executor import HttpToolExecutor


if TYPE_CHECKING:
    from pydantic import BaseModel


class MockToolHandler:
    """Mock tool handler for testing."""

    def __init__(self):
        self.calls: list[tuple[str, BaseModel]] = []

    async def __call__(self, method_name: str, input_props: BaseModel) -> str:
        """Mock handler that records calls."""
        self.calls.append((method_name, input_props))

        match method_name:
            case "get_weather":
                location = getattr(input_props, "location", "Unknown")
                units = getattr(input_props, "units", "celsius")
                temp = "72°F" if units == "fahrenheit" else "22°C"
                return f"Weather in {location}: sunny, {temp}"

            case "create_calendar_event":
                title = getattr(input_props, "title", "Untitled")
                return f"Created event: {title}"

            case "simple_tool":
                message = getattr(input_props, "message", "")
                return f"Echo: {message}"

            case _:
                return f"Mock result for {method_name}: {input_props}"


@pytest.fixture
def weather_schema() -> dict[str, Any]:
    """Simple weather tool schema."""
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius",
                    },
                },
                "required": ["location"],
            },
        },
    }


@pytest.fixture
def calendar_schema() -> dict[str, Any]:
    """Calendar event tool schema."""
    return {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a calendar event",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title"},
                    "start_time": {"type": "string", "format": "date-time"},
                },
                "required": ["title"],
            },
        },
    }


@pytest.fixture
def mock_handler() -> MockToolHandler:
    """Mock tool handler."""
    return MockToolHandler()


class TestHttpToolExecutor:
    """Tests for HttpToolExecutor class."""


def test_init(weather_schema: dict[str, Any], mock_handler: MockToolHandler):
    """Test executor initialization."""
    executor = HttpToolExecutor(
        schemas=[weather_schema], handler=mock_handler, base_url="http://test:8000"
    )

    assert executor.schemas == [weather_schema]
    assert executor.handler == mock_handler
    assert executor.base_url == "http://test:8000"
    assert executor._tool_mappings is None
    assert executor._tools_code is None


async def test_load_schemas_from_dicts(
    weather_schema: dict[str, Any], mock_handler: MockToolHandler
):
    """Test loading schemas from dictionaries."""
    executor = HttpToolExecutor([weather_schema], mock_handler)

    schemas = await executor._load_schemas()

    assert len(schemas) == 1
    assert schemas[0] == weather_schema


async def test_load_schemas_from_files(
    weather_schema: dict[str, Any], mock_handler: MockToolHandler
):
    """Test loading schemas from files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        schema_file = Path(temp_dir) / "weather.json"
        schema_file.write_text(json.dumps(weather_schema))

        executor = HttpToolExecutor([schema_file], mock_handler)
        schemas = await executor._load_schemas()

        assert len(schemas) == 1
        assert schemas[0] == weather_schema


async def test_get_tool_mappings(
    weather_schema: dict[str, Any],
    calendar_schema: dict[str, Any],
    mock_handler: MockToolHandler,
):
    """Test tool name to input class mappings."""
    executor = HttpToolExecutor([weather_schema, calendar_schema], mock_handler)

    mappings = await executor._get_tool_mappings()

    assert "get_weather" in mappings
    assert "create_calendar_event" in mappings
    assert mappings["get_weather"] == "GetWeatherInput"
    assert mappings["create_calendar_event"] == "CreateCalendarEventInput"


async def test_generate_tools_code(
    weather_schema: dict[str, Any], mock_handler: MockToolHandler
):
    """Test generating HTTP wrapper tools code."""
    executor = HttpToolExecutor([weather_schema], mock_handler)

    tools_code = await executor.generate_tools_code()

    assert isinstance(tools_code, str)
    assert len(tools_code) > 0
    assert "GetWeatherInput" in tools_code
    assert "async def get_weather" in tools_code
    assert "httpx.AsyncClient" in tools_code
    assert "__all__" in tools_code


async def test_generate_tools_code_caching(
    weather_schema: dict[str, Any], mock_handler: MockToolHandler
):
    """Test that tools code generation is cached."""
    executor = HttpToolExecutor([weather_schema], mock_handler)

    code1 = await executor.generate_tools_code()
    code2 = await executor.generate_tools_code()

    # Should be the same object (cached)
    assert code1 is code2


async def test_generate_server_app(
    weather_schema: dict[str, Any], mock_handler: MockToolHandler
):
    """Test generating FastAPI server."""
    pytest.importorskip("fastapi")  # Skip if FastAPI not available

    executor = HttpToolExecutor([weather_schema], mock_handler)

    app = await executor.generate_server_app()

    assert app.title == "Tool Server"
    assert app.version == "1.0.0"


async def test_get_tool_functions(
    weather_schema: dict[str, Any], mock_handler: MockToolHandler
):
    """Test getting ready-to-use tool functions."""
    executor = HttpToolExecutor([weather_schema], mock_handler)

    tools = await executor.get_tool_functions()

    assert "get_weather" in tools
    assert callable(tools["get_weather"])


async def test_save_to_files(
    weather_schema: dict[str, Any], mock_handler: MockToolHandler
):
    """Test saving generated code to files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "output"

        executor = HttpToolExecutor([weather_schema], mock_handler)
        saved_files = await executor.save_to_files(output_dir)

        assert "tools" in saved_files
        assert "server_example" in saved_files

        tools_file = saved_files["tools"]
        server_file = saved_files["server_example"]

        assert tools_file.exists()
        assert server_file.exists()

        tools_content = tools_file.read_text()
        assert "GetWeatherInput" in tools_content
        assert "async def get_weather" in tools_content


async def test_multiple_schemas(
    weather_schema: dict[str, Any], calendar_schema, mock_handler: MockToolHandler
):
    """Test with multiple tool schemas."""
    executor = HttpToolExecutor([weather_schema, calendar_schema], mock_handler)

    # Test tool mappings
    mappings = await executor._get_tool_mappings()
    assert len(mappings) == 2  # noqa: PLR2004

    # Test code generation
    code = await executor.generate_tools_code()
    assert "GetWeatherInput" in code
    assert "CreateCalendarEventInput" in code
    assert "async def get_weather" in code
    assert "async def create_calendar_event" in code

    # Test tool functions
    tools = await executor.get_tool_functions()
    assert len(tools) == 2  # noqa: PLR2004
    assert "get_weather" in tools
    assert "create_calendar_event" in tools


async def test_simple_manual_schema(mock_handler):
    """Test with a simple manually created schema."""
    simple_schema = {
        "type": "function",
        "function": {
            "name": "simple_tool",
            "description": "A simple test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Input message"}
                },
                "required": ["message"],
            },
        },
    }

    executor = HttpToolExecutor([simple_schema], mock_handler)

    # Test generation
    code = await executor.generate_tools_code()
    assert "SimpleToolInput" in code
    assert "async def simple_tool" in code

    # Test tool functions
    tools = await executor.get_tool_functions()
    assert "simple_tool" in tools


async def test_invalid_schema_type(mock_handler: MockToolHandler):
    """Test with invalid schema type."""
    executor = HttpToolExecutor(
        [123],  # type: ignore
        mock_handler,
    )
    with pytest.raises(TypeError, match="Invalid schema type"):
        await executor._load_schemas()


async def test_nonexistent_file(mock_handler):
    """Test with nonexistent schema file."""
    nonexistent_file = Path("nonexistent.json")
    executor = HttpToolExecutor([nonexistent_file], mock_handler)

    with pytest.raises(FileNotFoundError):
        await executor._load_schemas()


async def test_base_url_customization(
    weather_schema: dict[str, Any], mock_handler: MockToolHandler
):
    """Test custom base URL in generated code."""
    custom_url = "http://custom:9000"
    executor = HttpToolExecutor([weather_schema], mock_handler, base_url=custom_url)

    code = await executor.generate_tools_code()

    assert custom_url in code


async def test_tool_mappings_caching(
    weather_schema: dict[str, Any], mock_handler: MockToolHandler
):
    """Test that tool mappings are cached."""
    executor = HttpToolExecutor([weather_schema], mock_handler)

    mappings1 = await executor._get_tool_mappings()
    mappings2 = await executor._get_tool_mappings()

    # Should be the same object (cached)
    assert mappings1 is mappings2
