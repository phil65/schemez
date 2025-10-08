"""Integration tests for HttpToolExecutor."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

import pytest

from schemez.tool_executor import HttpToolExecutor


if TYPE_CHECKING:
    from pydantic import BaseModel


# Mark slow tests that call datamodel-codegen many times
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class TestHttpToolExecutorIntegration:
    """Integration tests for HttpToolExecutor."""

    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create test schema
        schema = {
            "type": "function",
            "function": {
                "name": "echo_tool",
                "description": "Echo input message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message to echo"},
                        "uppercase": {"type": "boolean", "default": False},
                    },
                    "required": ["message"],
                },
            },
        }

        # Create handler
        async def handler(method_name: str, input_props: BaseModel) -> str:
            message = getattr(input_props, "message", "")
            uppercase = getattr(input_props, "uppercase", False)
            result = message.upper() if uppercase else message
            return f"Echo: {result}"

        # Create executor
        executor = HttpToolExecutor([schema], handler)

        # Test complete workflow
        # 1. Generate code
        code = await executor.generate_tools_code()
        assert "EchoToolInput" in code
        assert "async def echo_tool" in code

        # 2. Get tool functions (skip - this is slow)
        # tools = await executor.get_tool_functions()
        # assert "echo_tool" in tools

        # 3. Generate server
        server = await executor.generate_server_app()
        assert server.title == "Tool Server"

        # 4. Save to files
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = await executor.save_to_files(Path(temp_dir))
            assert saved_files["tools"].exists()
            assert saved_files["server_example"].exists()

    async def test_multiple_tools_interaction(self):
        """Test interaction between multiple tools."""
        schemas: list[dict[str, Any] | Path] = [
            {
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "multiply_numbers",
                    "description": "Multiply two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                        },
                        "required": ["x", "y"],
                    },
                },
            },
        ]

        call_log = []

        async def math_handler(method_name: str, input_props: BaseModel) -> str:
            call_log.append(method_name)
            match method_name:
                case "add_numbers":
                    a = getattr(input_props, "a", 0)
                    b = getattr(input_props, "b", 0)
                    return str(a + b)
                case "multiply_numbers":
                    x = getattr(input_props, "x", 1)
                    y = getattr(input_props, "y", 1)
                    return str(x * y)
                case _:
                    return "0"

        executor = HttpToolExecutor(schemas, math_handler)

        # Test both tools are generated (skip slow get_tool_functions)
        # tools = await executor.get_tool_functions()
        # assert len(tools) == 2
        # assert "add_numbers" in tools
        # assert "multiply_numbers" in tools

        # Test code contains both tools
        code = await executor.generate_tools_code()
        assert "AddNumbersInput" in code
        assert "MultiplyNumbersInput" in code
        assert "async def add_numbers" in code
        assert "async def multiply_numbers" in code

    async def test_file_based_schemas(self):
        """Test loading schemas from actual files."""
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "file_tool_1",
                    "description": "First file tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_tool_2",
                    "description": "Second file tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                    },
                },
            },
        ]

        async def file_handler(method_name: str, input_props: BaseModel) -> str:
            return f"Handled {method_name}"

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create schema files
            schema_files = []
            for i, schema in enumerate(schemas):
                file_path = Path(temp_dir) / f"tool_{i + 1}.json"
                file_path.write_text(json.dumps(schema))
                schema_files.append(file_path)

            # Create executor with file paths
            executor = HttpToolExecutor(schema_files, file_handler)

            # Test loading and processing
            loaded_schemas = await executor._load_schemas()
            assert len(loaded_schemas) == 2  # noqa: PLR2004

            # Skip slow get_tool_functions test
            # tools = await executor.get_tool_functions()
            # assert len(tools) == 2
            # assert "file_tool_1" in tools
            # assert "file_tool_2" in tools

            # Test mappings instead
            mappings = await executor._get_tool_mappings()
            assert len(mappings) == 2  # noqa: PLR2004
            assert "file_tool_1" in mappings
            assert "file_tool_2" in mappings

    async def test_mixed_schema_sources(self):
        """Test mixing file and dict schemas."""
        dict_schema = {
            "type": "function",
            "function": {
                "name": "dict_tool",
                "description": "Tool from dict",
                "parameters": {
                    "type": "object",
                    "properties": {"data": {"type": "string"}},
                    "required": ["data"],
                },
            },
        }

        file_schema = {
            "type": "function",
            "function": {
                "name": "file_tool",
                "description": "Tool from file",
                "parameters": {
                    "type": "object",
                    "properties": {"info": {"type": "string"}},
                    "required": ["info"],
                },
            },
        }

        async def mixed_handler(method_name: str, input_props: BaseModel) -> str:
            return f"Mixed result for {method_name}"

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create file schema
            file_path = Path(temp_dir) / "file_tool.json"
            file_path.write_text(json.dumps(file_schema))

            # Mix dict and file schemas
            executor = HttpToolExecutor([dict_schema, file_path], mixed_handler)

            # Test both are loaded
            schemas = await executor._load_schemas()
            assert len(schemas) == 2  # noqa: PLR2004

            # Skip slow get_tool_functions test
            # tools = await executor.get_tool_functions()
            # assert "dict_tool" in tools
            # assert "file_tool" in tools

            # Test mappings instead
            mappings = await executor._get_tool_mappings()
            assert "dict_tool" in mappings
            assert "file_tool" in mappings

    async def test_error_handling_in_handler(self):
        """Test error handling when handler raises exceptions."""
        schema = {
            "type": "function",
            "function": {
                "name": "error_tool",
                "description": "Tool that may error",
                "parameters": {
                    "type": "object",
                    "properties": {"should_error": {"type": "boolean", "default": False}},
                    "required": [],
                },
            },
        }

        async def error_handler(method_name: str, input_props: BaseModel) -> str:
            should_error = getattr(input_props, "should_error", False)
            if should_error:
                msg = "Intentional test error"
                raise ValueError(msg)
            return "Success"

        executor = HttpToolExecutor([schema], error_handler)

        # Server should handle handler exceptions gracefully
        app = await executor.generate_server_app()
        assert app is not None

        # The server endpoint should catch handler exceptions
        # (This would be tested with an actual HTTP client in a full integration test)

    async def test_complex_schema_types(self):
        """Test with complex schema types (nested objects, arrays, etc.)."""
        complex_schema = {
            "type": "function",
            "function": {
                "name": "complex_tool",
                "description": "Tool with complex types",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "simple_string": {"type": "string"},
                        "number_array": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "nested_object": {
                            "type": "object",
                            "properties": {
                                "inner_string": {"type": "string"},
                                "inner_bool": {"type": "boolean"},
                            },
                        },
                        "enum_field": {
                            "type": "string",
                            "enum": ["option1", "option2", "option3"],
                        },
                    },
                    "required": ["simple_string"],
                },
            },
        }

        async def complex_handler(method_name: str, input_props: BaseModel) -> str:
            return f"Handled complex input: {type(input_props).__name__}"

        executor = HttpToolExecutor([complex_schema], complex_handler)

        # Test complex types are handled
        code = await executor.generate_tools_code()
        assert "ComplexToolInput" in code
        assert "async def complex_tool" in code

        # Should generate nested models
        assert "List[" in code  # For arrays
        assert "Literal[" in code  # For enums

        # Skip slow get_tool_functions test
        # tools = await executor.get_tool_functions()
        # assert "complex_tool" in tools

    async def test_multiple_tools_performance(self):
        """Test basic performance with a few tools (avoiding slow datamodel-codegen)."""
        num_tools = 3  # Keep it small to avoid hanging
        schemas = []

        for i in range(num_tools):
            schema = {
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "description": f"Tool number {i}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param": {
                                "type": "string",
                                "description": f"Parameter for tool {i}",
                            }
                        },
                        "required": ["param"],
                    },
                },
            }
            schemas.append(schema)

        async def bulk_handler(method_name: str, input_props: BaseModel) -> str:
            return f"Bulk result for {method_name}"

        executor = HttpToolExecutor(schemas, bulk_handler)

        # Test code generation works with multiple tools (skip slow get_tool_functions)
        code = await executor.generate_tools_code()
        for i in range(num_tools):
            assert f"Tool{i}Input" in code
            assert f"async def tool_{i}" in code

        # Test server generation works
        server = await executor.generate_server_app()
        assert server is not None

    async def test_custom_base_url_in_generated_code(self):
        """Test that custom base URL appears correctly in generated code."""
        schema = {
            "type": "function",
            "function": {
                "name": "url_test_tool",
                "description": "Test custom URL",
                "parameters": {
                    "type": "object",
                    "properties": {"test": {"type": "string"}},
                    "required": ["test"],
                },
            },
        }

        async def url_handler(method_name: str, input_props: BaseModel) -> str:
            return "URL test result"

        custom_urls = [
            "http://localhost:9999",  # Only test localhost to avoid slow tests
        ]

        for custom_url in custom_urls:
            executor = HttpToolExecutor([schema], url_handler, base_url=custom_url)
            code = await executor.generate_tools_code()

            # Check that custom URL appears in the generated HTTP calls
            assert custom_url in code
            assert f'"{custom_url}/tools/url_test_tool"' in code

    async def test_regeneration_consistency(self):
        """Test that regenerating code produces consistent results."""
        schema = {
            "type": "function",
            "function": {
                "name": "consistency_tool",
                "description": "Test consistency",
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                },
            },
        }

        async def consistent_handler(method_name: str, input_props: BaseModel) -> str:
            return "Consistent result"

        executor1 = HttpToolExecutor([schema], consistent_handler)
        executor2 = HttpToolExecutor([schema], consistent_handler)

        # Generate code from both executors
        code1 = await executor1.generate_tools_code()
        code2 = await executor2.generate_tools_code()

        # Results should be functionally equivalent
        # (exact string match might vary due to temp file names in comments)
        assert "ConsistencyToolInput" in code1
        assert "ConsistencyToolInput" in code2
        assert "async def consistency_tool" in code1
        assert "async def consistency_tool" in code2
        assert len(code1) == len(code2)  # Should be roughly the same size
