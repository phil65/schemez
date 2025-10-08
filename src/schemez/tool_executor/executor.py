"""HTTP tool executor for managing tool generation and execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
import logging
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


try:
    from fastapi import FastAPI, HTTPException

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from schemez.functionschema import FunctionSchema
from schemez.helpers import model_to_python_code
from schemez.schema import Schema
from schemez.tool_executor.types import ToolHandler


logger = logging.getLogger(__name__)


class HttpToolExecutor:
    """Manages HTTP tool generation and execution."""

    def __init__(
        self,
        schemas: list[dict | Path],
        handler: ToolHandler,
        base_url: str = "http://localhost:8000",
    ):
        """Initialize the tool executor.

        Args:
            schemas: List of tool schema dictionaries or file paths
            handler: User-provided tool handler function
            base_url: Base URL for the tool server
        """
        self.schemas = schemas
        self.handler = handler
        self.base_url = base_url

        # Cached artifacts
        self._tool_mappings: dict[str, str] | None = None
        self._tools_code: str | None = None
        self._server_app: FastAPI | None = None
        self._tool_functions: dict[str, Callable] | None = None

    async def _load_schemas(self) -> list[dict]:
        """Load and normalize schemas from various sources."""
        loaded_schemas = []

        for schema in self.schemas:
            if isinstance(schema, dict):
                loaded_schemas.append(schema)
            elif isinstance(schema, (str, Path)):
                with Path(schema).open() as f:
                    loaded_schemas.append(json.load(f))
            else:
                msg = f"Invalid schema type: {type(schema)}"
                raise ValueError(msg)

        return loaded_schemas

    async def _get_tool_mappings(self) -> dict[str, str]:
        """Get tool name to input class mappings."""
        if self._tool_mappings is None:
            self._tool_mappings = {}
            schemas = await self._load_schemas()

            for schema_dict in schemas:
                function_schema = FunctionSchema.from_dict(schema_dict)
                input_class_name = f"{''.join(word.title() for word in function_schema.name.split('_'))}Input"
                self._tool_mappings[function_schema.name] = input_class_name

        return self._tool_mappings

    def _clean_generated_code(self, code: str) -> str:
        """Clean generated code by removing future imports and headers."""
        lines = code.split("\n")
        cleaned_lines = []
        skip_until_class = True

        for line in lines:
            # Skip lines until we find a class or other meaningful content
            if skip_until_class:
                if line.strip().startswith("class ") or (
                    line.strip()
                    and not line.startswith("#")
                    and not line.startswith("from __future__")
                ):
                    skip_until_class = False
                    cleaned_lines.append(line)
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    async def _generate_input_model(self, schema_dict: dict) -> tuple[str, str]:
        """Generate input model code from schema."""
        start_time = time.time()
        logger.debug(f"Generating input model for {schema_dict['name']}")

        class TempInputSchema(Schema):
            pass

        TempInputSchema.model_json_schema = lambda: schema_dict["parameters"]

        input_class_name = (
            f"{''.join(word.title() for word in schema_dict['name'].split('_'))}Input"
        )

        input_code = await model_to_python_code(
            TempInputSchema,
            class_name=input_class_name,
        )

        elapsed = time.time() - start_time
        logger.debug(f"Generated input model for {schema_dict['name']} in {elapsed:.2f}s")

        return input_code, input_class_name

    async def _generate_http_wrapper(
        self, schema_dict: dict, input_class_name: str
    ) -> str:
        """Generate HTTP wrapper function."""
        name = schema_dict["name"]
        description = schema_dict.get("description", "")

        wrapper_code = f'''
async def {name}(input: {input_class_name}) -> str:
    """{description}

    Args:
        input: Function parameters

    Returns:
        String response from the tool server
    """
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "{self.base_url}/tools/{name}",
            json=input.model_dump(),
            timeout=30.0
        )
        response.raise_for_status()
        return response.text
'''
        return wrapper_code

    async def generate_tools_code(self) -> str:
        """Generate HTTP wrapper tools as Python code."""
        if self._tools_code is not None:
            return self._tools_code

        start_time = time.time()
        logger.info("Starting tools code generation")

        schemas = await self._load_schemas()
        code_parts = []
        tool_mappings = await self._get_tool_mappings()

        # Module header
        header = '''"""Generated HTTP wrapper tools."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, List, Any
from datetime import datetime

'''
        code_parts.append(header)

        # Generate models and wrappers for each tool
        all_exports = []
        for schema_dict in schemas:
            function_schema = FunctionSchema.from_dict(schema_dict)

            schema_data = {
                "name": function_schema.name,
                "description": function_schema.description,
                "parameters": function_schema.parameters,
            }

            # Generate input model (strip future imports from generated code)
            input_code, input_class_name = await self._generate_input_model(schema_data)
            # Remove future imports and datamodel-codegen header from individual models
            cleaned_input_code = self._clean_generated_code(input_code)
            code_parts.append(cleaned_input_code)

            # Generate HTTP wrapper
            wrapper_code = await self._generate_http_wrapper(
                schema_data, input_class_name
            )
            code_parts.append(wrapper_code)

            all_exports.extend([input_class_name, function_schema.name])

        # Add exports
        code_parts.append(f"\n__all__ = {all_exports}\n")

        self._tools_code = "\n".join(code_parts)
        elapsed = time.time() - start_time
        logger.info(f"Tools code generation completed in {elapsed:.2f}s")
        return self._tools_code

    async def generate_server_app(self) -> FastAPI:
        """Create configured FastAPI server."""
        if self._server_app is not None:
            return self._server_app

        tool_mappings = await self._get_tool_mappings()
        schemas = await self._load_schemas()

        # Create app
        app = FastAPI(title="Tool Server", version="1.0.0")

        # Build input model mapping for runtime
        input_models = {}
        for schema_dict in schemas:
            function_schema = FunctionSchema.from_dict(schema_dict)

            # Create input model class dynamically
            class TempInputSchema(Schema):
                pass

            schema_data = {
                "name": function_schema.name,
                "parameters": function_schema.parameters,
            }
            TempInputSchema.model_json_schema = lambda s=schema_data: s["parameters"]

            # Generate and exec the model code to get the class
            input_code, input_class_name = await self._generate_input_model(schema_data)

            # Create a namespace and execute the model code
            namespace = {
                "BaseModel": BaseModel,
                "Field": BaseModel.__fields_set__,
                "Literal": Any,
            }
            exec(input_code, namespace)

            input_models[function_schema.name] = namespace[input_class_name]

        @app.post("/tools/{tool_name}")
        async def handle_tool_call(tool_name: str, input_data: dict) -> str:
            """Generic endpoint that routes all tool calls to user handler."""
            # Validate tool exists
            if tool_name not in tool_mappings:
                raise HTTPException(
                    status_code=404,
                    detail=f"Tool '{tool_name}' not found. Available: {list(tool_mappings.keys())}",
                )

            # Validate and parse input
            input_model_class = input_models[tool_name]
            try:
                input_props = input_model_class.model_validate(input_data)
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid input for tool '{tool_name}': {e}"
                )

            # Call user's handler
            try:
                result = await self.handler(tool_name, input_props)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Tool execution failed: {e}")

        self._server_app = app
        return app

    async def get_tool_functions(self) -> dict[str, Callable]:
        """Get ready-to-use tool functions."""
        if self._tool_functions is not None:
            return self._tool_functions

        start_time = time.time()
        logger.info("Starting tool functions generation")

        # Generate and execute the tools code
        tools_code = await self.generate_tools_code()
        logger.debug(f"Generated {len(tools_code)} characters of code")

        # Create namespace and execute
        namespace = {
            "BaseModel": BaseModel,
            "Field": BaseModel.__fields_set__,
            "Literal": Any,
            "List": list,
            "datetime": __import__("datetime").datetime,
        }

        logger.debug("Executing generated tools code...")
        exec_start = time.time()
        exec(tools_code, namespace)
        exec_elapsed = time.time() - exec_start
        logger.debug(f"Code execution completed in {exec_elapsed:.2f}s")

        # Extract tool functions
        tool_mappings = await self._get_tool_mappings()
        self._tool_functions = {}

        for tool_name in tool_mappings:
            if tool_name in namespace:
                self._tool_functions[tool_name] = namespace[tool_name]

        elapsed = time.time() - start_time
        logger.info(f"Tool functions generation completed in {elapsed:.2f}s")
        return self._tool_functions

    async def start_server(
        self, host: str = "0.0.0.0", port: int = 8000, background: bool = False
    ) -> None:
        """Start the FastAPI server.

        Args:
            host: Host to bind to
            port: Port to bind to
            background: If True, run server in background task
        """
        app = await self.generate_server_app()

        if background:
            # Start server in background
            import uvicorn

            config = uvicorn.Config(app, host=host, port=port)
            server = uvicorn.Server(config)
            task = asyncio.create_task(server.serve())
            return task
        # Run server blocking
        import uvicorn

        uvicorn.run(app, host=host, port=port)

    async def save_to_files(self, output_dir: Path) -> dict[str, Path]:
        """Save generated code to files.

        Args:
            output_dir: Directory to save files to

        Returns:
            Dictionary mapping file types to paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save tools module
        tools_code = await self.generate_tools_code()
        tools_file = output_dir / "generated_tools.py"
        tools_file.write_text(tools_code)
        saved_files["tools"] = tools_file

        # Save server code (as template/example)
        server_template = f'''"""FastAPI server using HttpToolExecutor."""

import asyncio
from pathlib import Path

from schemez.tool_executor import HttpToolExecutor, ToolHandler
from pydantic import BaseModel


async def my_tool_handler(method_name: str, input_props: BaseModel) -> str:
    """Implement your tool logic here."""
    match method_name:
        case _:
            return f"Mock result for {{method_name}}: {{input_props}}"


async def main():
    """Start the server with your handler."""
    executor = HttpToolExecutor(
        schemas=[],  # Add your schema files/dicts here
        handler=my_tool_handler,
        base_url="{self.base_url}"
    )

    await executor.start_server()


if __name__ == "__main__":
    asyncio.run(main())
'''

        server_file = output_dir / "server_example.py"
        server_file.write_text(server_template)
        saved_files["server_example"] = server_file

        return saved_files
