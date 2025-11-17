"""HTTP tool executor for managing tool generation and execution."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic_core import from_json
from upath import UPath

from schemez import log
from schemez.functionschema import FunctionSchema
from schemez.tool_executor.helpers import clean_generated_code, generate_input_model


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fastapi import FastAPI
    from upath.types import JoinablePathLike

    from schemez.tool_executor.types import ToolHandler


logger = log.get_logger(__name__)


class HttpToolExecutor:
    """Manages HTTP tool generation and execution."""

    def __init__(
        self,
        schemas: Sequence[dict[str, Any] | JoinablePathLike],
        handler: ToolHandler,
        base_url: str = "http://localhost:8000",
    ) -> None:
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
        self._tool_functions: dict[str, Callable[..., Any]] | None = None

    async def _load_schemas(self) -> list[dict[str, Any]]:
        """Load and normalize schemas from various sources."""
        loaded_schemas = []

        for schema in self.schemas:
            match schema:
                case dict():
                    loaded_schemas.append(schema)
                case str() | Path() | UPath():
                    text = UPath(schema).read_text("utf-8")
                    loaded_schemas.append(from_json(text))
                case _:
                    msg = f"Invalid schema type: {type(schema)}"
                    raise TypeError(msg)

        return loaded_schemas

    async def _get_tool_mappings(self) -> dict[str, str]:
        """Get tool name to input class mappings."""
        if self._tool_mappings is None:
            self._tool_mappings = {}
            schemas = await self._load_schemas()

            for schema_dict in schemas:
                function_schema = FunctionSchema.from_dict(schema_dict)
                name = "".join(word.title() for word in function_schema.name.split("_"))
                input_class_name = f"{name}Input"
                self._tool_mappings[function_schema.name] = input_class_name

        return self._tool_mappings

    async def _generate_http_wrapper(
        self, schema_dict: dict[str, Any], input_class_name: str
    ) -> str:
        """Generate HTTP wrapper function."""
        name = schema_dict["name"]
        description = schema_dict.get("description", "")

        return f'''
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

    async def generate_tools_code(self) -> str:
        """Generate HTTP wrapper tools as Python code."""
        if self._tools_code is not None:
            return self._tools_code

        start_time = time.time()
        logger.info("Starting tools code generation")

        schemas = await self._load_schemas()
        code_parts: list[str] = []
        await self._get_tool_mappings()

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
            input_code, input_class_name = await generate_input_model(schema_data)
            # Remove future imports and datamodel-codegen header from individual models
            cleaned_input_code = clean_generated_code(input_code)
            code_parts.append(cleaned_input_code)
            wrapper_code = await self._generate_http_wrapper(
                schema_data, input_class_name
            )
            code_parts.append(wrapper_code)

            all_exports.extend([input_class_name, function_schema.name])

        # Add exports
        code_parts.append(f"\n__all__ = {all_exports}\n")

        self._tools_code = "\n".join(code_parts)
        elapsed = time.time() - start_time
        logger.info(f"Tools code generation completed in {elapsed:.2f}s")  # noqa: G004
        return self._tools_code

    async def generate_server_app(self) -> FastAPI:
        """Create configured FastAPI server."""
        from fastapi import FastAPI, HTTPException

        if self._server_app is not None:
            return self._server_app

        tool_mappings = await self._get_tool_mappings()
        app = FastAPI(title="Tool Server", version="1.0.0")

        @app.post("/tools/{tool_name}")
        async def handle_tool_call(tool_name: str, input_data: dict[str, Any]) -> str:
            """Generic endpoint that routes all tool calls to user handler."""
            # Validate tool exists
            if tool_name not in tool_mappings:
                tools = list(tool_mappings.keys())
                detail = f"Tool '{tool_name}' not found. Available: {tools}"
                raise HTTPException(status_code=404, detail=detail)

            # Create a simple BaseModel for validation
            class DynamicInput(BaseModel):
                pass

            # Add fields dynamically (basic validation only)
            dynamic_input = DynamicInput(**input_data)

            # Call user's handler
            try:
                return await self.handler(tool_name, dynamic_input)
            except Exception as e:
                msg = f"Tool execution failed: {e}"
                raise HTTPException(status_code=500, detail=msg) from e

        self._server_app = app
        return app

    async def get_tool_functions(self) -> dict[str, Callable[..., Any]]:
        """Get ready-to-use tool functions."""
        if self._tool_functions is not None:
            return self._tool_functions

        start_time = time.time()
        logger.info("Starting tool functions generation")
        tools_code = await self.generate_tools_code()
        logger.debug("Generated %s characters of code", len(tools_code))
        namespace = {
            "BaseModel": BaseModel,
            "Field": BaseModel.model_fields_set,
            "Literal": Any,
            "List": list,
            "datetime": __import__("datetime").datetime,
        }
        logger.debug("Executing generated tools code...")
        exec_start = time.time()
        exec(tools_code, namespace)
        exec_elapsed = time.time() - exec_start
        logger.debug("Code execution completed in %.2fs", exec_elapsed)

        # Extract tool functions
        tool_mappings = await self._get_tool_mappings()
        self._tool_functions = {
            tool_name: namespace[tool_name]
            for tool_name in tool_mappings
            if tool_name in namespace
        }

        elapsed = time.time() - start_time
        logger.info(f"Tool functions generation completed in {elapsed:.2f}s")  # noqa: G004
        return self._tool_functions

    async def start_server(
        self, host: str = "0.0.0.0", port: int = 8000, background: bool = False
    ) -> None | asyncio.Task[None]:
        """Start the FastAPI server.

        Args:
            host: Host to bind to
            port: Port to bind to
            background: If True, run server in background task
        """
        import uvicorn

        app = await self.generate_server_app()

        if background:
            config = uvicorn.Config(app, host=host, port=port)
            server = uvicorn.Server(config)
            return asyncio.create_task(server.serve())

        uvicorn.run(app, host=host, port=port)
        return None

    async def save_to_files(self, output_dir: JoinablePathLike) -> dict[str, UPath]:
        """Save generated code to files.

        Args:
            output_dir: Directory to save files to

        Returns:
            Dictionary mapping file types to paths
        """
        output_dir = UPath(output_dir)
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
