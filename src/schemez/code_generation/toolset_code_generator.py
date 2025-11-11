"""Orchestrates code generation for multiple tools."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import inspect
import time
from typing import TYPE_CHECKING, Any

from schemez import log
from schemez.code_generation.namespace_callable import NamespaceCallable
from schemez.code_generation.tool_code_generator import ToolCodeGenerator
from schemez.helpers import model_to_python_code


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fastapi import FastAPI

    from schemez.functionschema import FunctionSchema


logger = log.get_logger(__name__)


USAGE = """\
Usage notes:
- Write your code inside an 'async def main():' function
- All tool functions are async, use 'await'
- Use 'return' statements to return values from main()
- Generated model classes are available for type checking
- Use 'await report_progress(current, total, message)' for long-running operations
- DO NOT call asyncio.run() or try to run the main function yourself
- DO NOT import asyncio or other modules - tools are already available
- Example:
    async def main():
        for i in range(5):
            await report_progress(i, 5, f'Step {i+1} for {name}')
            should_continue = await ask_user('Continue?', 'bool')
            if not should_continue:
                break
        return f'Completed for {name}'

"""


@dataclass
class ToolsetCodeGenerator:
    """Generates code artifacts for multiple tools."""

    generators: Sequence[ToolCodeGenerator]
    """ToolCodeGenerator instances for each tool."""

    include_signatures: bool = True
    """Include function signatures in documentation."""

    include_docstrings: bool = True
    """Include function docstrings in documentation."""

    @classmethod
    def from_callables(
        cls,
        callables: Sequence[Callable],
        include_signatures: bool = True,
        include_docstrings: bool = True,
        exclude_types: list[type] | None = None,
    ) -> ToolsetCodeGenerator:
        """Create a ToolsetCodeGenerator from a sequence of callables.

        Args:
            callables: Callables to generate code for
            include_signatures: Include function signatures in documentation
            include_docstrings: Include function docstrings in documentation
            exclude_types: Parameter Types to exclude from the generated code
                           Often used for context parameters.

        Returns:
            ToolsetCodeGenerator instance
        """
        generators = [
            ToolCodeGenerator.from_callable(i, exclude_types=exclude_types)
            for i in callables
        ]
        return cls(generators, include_signatures, include_docstrings)

    @classmethod
    def from_schemas(
        cls,
        schemas: Sequence[FunctionSchema],
        include_signatures: bool = True,
        include_docstrings: bool = True,
    ) -> ToolsetCodeGenerator:
        """Create a ToolsetCodeGenerator from schemas only (no execution capability).

        This approach still allows generating client code.

        Args:
            schemas: FunctionSchemas to generate code for
            include_signatures: Include function signatures in documentation
            include_docstrings: Include function docstrings in documentation

        Returns:
            ToolsetCodeGenerator instance
        """
        generators = [ToolCodeGenerator.from_schema(schema) for schema in schemas]
        return cls(generators, include_signatures, include_docstrings)

    def generate_tool_description(self) -> str:
        """Generate comprehensive tool description with available functions."""
        if not self.generators:
            return "Execute Python code (no tools available)"

        return_models = self.generate_return_models()
        parts = [
            "Execute Python code with the following tools available as async functions:",
            "",
        ]

        if return_models:
            parts.extend([
                "# Generated return type models",
                return_models,
                "",
                "# Available functions:",
                "",
            ])

        for generator in self.generators:
            if self.include_signatures:
                signature = generator.get_function_signature()
                parts.append(f"async def {signature}:")
            else:
                parts.append(f"async def {generator.name}(...):")

            # Use schema description or callable docstring if available
            docstring = None
            if self.include_docstrings:
                if generator.schema.description:
                    docstring = generator.schema.description
                elif generator.callable and generator.callable.__doc__:
                    docstring = generator.callable.__doc__

            if docstring:
                indented_desc = "    " + docstring.replace("\n", "\n    ")

                # Add warning for async functions without proper return type hints
                if generator.callable and inspect.iscoroutinefunction(generator.callable):
                    sig = inspect.signature(generator.callable)
                    if sig.return_annotation == inspect.Signature.empty:
                        indented_desc += "\n    \n    Note: This async function should explicitly return a value."  # noqa: E501

                parts.append(f'    """{indented_desc}"""')
            parts.append("")

        parts.append(USAGE)

        return "\n".join(parts)

    def generate_execution_namespace(self) -> dict[str, Any]:
        """Build Python namespace with tool functions and generated models.

        Raises:
            ValueError: If any generator lacks a callable
        """
        namespace: dict[str, Any] = {"__builtins__": __builtins__, "_result": None}

        # Add tool functions - all generators must have callables for execution
        for generator in self.generators:
            namespace[generator.name] = NamespaceCallable.from_generator(generator)

        # Add generated model classes to namespace
        if models_code := self.generate_return_models():
            with contextlib.suppress(Exception):
                exec(models_code, namespace)

        return namespace

    def generate_return_models(self) -> str:
        """Generate Pydantic models for tool return types."""
        model_parts = [
            code for g in self.generators if (code := g.generate_return_model())
        ]
        return "\n\n".join(model_parts) if model_parts else ""

    def add_all_routes(self, app: FastAPI, path_prefix: str = "/tools") -> None:
        """Add FastAPI routes for all tools.

        Args:
            app: FastAPI application instance
            path_prefix: Path prefix for routes
        """
        for generator in self.generators:
            if generator.callable is None:
                tool_name = generator.name
                msg = (
                    f"Callable required for route generation for tool '{tool_name}'. "
                    "Use from_callables() or provide callable when creating generator."
                )
                raise ValueError(msg)
            generator.add_route_to_app(app, path_prefix)

    def generate_client_code(
        self, base_url: str = "http://localhost:8000", path_prefix: str = "/tools"
    ) -> str:
        """Generate HTTP client code for all tools.

        Creates a complete Python module with HTTP wrapper functions that call
        the corresponding server endpoints. This is the client-side counterpart
        to the `add_all_routes` method.

        Args:
            base_url: Base URL of the tool server
            path_prefix: Path prefix for routes (must match server-side)

        Returns:
            Complete Python module code with:
            - Pydantic input models for each tool
            - Async HTTP wrapper functions
            - Proper imports and exports

        Example:
            >>> toolset = ToolsetCodeGenerator.from_callables([greet, add_numbers])
            >>> client_code = await toolset.generate_client_code()
            >>> # Save to file or exec() the generated code
        """
        start_time = time.time()
        logger.info("Starting client code generation")

        code_parts: list[str] = []

        # Module header
        header = '''"""Generated HTTP client tools."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, List, Any, Dict
from datetime import datetime

'''
        code_parts.append(header)

        # Generate models and wrappers for each tool
        all_exports = []
        for generator in self.generators:
            # Generate input model from schema parameters
            try:
                # Get parameters schema and generate model if it has properties
                params_schema = generator.schema.parameters
                if params_schema.get("properties"):
                    # Use the same pattern as HttpToolExecutor
                    words = [word.title() for word in generator.name.split("_")]
                    input_class_name = f"{''.join(words)}Input"

                    model_code = model_to_python_code(
                        params_schema, class_name=input_class_name
                    )
                    if model_code:
                        # Clean up the model code (remove duplicate imports)
                        cleaned_model = self._clean_generated_code(model_code)
                        code_parts.append(cleaned_model)
                    else:
                        # Fallback for tools without parameters
                        input_class_name = "BaseModel"
                else:
                    # No parameters, use BaseModel
                    input_class_name = "BaseModel"
            except (ValueError, TypeError, AttributeError):
                # Fallback input model for schema parsing errors
                input_class_name = "BaseModel"

            # Generate HTTP wrapper function
            description = generator.schema.description or f"Call {generator.name} tool"
            wrapper_code = f'''
async def {generator.name}(input: {input_class_name}) -> str:
    """{description}

    Args:
        input: Function parameters

    Returns:
        String response from the tool server
    """
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "{base_url}{path_prefix}/{generator.name}",
            params=input.model_dump() if hasattr(input, 'model_dump') else {{}},
            timeout=30.0
        )
        response.raise_for_status()
        return response.text
'''
            code_parts.append(wrapper_code)

            if input_class_name != "BaseModel":
                all_exports.append(input_class_name)
            all_exports.append(generator.name)

        # Add exports
        code_parts.append(f"\n__all__ = {all_exports}\n")

        client_code = "\n".join(code_parts)
        elapsed = time.time() - start_time
        logger.info("Client code generation completed in %.2fs", elapsed)
        return client_code

    def generate_function_stubs(self) -> str:
        """Generate clean function stubs for LLM consumption.

        Returns Python code with just signatures, docstrings, and input models
        - no HTTP implementation details. Perfect for showing LLMs what
        functions are available without implementation noise.

        Returns:
            Clean Python code with function stubs
        """
        start_time = time.time()
        logger.info("Starting function stubs generation")

        code_parts: list[str] = []

        # Module header
        header = '"""Available tool functions."""\n\n'
        code_parts.append(header)

        # Generate models and stubs for each tool
        all_exports = []
        for generator in self.generators:
            # Generate input model from schema parameters
            try:
                params_schema = generator.schema.parameters
                if params_schema.get("properties"):
                    # Use the same pattern as client code generation
                    words = [word.title() for word in generator.name.split("_")]
                    input_class_name = f"{''.join(words)}Input"

                    model_code = model_to_python_code(
                        params_schema, class_name=input_class_name
                    )
                    if model_code:
                        # Clean up the model code (remove duplicate imports)
                        cleaned_model = self._clean_generated_code(model_code)
                        code_parts.append(cleaned_model)
                    else:
                        # Fallback for tools without parameters
                        input_class_name = "BaseModel"
                else:
                    # No parameters, use BaseModel
                    input_class_name = "BaseModel"
            except (ValueError, TypeError, AttributeError):
                # Fallback input model for schema parsing errors
                input_class_name = "BaseModel"

            # Generate function stub
            description = generator.schema.description or f"Call {generator.name} tool"

            # Get return type hint from schema or default to str
            try:
                signature_str = str(generator.schema.to_python_signature())
                # Extract return type from signature
                return_hint = (
                    signature_str.split(" -> ")[1] if " -> " in signature_str else "Any"
                )
            except Exception:  # noqa: BLE001
                return_hint = "Any"

            stub_code = f'''
async def {generator.name}(input: {input_class_name}) -> {return_hint}:
    """{description}

    Args:
        input: Function parameters

    Returns:
        Function result
    """
    ...
'''
            code_parts.append(stub_code)

            if input_class_name != "BaseModel":
                all_exports.append(input_class_name)
            all_exports.append(generator.name)

        # Add exports
        code_parts.append(f"\n__all__ = {all_exports}\n")

        stubs_code = "\n".join(code_parts)
        elapsed = time.time() - start_time
        logger.info("Function stubs generation completed in %.2fs", elapsed)
        return stubs_code

    def _clean_generated_code(self, code: str) -> str:
        """Clean up generated code by removing redundant imports and headers."""
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
                    and not line.startswith("from pydantic import")
                    and not line.startswith("from typing import")
                    and not line.startswith("from datetime import")
                ):
                    skip_until_class = False
                    cleaned_lines.append(line)
                continue
            # Skip redundant imports that are already in the header
            if (
                line.strip().startswith("from __future__")
                or line.strip().startswith("from pydantic import")
                or line.strip().startswith("from typing import")
                or line.strip().startswith("from datetime import")
            ):
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)


if __name__ == "__main__":

    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone with a custom message."""
        return f"{greeting}, {name}!"

    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    generator = ToolsetCodeGenerator.from_callables([greet, add_numbers])

    # Test client code generation
    client_code = generator.generate_client_code()
    print("Generated client code:")
    print(client_code)
