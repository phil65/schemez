"""Orchestrates code generation for multiple tools."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import inspect
import time
from typing import TYPE_CHECKING, Any, Literal

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
class GeneratedCode:
    """Structured code generation result."""

    models: str
    """Generated Pydantic input models."""

    http_methods: str
    """HTTP client methods using models."""

    clean_methods: str
    """Clean signature methods without models."""

    stubs: str
    """Function stubs for LLM consumption."""

    imports: str = ""
    """Common imports."""

    exports: list[str] = field(default_factory=list)
    """Exported names."""

    def get_client_code(
        self, mode: Literal["models", "simple", "stubs"] = "models"
    ) -> str:
        """Generate client code in specified mode.

        Args:
            mode: Generation mode - "models" (with Pydantic models),
                  "simple" (clean signatures), or "stubs" (function stubs)

        Returns:
            Formatted client code
        """
        parts = []
        if self.imports:
            parts.append(self.imports)

        if mode == "models":
            # Complete HTTP client with models
            if self.models:
                parts.append(self.models)
            if self.http_methods:
                parts.append(self.http_methods)
            if self.exports:
                parts.append(f"\n__all__ = {self.exports}\n")

        elif mode == "simple":
            # Clean client with natural signatures
            if self.clean_methods:
                parts.append(self.clean_methods)
            if self.exports:
                exports = [name for name in self.exports if not name.endswith("Input")]
                parts.append(f"\n__all__ = {exports}\n")

        elif mode == "stubs":
            # Function stubs for LLM consumption
            if self.models:
                parts.append(self.models)
            if self.stubs:
                parts.append(self.stubs)
            if self.exports:
                parts.append(f"\n__all__ = {self.exports}\n")

        else:
            msg = f"Unknown mode: {mode}. Use 'models', 'simple', or 'stubs'"
            raise ValueError(msg)

        return "\n".join(parts)


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

    def generate_structured_code(
        self, base_url: str = "http://localhost:8000", path_prefix: str = "/tools"
    ) -> GeneratedCode:
        """Generate structured code with all components.

        Args:
            base_url: Base URL of the tool server
            path_prefix: Path prefix for routes

        Returns:
            GeneratedCode with all components separated
        """
        start_time = time.time()
        logger.info("Starting structured code generation")

        # Common imports
        imports = '''"""Generated tool client code."""

from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, List, Any, Dict
from datetime import datetime
import httpx

'''

        # Generate input models
        models_parts = []
        http_methods_parts = []
        clean_methods_parts = []
        stubs_parts = []
        all_exports = []

        for generator in self.generators:
            # Generate input model from schema parameters
            input_class_name = None
            try:
                params_schema = generator.schema.parameters
                if params_schema.get("properties"):
                    words = [word.title() for word in generator.name.split("_")]
                    input_class_name = f"{''.join(words)}Input"

                    model_code = model_to_python_code(
                        params_schema, class_name=input_class_name
                    )
                    if model_code:
                        cleaned_model = self._clean_generated_code(model_code)
                        models_parts.append(cleaned_model)
                        all_exports.append(input_class_name)
            except (ValueError, TypeError, AttributeError):
                input_class_name = None

            # Generate HTTP method with model
            if input_class_name:
                http_method = f'''
async def {generator.name}(input: {input_class_name}) -> str:
    """{generator.schema.description or f"Call {generator.name} tool"}

    Args:
        input: Function parameters

    Returns:
        String response from the tool server
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "{base_url}{path_prefix}/{generator.name}",
            params=input.model_dump() if hasattr(input, 'model_dump') else {{}},
            timeout=30.0
        )
        response.raise_for_status()
        return response.text
'''
            else:
                http_method = f'''
async def {generator.name}() -> str:
    """{generator.schema.description or f"Call {generator.name} tool"}

    Returns:
        String response from the tool server
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "{base_url}{path_prefix}/{generator.name}",
            timeout=30.0
        )
        response.raise_for_status()
        return response.text
'''
            http_methods_parts.append(http_method)

            # Generate clean method with natural signature
            signature_str = generator.get_function_signature()
            params_schema = generator.schema.parameters
            param_names = list(params_schema.get("properties", {}).keys())

            clean_method = f'''
async def {signature_str}:
    """{generator.schema.description or f"Call {generator.name} tool"}"""
    # Build parameters dict
    params = {{{", ".join(f'"{name}": {name}' for name in param_names)}}}
    # Remove None values
    clean_params = {{k: v for k, v in params.items() if v is not None}}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "{base_url}{path_prefix}/{generator.name}",
            params=clean_params,
            timeout=30.0
        )
        response.raise_for_status()
        # Parse JSON response and return the result
        result = response.json()
        return result.get("result", response.text)
'''
            clean_methods_parts.append(clean_method)

            # Generate stub
            if input_class_name:
                stub = f'''
async def {generator.name}(input: {input_class_name}) -> str:
    """{generator.schema.description or f"Call {generator.name} tool"}

    Args:
        input: Function parameters

    Returns:
        Function result
    """
    ...
'''
            else:
                stub = f'''
async def {generator.name}() -> str:
    """{generator.schema.description or f"Call {generator.name} tool"}

    Returns:
        Function result
    """
    ...
'''
            stubs_parts.append(stub)

            all_exports.append(generator.name)

        elapsed = time.time() - start_time
        logger.info("Structured code generation completed in %.2fs", elapsed)

        return GeneratedCode(
            models="\n".join(models_parts),
            http_methods="\n".join(http_methods_parts),
            clean_methods="\n".join(clean_methods_parts),
            stubs="\n".join(stubs_parts),
            imports=imports,
            exports=all_exports,
        )

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

    def generate_code(
        self,
        mode: Literal["models", "simple", "stubs"] = "models",
        base_url: str = "http://localhost:8000",
        path_prefix: str = "/tools",
    ) -> str:
        """Generate client code in the specified mode.

        Args:
            mode: Generation mode - "models" (with Pydantic models),
                  "simple" (clean signatures), or "stubs" (function stubs)
            base_url: Base URL of the tool server
            path_prefix: Path prefix for routes (must match server-side)

        Returns:
            Generated client code in the specified mode
        """
        structured_code = self.generate_structured_code(base_url, path_prefix)
        return structured_code.get_client_code(mode)

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

    # Test new structured code generation
    print("🚀 MODELS MODE (with Pydantic models):")
    print("=" * 50)
    models_code = generator.generate_code(mode="models")
    print(models_code[:400] + "...\n")

    print("🚀 SIMPLE MODE (clean signatures):")
    print("=" * 50)
    simple_code = generator.generate_code(mode="simple")
    print(simple_code[:400] + "...\n")

    print("🚀 STUBS MODE (for LLM consumption):")
    print("=" * 50)
    stubs_code = generator.generate_code(mode="stubs")
    print(stubs_code[:400] + "...")
