"""Orchestrates code generation for multiple tools."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import inspect
import time
from typing import TYPE_CHECKING, Any, Literal

from schemez import log
from schemez.code_generation.codegen import clean_generated_code, generate_tool_module
from schemez.code_generation.namespace_callable import NamespaceCallable
from schemez.code_generation.tool_code_generator import ToolCodeGenerator
from schemez.helpers import model_to_python_code


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fastapi import FastAPI

    from schemez.functionschema import FunctionSchema


logger = log.get_logger(__name__)

ArgsFormat = Literal["model", "explicit"]
OutputType = Literal["stubs", "implementation"]


IMPORTS = '''"""Generated tool client code."""

from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, List, Any, Dict
from datetime import datetime
import httpx

'''

STUB_IMPORTS = '''"""Generated tool stubs for LLM consumption."""

from __future__ import annotations

from pydantic import BaseModel

'''


@dataclass
class GeneratedCode:
    """Structured code generation result."""

    models: str
    """Generated Pydantic input models."""

    http_methods: str
    """HTTP client methods using models."""

    clean_methods: str
    """Clean signature methods without models."""

    model_stubs: str
    """Model-based function stubs for LLM consumption."""

    explicit_stubs: str
    """Explicit signature function stubs for LLM consumption."""

    imports: str = ""
    """Common imports."""

    def get_client_code(
        self,
        args_format: ArgsFormat = "explicit",
        output_type: OutputType = "implementation",
    ) -> str:
        """Generate client code with specified format and type.

        Args:
            args_format: Argument format - "model" (with Pydantic models) or
                "explicit" (function signatures)
            output_type: Output type - "stubs" (type stubs) or
                "implementation" (working HTTP client)

        Returns:
            Formatted client code
        """
        parts = []

        # Get stub imports for LLM consumption

        match (args_format, output_type):
            case ("model", "implementation"):
                # HTTP client with models
                if self.imports:
                    parts.append(self.imports)
                if self.models:
                    parts.append(self.models)
                if self.http_methods:
                    parts.append(self.http_methods)

            case ("explicit", "implementation"):
                # HTTP client with explicit signatures
                if self.imports:
                    parts.append(self.imports)
                if self.clean_methods:
                    parts.append(self.clean_methods)

            case ("model", "stubs"):
                # Model-based stubs for LLM consumption (minimal imports)
                parts.append(STUB_IMPORTS)
                if self.models:
                    parts.append(self.models)
                if self.model_stubs:
                    parts.append(self.model_stubs)

            case ("explicit", "stubs"):
                # Explicit signature stubs for LLM consumption (minimal imports)
                parts.append(STUB_IMPORTS)
                if self.explicit_stubs:
                    parts.append(self.explicit_stubs)

            case _:
                msg = (
                    f"Unknown combination: args_format={args_format}, "
                    f"output_type={output_type}"
                )
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
        parts = ["Execute Python code with the following async functions available:", ""]
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
        model_parts = [c for g in self.generators if (c := g.generate_return_model())]
        return "\n\n".join(model_parts) if model_parts else ""

    def generate_sandbox_files(self) -> dict[str, str]:
        """Generate Python files for sandbox execution (ctx-zip style approach).

        Returns dict mapping file paths to file contents for writing to sandbox.
        Each tool becomes a separate .py file that can be imported directly.
        """
        files = {}

        # Generate individual tool files
        for generator in self.generators:
            tool_code = generate_tool_module(generator)
            files[f"tools/{generator.name}.py"] = tool_code

        # Generate __init__.py for tools package
        tool_imports = [f"from .{gen.name} import {gen.name}" for gen in self.generators]
        init_content = (
            "\n".join(tool_imports)
            + "\n\n__all__ = "
            + str([gen.name for gen in self.generators])
        )
        files["tools/__init__.py"] = init_content

        # Generate models file if needed
        if models_code := self.generate_return_models():
            files["tools/models.py"] = (
                f'"""Generated Pydantic models for tool return types."""\n\nfrom pydantic import BaseModel\n\n{models_code}'  # noqa: E501
            )

        return files

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
        # Generate input models
        models_parts: list[str] = []
        http_methods_parts: list[str] = []
        clean_methods_parts: list[str] = []
        model_stubs_parts: list[str] = []
        explicit_stubs_parts: list[str] = []

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
                        cleaned_model = clean_generated_code(model_code)
                        models_parts.append(cleaned_model)
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

            # Generate model-based stub
            if input_class_name:
                model_stub = f'''
async def {generator.name}(input: {input_class_name}) -> str:
    """{generator.schema.description or f"Call {generator.name} tool"}

    Args:
        input: Function parameters

    Returns:
        Function result
    """
    ...
'''
                model_stubs_parts.append(model_stub)

            # Generate explicit signature stub
            explicit_stub = f'''
async def {signature_str}:
    """{generator.schema.description or f"Call {generator.name} tool"}"""
    ...
'''
            explicit_stubs_parts.append(explicit_stub)

        elapsed = time.time() - start_time
        logger.info("Structured code generation completed in %.2fs", elapsed)

        return GeneratedCode(
            models="\n".join(models_parts),
            http_methods="\n".join(http_methods_parts),
            clean_methods="\n".join(clean_methods_parts),
            model_stubs="\n".join(model_stubs_parts),
            explicit_stubs="\n".join(explicit_stubs_parts),
            imports=IMPORTS,
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
        args_format: ArgsFormat = "explicit",
        output_type: OutputType = "implementation",
        base_url: str = "http://localhost:8000",
        path_prefix: str = "/tools",
    ) -> str:
        """Generate client code with specified format and type.

        Args:
            args_format: Argument format - "model" (with Pydantic models) or
                "explicit" (function signatures)
            output_type: Output type - "stubs" (type stubs) or
                "implementation" (working HTTP client)
            base_url: Base URL of the tool server
            path_prefix: Path prefix for routes (must match server-side)

        Returns:
            Generated client code
        """
        structured_code = self.generate_structured_code(base_url, path_prefix)
        return structured_code.get_client_code(args_format, output_type)


if __name__ == "__main__":

    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone with a custom message."""
        return f"{greeting}, {name}!"

    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    generator = ToolsetCodeGenerator.from_callables([greet, add_numbers])

    # Test new structured code generation
    print("🚀 MODEL IMPLEMENTATION (HTTP client with Pydantic models):")
    print("=" * 50)
    model_impl_code = generator.generate_code(args_format="model")
    print(model_impl_code)

    print("🚀 EXPLICIT IMPLEMENTATION (HTTP client with explicit signatures):")
    print("=" * 50)
    explicit_impl_code = generator.generate_code(args_format="explicit")
    print(explicit_impl_code)

    print("🚀 MODEL STUBS (Pydantic model stubs for LLM):")
    print("=" * 50)
    model_stubs_code = generator.generate_code(args_format="model", output_type="stubs")
    print(model_stubs_code)

    print("🚀 EXPLICIT STUBS (Function signature stubs for LLM):")
    print("=" * 50)
    explicit_stubs_code = generator.generate_code(output_type="stubs")
    print(explicit_stubs_code[:400] + "...")
