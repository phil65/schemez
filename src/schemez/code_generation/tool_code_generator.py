"""Meta-resource provider that exposes tools through Python execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from schemez import create_schema
from schemez.code_generation.route_helpers import (
    create_param_model,
    create_route_handler,
    generate_func_code,
)
from schemez.functionschema import FunctionSchema


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import FastAPI

    from schemez.functionschema import FunctionSchema, SchemaType


@dataclass
class ToolCodeGenerator:
    """Generates code artifacts for a single tool."""

    schema: FunctionSchema
    """Schema of the tool (primary source of truth)."""

    callable: Callable[..., Any] | None = None
    """Optional callable for actual execution. Required only for FastAPI route generation
    and Python namespace execution. All other operations (client code generation,
    signatures, models) work purely from the schema."""

    name_override: str | None = None
    """Name override for the function to generate code for."""

    @classmethod
    def from_callable(
        cls,
        fn: Callable[..., Any],
        exclude_types: list[type] | None = None,
        schema_type: SchemaType = "simple",
    ) -> ToolCodeGenerator:
        """Create a ToolCodeGenerator from a callable."""
        schema = create_schema(fn, exclude_types=exclude_types, mode=schema_type)
        return cls(schema=schema, callable=fn)

    @classmethod
    def from_schema(
        cls,
        schema: FunctionSchema,
        name_override: str | None = None,
    ) -> ToolCodeGenerator:
        """Create a ToolCodeGenerator from a schema only (no execution capability)."""
        return cls(schema=schema, callable=None, name_override=name_override)

    @property
    def name(self) -> str:
        """Name of the tool."""
        if self.name_override:
            return self.name_override
        if self.callable:
            return self.callable.__name__
        return self.schema.name

    def get_function_signature(self) -> str:
        """Assembles a signature for given generator, excluding (async) def."""
        return f"{self.name}{self.schema.to_python_signature()}"

    def get_function_definition(self, include_docstrings: bool = True) -> str:
        """Extract function definition using FunctionSchema."""
        parts = []
        parts.append(f"async def {self.get_function_signature()}")
        if include_docstrings and self.schema.description:
            lines = self.schema.description.split("\n")
            parts.extend(f"    {i.strip()}" for i in lines if i.strip())
        return "\n".join(parts)

    def generate_return_model(self) -> str | None:
        """Generate Pydantic model code for the tool's return type."""
        try:
            if self.schema.returns.get("type") not in {"object", "array"}:
                return None

            class_name = f"{self.name.title()}Response"
            model_code = self.schema.to_return_model_code(class_name=class_name)
            return model_code.strip() or None

        except Exception:  # noqa: BLE001
            return None

    def generate_parameter_model(self) -> str | None:
        """Generate Pydantic model code for the tool's parameters."""
        try:
            if not self.schema.parameters.get("properties"):
                return None

            class_name = f"{self.name.title()}Params"
            model_code = self.schema.to_parameter_model_code(class_name=class_name)
            return model_code.strip() or None

        except Exception:  # noqa: BLE001
            return None

    def generate_route_handler(self) -> Callable[..., Awaitable[dict[str, Any]]]:
        """Generate FastAPI route handler for this tool.

        Returns:
            Async route handler function

        Raises:
            ValueError: If callable is not provided
        """
        if self.callable is None:
            msg = f"Callable required for route generation for tool '{self.name}'"
            raise ValueError(msg)
        param_cls = create_param_model(dict(self.schema.parameters))
        return create_route_handler(self.callable, param_cls)

    def add_route_to_app(self, app: FastAPI, path_prefix: str = "/tools") -> None:
        """Add this tool's route to FastAPI app.

        Args:
            app: FastAPI application instance
            path_prefix: Path prefix for the route

        Raises:
            ValueError: If callable is not provided
        """
        if self.callable is None:
            msg = f"Callable required for route generation for tool '{self.name}'"
            raise ValueError(msg)
        param_cls = create_param_model(dict(self.schema.parameters))
        route_handler = self.generate_route_handler()
        # Set up the route with proper parameter annotations for FastAPI
        if param_cls:
            func_code = generate_func_code(param_cls.model_fields)
            namespace = {"route_handler": route_handler, "Any": Any}
            exec(func_code, namespace)  # Execute the dynamic function creation
            dynamic_handler: Callable = namespace["dynamic_handler"]  # type: ignore[assignment, type-arg]
        else:

            async def dynamic_handler() -> dict[str, Any]:
                return await route_handler()

        app.get(f"{path_prefix}/{self.name}")(dynamic_handler)  # Add route to FastAPI app


if __name__ == "__main__":
    import webbrowser

    generator = ToolCodeGenerator.from_callable(webbrowser.open)
    sig = generator.get_function_signature()
    print(sig)
