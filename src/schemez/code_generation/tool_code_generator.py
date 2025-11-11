"""Meta-resource provider that exposes tools through Python execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from schemez import create_schema
from schemez.code_generation.route_helpers import (
    create_param_model,
    create_route_handler,
    generate_func_code,
)
from schemez.functionschema import FunctionSchema


if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import FastAPI

    from schemez.functionschema import FunctionSchema
    from schemez.typedefs import ToolParameters


TYPE_MAP = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "null": "None",
}


@dataclass
class ToolCodeGenerator:
    """Generates code artifacts for a single tool."""

    schema: FunctionSchema
    """Schema of the tool (primary source of truth)."""

    callable: Callable | None = None
    """Optional callable for actual execution. Required only for FastAPI route generation
    and Python namespace execution. All other operations (client code generation,
    signatures, models) work purely from the schema."""

    name_override: str | None = None
    """Name override for the function to generate code for."""

    exclude_types: list[type] = field(default_factory=list)
    """Exclude parameters from generated code (like context types)."""

    @classmethod
    def from_callable(
        cls,
        fn: Callable,
        exclude_types: list[type] | None = None,
    ) -> ToolCodeGenerator:
        """Create a ToolCodeGenerator from a callable."""
        schema = create_schema(fn, exclude_types=exclude_types)
        return cls(schema=schema, callable=fn, exclude_types=exclude_types or [])

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

    def _get_schema_params(self) -> ToolParameters:
        """Get parameters from the schema."""
        return self.schema.parameters

    def get_function_signature(self) -> str:
        """Extract function signature using FunctionSchema."""
        try:
            sig = self.schema.to_python_signature()
        except Exception:  # noqa: BLE001
            return f"{self.name}(...) -> Any"
        else:
            return f"{self.name}{sig}"

    def generate_return_model(self) -> str | None:
        """Generate Pydantic model code for the tool's return type."""
        try:
            if self.schema.returns.get("type") not in {"object", "array"}:
                return None

            class_name = f"{self.name.title()}Response"
            model_code = self.schema.to_pydantic_model_code(class_name=class_name)
            return model_code.strip() or None

        except Exception:  # noqa: BLE001
            return None

    # Route generation methods
    def generate_route_handler(self) -> Callable:
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
            # Get field information from the generated model
            model_fields = param_cls.model_fields
            func_code = generate_func_code(model_fields)
            # Execute the dynamic function creation
            namespace = {"route_handler": route_handler, "Any": Any}
            exec(func_code, namespace)
            dynamic_handler: Callable = namespace["dynamic_handler"]  # type: ignore
        else:

            async def dynamic_handler() -> dict[str, Any]:
                return await route_handler()

        # Add route to FastAPI app
        app.get(f"{path_prefix}/{self.name}")(dynamic_handler)


if __name__ == "__main__":
    import webbrowser

    generator = ToolCodeGenerator.from_callable(webbrowser.open)
    sig = generator.get_function_signature()
    print(sig)
