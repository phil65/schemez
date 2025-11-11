"""Meta-resource provider that exposes tools through Python execution."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import TYPE_CHECKING, Any, get_origin

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
    from schemez.typedefs import Property, ToolParameters


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

    callable: Callable
    """Tool to generate code for."""

    schema: FunctionSchema
    """Schema of the tool."""

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
        """Create a ToolCodeGenerator from a Tool."""
        schema = create_schema(fn)
        return cls(schema=schema, callable=fn, exclude_types=exclude_types or [])

    @property
    def name(self) -> str:
        """Name of the tool."""
        return self.name_override or self.callable.__name__

    def _get_schema_params(self) -> ToolParameters:
        """Get parameters from the schema."""
        return self.schema.parameters

    def _infer_parameter_type(self, param_name: str, param_info: Property) -> str:
        """Infer parameter type from schema and function inspection."""
        schema_type = param_info.get("type", "Any")

        # If schema has a specific type, use it
        if schema_type != "object":
            return TYPE_MAP.get(schema_type, "Any")

        # For 'object' type, try to infer from function signature
        try:
            callable_func = self.callable
            # Use wrapped signature if available (for context parameter hiding)
            sig = getattr(callable_func, "__signature__", None) or inspect.signature(
                callable_func
            )

            if param_name in sig.parameters:
                param = sig.parameters[param_name]

                # Try annotation first
                if param.annotation != inspect.Parameter.empty:
                    if hasattr(param.annotation, "__name__"):
                        return param.annotation.__name__
                    return str(param.annotation)

                # Infer from default value
                if param.default != inspect.Parameter.empty:
                    default_type = type(param.default).__name__
                    # Map common types
                    if default_type in ["int", "float", "str", "bool"]:
                        return default_type
                # If no default and it's required, assume str for web-like functions
                required = set(self.schema.parameters.get("required", []))
                if param_name in required:
                    return "str"

        except Exception:  # noqa: BLE001
            pass

        # Fallback to Any for unresolved object types
        return "Any"

    def get_function_signature(self) -> str:
        """Extract function signature using FunctionSchema."""
        try:
            sig = self.schema.to_python_signature()
            # Filter context parameters
            filtered_params = [
                p
                for p in sig.parameters.values()
                if not self._is_context_parameter(p.name)
            ]
            filtered_sig = sig.replace(parameters=filtered_params)
        except Exception:  # noqa: BLE001
            return f"{self.name}(...) -> Any"
        else:
            return f"{self.name}{filtered_sig}"

    def _get_callable_signature(self) -> inspect.Signature:
        """Get signature from callable, respecting wrapped signatures."""
        # Use wrapped signature if available (for context parameter hiding)
        return getattr(self.callable, "__signature__", None) or inspect.signature(
            self.callable
        )

    def _is_context_parameter(self, param_name: str) -> bool:
        """Check if a parameter is a context parameter that should be hidden."""
        try:
            sig = self._get_callable_signature()
            if param_name not in sig.parameters:
                return False

            param = sig.parameters[param_name]
            if param.annotation == inspect.Parameter.empty:
                return False

            annotation = param.annotation

            for typ in self.exclude_types:
                if self._types_match(annotation, typ):
                    return True
        except Exception:  # noqa: BLE001
            pass

        return False

    def _types_match(self, annotation: Any, exclude_type: type) -> bool:
        """Check if annotation matches exclude_type using various strategies."""
        try:
            # Direct type match
            if annotation is exclude_type:
                return True

            # Handle generic types - get origin for comparison
            origin_annotation = get_origin(annotation)
            if origin_annotation is exclude_type:
                return True

            # String-based comparison for forward references and __future__.annotations
            annotation_str = str(annotation)
            exclude_type_name = exclude_type.__name__
            exclude_type_full_name = f"{exclude_type.__module__}.{exclude_type.__name__}"

            # Check various string representations
            if (
                exclude_type_name in annotation_str
                or exclude_type_full_name in annotation_str
            ):
                # Be more specific to avoid false positives
                # Check if it's the exact type name, not just a substring
                import re

                patterns = [
                    rf"\b{re.escape(exclude_type_name)}\b",
                    rf"\b{re.escape(exclude_type_full_name)}\b",
                ]
                if any(re.search(pattern, annotation_str) for pattern in patterns):
                    return True

        except Exception:  # noqa: BLE001
            pass

        return False

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
        """
        param_cls = create_param_model(dict(self.schema.parameters))
        return create_route_handler(self.callable, param_cls)

    def add_route_to_app(self, app: FastAPI, path_prefix: str = "/tools") -> None:
        """Add this tool's route to FastAPI app.

        Args:
            app: FastAPI application instance
            path_prefix: Path prefix for the route
        """
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
