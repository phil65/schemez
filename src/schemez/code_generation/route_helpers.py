"""Helper functions for FastAPI route generation."""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any

from schemez.schema_to_type.simple_impl import json_schema_to_base_model


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic import BaseModel
    from pydantic.fields import FieldInfo


def create_param_model(parameters_schema: dict[str, Any]) -> type[BaseModel] | None:
    """Create Pydantic model for parameter validation using schemez.

    Args:
        parameters_schema: JSON schema for tool parameters

    Returns:
        Pydantic model class or None if no parameters
    """
    if parameters_schema.get("properties"):
        return json_schema_to_base_model(parameters_schema)
    return None


def generate_func_code(model_fields: dict[str, FieldInfo]) -> str:
    """Generate dynamic function code for FastAPI route handler.

    Args:
        model_fields: Model fields from Pydantic model

    Returns:
        Generated function code as string
    """
    route_params = []
    for name, field_info in model_fields.items():
        field_type = field_info.annotation
        if field_info.is_required():
            route_params.append(f"{name}: {field_type.__name__}")  # type: ignore[union-attr]
        else:
            route_params.append(f"{name}: {field_type.__name__} = None")  # type: ignore[union-attr]

    # Create function signature dynamically
    param_str = ", ".join(route_params)
    return f"""
async def dynamic_handler({param_str}) -> dict[str, Any]:
    kwargs = {{{", ".join(f'"{name}": {name}' for name in model_fields)}}}
    return await route_handler(**kwargs)
"""


def create_route_handler(
    tool_callable: Callable[..., Any], param_cls: type | None
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Create FastAPI route handler for a tool.

    Args:
        tool_callable: The tool function to execute
        param_cls: Pydantic model for parameter validation

    Returns:
        Async route handler function
    """

    async def route_handler(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Route handler for the tool."""
        if param_cls:
            params_instance = param_cls(**kwargs)  # Parse and validate parameters
            dct = params_instance.model_dump()  # Convert to dict and remove None values
            clean_params = {k: v for k, v in dct.items() if v is not None}
            result = await _execute_tool_function(tool_callable, **clean_params)
        else:
            result = await _execute_tool_function(tool_callable)
        return {"result": result}

    return route_handler


async def _execute_tool_function(tool_callable: Callable[..., Any], **kwargs: Any) -> Any:
    """Execute a tool function with the given parameters.

    Args:
        tool_callable: Tool function to execute
        **kwargs: Tool parameters

    Returns:
        Tool execution result
    """
    try:
        # Actually call the tool function
        if inspect.iscoroutinefunction(tool_callable):
            result = await tool_callable(**kwargs)
        else:
            # Run synchronous function in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: tool_callable(**kwargs)
            )
    except Exception as e:  # noqa: BLE001
        return f"Error executing {tool_callable.__name__}: {e!s}"
    else:
        return result


if __name__ == "__main__":
    from llmling_agent.tools.base import Tool

    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone."""
        return f"{greeting}, {name}!"

    # Create a tool and demonstrate helper functions
    tool = Tool.from_callable(greet)
    schema = tool.schema["function"]
    parameters_schema = schema.get("parameters", {})
    param_cls = create_param_model(dict(parameters_schema))
    print(f"Generated parameter model: {param_cls}")

    if param_cls:
        print(f"Model fields: {param_cls.model_fields}")
        func_code = generate_func_code(param_cls.model_fields)
        print(f"Generated function code:\n{func_code}")

    handler = create_route_handler(greet, param_cls)
    print(f"Generated route handler: {handler}")
