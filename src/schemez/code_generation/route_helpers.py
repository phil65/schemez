"""Helper functions for FastAPI route generation."""

import asyncio
import inspect
from typing import TYPE_CHECKING, Any

from schemez.helpers import get_object_name
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


_BUILTIN_TYPES: dict[Any, str] = {t: t.__name__ for t in (str, int, float, bool, list, dict)}


def generate_func_code(model_fields: dict[str, FieldInfo]) -> str:
    """Generate dynamic function code for FastAPI route handler.

    Args:
        model_fields: Model fields from Pydantic model

    Returns:
        Generated function code as string
    """
    route_params = []
    for name, field_info in model_fields.items():
        # Only builtins resolve in the handler's namespace (enums, models: Any)
        type_name = _BUILTIN_TYPES.get(field_info.annotation, "Any")
        if field_info.is_required():
            route_params.append(f"{name}: {type_name}")
        else:
            route_params.append(f"{name}: {type_name} = None")

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
            model_fields = getattr(param_cls, "model_fields", {})
            filtered = {
                k: v
                for k, v in kwargs.items()
                if not (v is None and k in model_fields and not model_fields[k].is_required())
            }
            params_instance = param_cls(**filtered)  # Parse and validate parameters
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
        name = get_object_name(tool_callable, "unknown")
        return f"Error executing {name}: {e!s}"
    else:
        return result


if __name__ == "__main__":
    from schemez import create_schema

    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone."""
        return f"{greeting}, {name}!"

    # Create a schema and demonstrate helper functions
    parameters_schema = create_schema(greet).parameters
    param_cls = create_param_model(dict(parameters_schema))
    print(f"Generated parameter model: {param_cls}")

    if param_cls:
        print(f"Model fields: {param_cls.model_fields}")
        func_code = generate_func_code(param_cls.model_fields)
        print(f"Generated function code:\n{func_code}")

    handler = create_route_handler(greet, param_cls)
    print(f"Generated route handler: {handler}")
