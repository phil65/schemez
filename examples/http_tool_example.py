"""Example demonstrating HttpToolExecutor for HTTP tool generation.

This example shows how to:
1. Define tool schemas (JSON Schema format)
2. Implement a tool handler function
3. Generate HTTP wrapper tools
4. Start a server to handle tool calls
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from schemez.tool_executor import HttpToolExecutor


if TYPE_CHECKING:
    from pydantic import BaseModel


# Example tool schemas
WEATHER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name or location"},
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                    "description": "Temperature units",
                },
            },
            "required": ["location"],
        },
    },
}

CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform basic mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "Mathematical operation to perform",
                },
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["operation", "a", "b"],
        },
    },
}


async def my_tool_handler(method_name: str, input_props: BaseModel) -> str:
    """Tool handler that implements the actual tool logic.

    This is where you implement your business logic for each tool.
    The HttpToolExecutor will call this function with validated inputs.

    Args:
        method_name: Name of the tool being called
        input_props: Pydantic model with validated input parameters

    Returns:
        String result from the tool execution
    """
    print(f"🔧 Handling tool: {method_name}")
    print(f"📝 Input: {input_props}")

    match method_name:
        case "get_weather":
            location = getattr(input_props, "location", "Unknown")
            units = getattr(input_props, "units", "celsius")

            # Mock weather data
            temp = "72°F" if units == "fahrenheit" else "22°C"

            return f"Weather in {location}: Sunny, {temp}"

        case "calculate":
            operation = getattr(input_props, "operation", "add")
            a = getattr(input_props, "a", 0)
            b = getattr(input_props, "b", 0)

            match operation:
                case "add":
                    result = a + b
                case "subtract":
                    result = a - b
                case "multiply":
                    result = a * b
                case "divide":
                    if b == 0:
                        return "Error: Division by zero"
                    result = a / b
                case _:
                    return f"Error: Unknown operation {operation}"

            return f"{a} {operation} {b} = {result}"

        case _:
            return f"Error: Unknown tool {method_name}"


async def main():
    """Main example demonstrating HttpToolExecutor usage."""
    print("🚀 HttpToolExecutor Example")
    print("=" * 50)

    # 1. Create HttpToolExecutor with schemas and handler
    print("📋 Setting up HttpToolExecutor with tool schemas...")

    # You can use schemas as dicts or load from files
    schemas: list[dict[str, Any]] = [WEATHER_SCHEMA, CALCULATOR_SCHEMA]

    executor = HttpToolExecutor(
        schemas=schemas, handler=my_tool_handler, base_url="http://localhost:8000"
    )

    # 2. Generate HTTP wrapper tools
    print("\n🔨 Generating HTTP wrapper tools...")
    tools_code = await executor.generate_tools_code()
    print(f"✅ Generated {len(tools_code)} characters of tools code")

    # 3. Get ready-to-use tool functions
    print("\n🛠️ Getting tool functions...")
    tools = await executor.get_tool_functions()
    print(f"✅ Available tools: {list(tools.keys())}")

    # 4. Save generated code to files
    print("\n💾 Saving generated code...")
    output_dir = Path(__file__).parent / "generated_output"
    saved_files = await executor.save_to_files(output_dir)
    print("✅ Saved files:")
    for file_type, file_path in saved_files.items():
        print(f"  {file_type}: {file_path}")

    # 5. Generate and start server (optional)
    print("\n🌐 Server setup...")
    try:
        server_app = await executor.generate_server_app()
        print(f"✅ Generated server: {server_app.title}")

        print("\n🎯 To start the server, run:")
        print("  uvicorn examples.http_tool_example:server --reload --port 8000")
        print("  (Note: Requires FastAPI: pip install schemez[tool_execution])")

        # For actual server startup (uncomment to start server):
        # await executor.start_server(port=8000)

    except ImportError as e:
        print(f"⚠️ FastAPI not available: {e}")
        print("  To install: pip install schemez[tool_execution]")

    print("\n" + "=" * 50)
    print("✅ Example completed!")
    print("\n📋 What was generated:")
    print("- HTTP wrapper functions for each tool")
    print("- Type-safe Pydantic input models")
    print("- FastAPI server with generic handler")
    print("- All ready for LLM orchestration!")


async def demo_tool_usage():
    """Demo how the generated tools would be used in practice."""
    print("\n🧪 Demo: Tool Usage")
    print("-" * 30)

    # This simulates how an LLM would use the generated tools
    executor = HttpToolExecutor([WEATHER_SCHEMA, CALCULATOR_SCHEMA], my_tool_handler)
    await executor.get_tool_functions()

    # Note: In practice, these would make HTTP calls to the server
    # For demo purposes, we're showing the interface
    print("Example LLM orchestration script:")
    print("""
# LLM-generated code would look like this:
from generated_tools import GetWeatherInput, CalculateInput
from generated_tools import get_weather, calculate

# Get weather for multiple cities
for city in ["San Francisco", "New York", "London"]:
    weather_input = GetWeatherInput(location=city, units="fahrenheit")
    result = await get_weather(weather_input)
    print(f"Weather: {result}")

# Do some calculations
calc_input = CalculateInput(operation="multiply", a=25, b=4)
result = await calculate(calc_input)
print(f"Calculation: {result}")
""")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())

    # Run the tool usage demo
    asyncio.run(demo_tool_usage())
