"""Helper functions for the tool executor."""

from __future__ import annotations

import logging
import time

from schemez.helpers import model_to_python_code
from schemez.schema import Schema


logger = logging.getLogger(__name__)


async def generate_input_model(schema_dict: dict) -> tuple[str, str]:
    """Generate input model code from schema."""
    start_time = time.time()
    logger.debug("Generating input model for %s", schema_dict["name"])

    class TempInputSchema(Schema):
        @classmethod
        def model_json_schema(cls, **kwargs):
            return schema_dict["parameters"]

    input_class_name = (
        f"{''.join(word.title() for word in schema_dict['name'].split('_'))}Input"
    )

    input_code = await model_to_python_code(
        TempInputSchema,
        class_name=input_class_name,
    )

    elapsed = time.time() - start_time
    logger.debug("Generated input model for %s in %.2fs", schema_dict["name"], elapsed)

    return input_code, input_class_name


def clean_generated_code(code: str) -> str:
    """Clean generated code by removing future imports and headers."""
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
            ):
                skip_until_class = False
                cleaned_lines.append(line)
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)
