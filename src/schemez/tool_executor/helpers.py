"""Helper functions for the tool executor."""

from __future__ import annotations

import time
from typing import Any

from schemez import log
from schemez.helpers import model_to_python_code


logger = log.get_logger(__name__)


async def generate_input_model(schema_dict: dict[str, Any]) -> tuple[str, str]:
    """Generate input model code from schema."""
    start_time = time.time()
    logger.debug("Generating input model for %s", schema_dict["name"])

    words = [word.title() for word in schema_dict["name"].split("_")]
    cls_name = f"{''.join(words)}Input"
    code = model_to_python_code(schema_dict["parameters"], class_name=cls_name)
    elapsed = time.time() - start_time
    logger.debug("Generated input model for %s in %.2fs", schema_dict["name"], elapsed)
    return code, cls_name


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
