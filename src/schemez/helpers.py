"""Helpers for BaseModels."""

from __future__ import annotations

import os

from pydantic import BaseModel


StrPath = str | os.PathLike[str]


def merge_models[T: BaseModel](base: T, overlay: T) -> T:
    """Deep merge two Pydantic models."""
    if not isinstance(overlay, type(base)):
        msg = f"Cannot merge different types: {type(base)} and {type(overlay)}"
        raise TypeError(msg)

    merged_data = base.model_dump()
    overlay_data = overlay.model_dump(exclude_none=True)
    for field_name, field_value in overlay_data.items():
        base_value = merged_data.get(field_name)

        match (base_value, field_value):
            case (list(), list()):
                merged_data[field_name] = [
                    *base_value,
                    *(item for item in field_value if item not in base_value),
                ]
            case (dict(), dict()):
                merged_data[field_name] = base_value | field_value
            case _:
                merged_data[field_name] = field_value

    return base.__class__.model_validate(merged_data)
