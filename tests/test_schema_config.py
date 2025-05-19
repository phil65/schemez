from __future__ import annotations

from pydantic import ValidationError
import pytest
import yamling

from schemez import InlineSchemaDef


YAML_CONFIG = """
fields:
  coverage:
    type: float
    description: "Test coverage percentage"
    ge: 80.0
    le: 100.0
  issues:
    type: int
    description: "Number of found issues"
    ge: 0
"""


def test_response_validation():
    data = yamling.load_yaml(YAML_CONFIG)
    definition = InlineSchemaDef.model_validate(data)
    quality_cls = definition.get_schema()

    # Valid data should work
    report = quality_cls(coverage=85.5, issues=3)
    assert report.coverage == 85.5  # noqa: PLR2004  # type: ignore
    assert report.issues == 3  # noqa: PLR2004  # type: ignore

    # Invalid data should fail validation
    with pytest.raises(ValidationError):
        quality_cls(coverage=50.0, issues=-1)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-vv"])
