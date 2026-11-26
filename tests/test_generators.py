"""Test data generation from JSON schema."""

import pytest

from schemez.generators import SchemaDataGenerator


@pytest.mark.parametrize("const", [0, False, "", None])
def test_falsy_const_is_used(const: object) -> None:
    generator = SchemaDataGenerator({"const": const})
    assert generator.generate() == const
    assert generator.generate_maximal() == const
