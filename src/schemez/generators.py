"""Generate test data that conforms to JSON Schema specifications.

Based on code from pydantic-ais TestModel, credits to them.
"""

from __future__ import annotations

from datetime import date, timedelta
import re
import string
from typing import Any


_CHARS = string.ascii_letters + string.digits + string.punctuation


class SchemaDataGenerator:
    """Generate test data that matches a JSON schema.

    This tries to generate the minimal viable data for the schema while respecting
    all constraints and preferences defined in the schema.

    Example:
        ```python
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 3},
                "age": {"type": "integer", "minimum": 0}
            },
            "required": ["name"]
        }
        generator = SchemaDataGenerator(schema)
        data = generator.generate()  # {"name": "abc", "age": 0}
        ```
    """

    def __init__(self, schema: dict[str, Any], *, seed: int = 0):
        """Initialize with a JSON schema and optional seed.

        Args:
            schema: The JSON schema to generate data for
            seed: Seed for deterministic generation (default: 0)
        """
        self.schema = schema
        self.defs = schema.get("$defs", {})
        self.seed = seed

    def generate(self) -> Any:
        """Generate data that matches the JSON schema.

        Returns:
            Generated data that conforms to the schema
        """
        return self._gen_any(self.schema)

    def generate_minimal(self) -> Any:
        """Generate minimal valid data for the schema.

        This generates the smallest possible valid data:
        - Only required fields for objects
        - Minimum lengths for strings/arrays
        - Minimum values for numbers

        Returns:
            Minimal valid data that conforms to the schema
        """
        # Use seed=0 to ensure minimal generation
        original_seed = self.seed
        self.seed = 0
        try:
            return self._gen_any(self.schema)
        finally:
            self.seed = original_seed

    def generate_maximal(self) -> Any:
        """Generate maximal valid data for the schema.

        This generates larger valid data where possible:
        - All optional fields for objects
        - Maximum lengths for strings/arrays (capped at reasonable limits)
        - Maximum values for numbers

        Returns:
            Maximal valid data that conforms to the schema
        """
        # Use higher seed to trigger more generation
        original_seed = self.seed
        self.seed = max(100, self.seed)
        try:
            return self._gen_any_maximal(self.schema)
        finally:
            self.seed = original_seed

    def _gen_any(self, schema: dict[str, Any]) -> Any:  # noqa: PLR0911
        """Generate data for any JSON Schema type."""
        # Preference hierarchy: const > enum > examples > generated
        if const := schema.get("const"):
            return const
        if enum := schema.get("enum"):
            return enum[self.seed % len(enum)]
        if examples := schema.get("examples"):
            return examples[self.seed % len(examples)]
        if ref := schema.get("$ref"):
            key = re.sub(r"^#/\$defs/", "", ref)
            return self._gen_any(self.defs[key])
        if any_of := schema.get("anyOf"):
            return self._gen_any(any_of[self.seed % len(any_of)])

        match schema.get("type"):
            case None:
                # if there's no type or ref, fallback to string
                return _char(seed=self.seed)
            case "object":
                return self._object_gen(schema)
            case "string":
                return _str_gen(schema, seed=self.seed)
            case "integer":
                return _int_gen(schema, seed=self.seed)
            case "number":
                return float(_int_gen(schema, seed=self.seed))
            case "boolean":
                return _bool_gen(seed=self.seed)
            case "array":
                return self._array_gen(schema)
            case "null":
                return None
            case list() as ls:
                # Handle union types like ["string", "null"]
                if non_null_types := [t for t in ls if t != "null"]:
                    selected_type = non_null_types[self.seed % len(non_null_types)]
                    return self._gen_any({**schema, "type": selected_type})
                return None
            case _ as type_:
                msg = f"Unknown type: {type_}"
                raise NotImplementedError(msg)

    def _gen_any_maximal(self, schema: dict[str, Any]) -> Any:  # noqa: PLR0911
        """Generate maximal data for any JSON Schema type."""
        # Same preference hierarchy, but with maximal generation
        if const := schema.get("const"):
            return const
        if enum := schema.get("enum"):
            # For maximal, pick the last enum value
            return enum[-1]
        if examples := schema.get("examples"):
            # For maximal, pick the last example
            return examples[-1]
        if ref := schema.get("$ref"):
            key = re.sub(r"^#/\$defs/", "", ref)
            return self._gen_any_maximal(self.defs[key])
        if any_of := schema.get("anyOf"):
            # Pick the last option for maximal
            return self._gen_any_maximal(any_of[-1])

        match schema.get("type"):
            case None:
                return _char(seed=self.seed)
            case "object":
                return self._object_gen_maximal(schema)
            case "string":
                return _str_gen_maximal(schema, seed=self.seed)
            case "integer":
                return _int_gen_maximal(schema)
            case "number":
                return float(_int_gen_maximal(schema))
            case "boolean":
                return True  # Always true for maximal
            case "array":
                return self._array_gen_maximal(schema)
            case "null":
                return None
            case list() as ls:
                if non_null_types := [t for t in ls if t != "null"]:
                    # Pick the last non-null type
                    selected_type = non_null_types[-1]
                    return self._gen_any_maximal({**schema, "type": selected_type})
                return None
            case _ as type_:
                msg = f"Unknown type: {type_}"
                raise NotImplementedError(msg)

    def _object_gen(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Generate data for a JSON Schema object."""
        required = set(schema.get("required", []))
        data: dict[str, Any] = {}

        if properties := schema.get("properties"):
            for key, value in properties.items():
                if key in required:
                    data[key] = self._gen_any(value)

        if addition_props := schema.get("additionalProperties"):
            add_prop_key = "example_name"
            while add_prop_key in data:
                add_prop_key += "_"
            if addition_props is True:
                data[add_prop_key] = _char(seed=self.seed)
            else:
                data[add_prop_key] = self._gen_any(addition_props)

        return data

    def _object_gen_maximal(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Generate maximal data for a JSON Schema object."""
        data: dict[str, Any] = {}

        if properties := schema.get("properties"):
            # Generate ALL properties for maximal
            for key, value in properties.items():
                data[key] = self._gen_any_maximal(value)

        if addition_props := schema.get("additionalProperties"):
            add_prop_key = "example_name"
            while add_prop_key in data:
                add_prop_key += "_"
            if addition_props is True:
                data[add_prop_key] = _char(seed=self.seed)
            else:
                data[add_prop_key] = self._gen_any_maximal(addition_props)

        return data

    def _array_gen(self, schema: dict[str, Any]) -> list[Any]:
        """Generate an array from a JSON Schema array type."""
        data: list[Any] = []
        unique_items = schema.get("uniqueItems")

        if prefix_items := schema.get("prefixItems"):
            for item in prefix_items:
                data.append(self._gen_any(item))
                if unique_items:
                    self.seed += 1

        items_schema = schema.get("items", {})
        if (min_items := schema.get("minItems", 0)) > len(data):
            for _ in range(min_items - len(data)):
                data.append(self._gen_any(items_schema))
                if unique_items:
                    self.seed += 1
        elif items_schema:
            # if there is an `items` schema, add an item unless it would break `maxItems`
            max_items = schema.get("maxItems")
            if max_items is None or max_items > len(data):
                data.append(self._gen_any(items_schema))
                if unique_items:
                    self.seed += 1

        return data

    def _array_gen_maximal(self, schema: dict[str, Any]) -> list[Any]:
        """Generate maximal array data."""
        data: list[Any] = []
        unique_items = schema.get("uniqueItems")

        if prefix_items := schema.get("prefixItems"):
            for item in prefix_items:
                data.append(self._gen_any_maximal(item))
                if unique_items:
                    self.seed += 1

        items_schema = schema.get("items", {})
        max_items = schema.get("maxItems")

        if max_items is not None:
            # Fill to max capacity, capped at reasonable limit
            effective_max = min(max_items, 20)
            while len(data) < effective_max:
                data.append(self._gen_any_maximal(items_schema))
                if unique_items:
                    self.seed += 1
        else:
            min_items = schema.get("minItems", 1)
            # For maximal without max constraint, use reasonable default
            while len(data) < max(min_items, 5):
                data.append(self._gen_any_maximal(items_schema))
                if unique_items:
                    self.seed += 1

        return data


def _str_gen(schema: dict[str, Any], seed: int = 0) -> str:  # noqa: PLR0911
    """Generate a string from a JSON Schema string type."""
    if (min_len := schema.get("minLength")) is not None:
        return _char(seed=seed) * min_len  # type: ignore[no-any-return]
    if schema.get("maxLength") == 0:
        return ""
    if fmt := schema.get("format"):
        match fmt:
            case "date":
                return (date(2024, 1, 1) + timedelta(days=seed)).isoformat()
            case "email":
                return f"user{seed}@example.com"
            case "uri":
                return f"https://example.com/resource{seed}"
            case "uuid":
                # Generate a simple UUID-like string
                return f"12345678-1234-1234-1234-12345678{seed:04d}"

    return _char(seed=seed)


def _str_gen_maximal(schema: dict[str, Any], seed: int = 0) -> str:
    """Generate maximal string data."""
    if (max_len := schema.get("maxLength")) is not None:
        # Cap at reasonable limit for maximal generation
        effective_max = min(max_len, 100)
        return _char(seed=seed) * effective_max  # type: ignore[no-any-return]

    min_len = schema.get("minLength", 1)
    # For maximal without max constraint, use a reasonable default
    return _char(seed=seed) * max(min_len, 10)  # type: ignore[no-any-return]


def _int_gen(schema: dict[str, Any], seed: int = 0) -> int:
    """Generate an integer from a JSON Schema integer type."""
    maximum = schema.get("maximum")
    if maximum is None:
        exc_max = schema.get("exclusiveMaximum")
        if exc_max is not None:
            maximum = exc_max - 1

    minimum = schema.get("minimum")
    if minimum is None:
        exc_min = schema.get("exclusiveMinimum")
        if exc_min is not None:
            minimum = exc_min + 1

    if minimum is not None and maximum is not None:
        return minimum + seed % (maximum - minimum + 1)  # type: ignore[no-any-return]
    if minimum is not None:
        return minimum + seed  # type: ignore[no-any-return]
    if maximum is not None:
        return maximum - seed  # type: ignore[no-any-return]
    return seed


def _bool_gen(seed: int = 0) -> bool:
    """Generate a boolean from a JSON Schema boolean type."""
    return bool(seed % 2)


def _char(seed: int = 0) -> str:
    """Generate a character sequence like Excel columns (a-z, aa-az, ...)."""
    chars = len(_CHARS)
    s = ""
    rem = seed // chars
    while rem > 0:
        s += _CHARS[(rem - 1) % chars]
        rem //= chars
    s += _CHARS[seed % chars]
    return s


def _int_gen_maximal(schema: dict[str, Any]) -> int:
    """Generate maximal integer data."""
    maximum = schema.get("maximum")
    if maximum is None:
        exc_max = schema.get("exclusiveMaximum")
        if exc_max is not None:
            maximum = exc_max - 1

    if maximum is not None:
        return maximum  # type: ignore[no-any-return]

    minimum = schema.get("minimum")
    if minimum is None:
        exc_min = schema.get("exclusiveMinimum")
        if exc_min is not None:
            minimum = exc_min + 1

    if minimum is not None:
        # For maximal without max constraint, use reasonable large value
        return minimum + 1000  # type: ignore[no-any-return]
    return 1000  # Reasonable maximal default


if __name__ == "__main__":
    from llmling_agent import AgentsManifest

    data = AgentsManifest.generate_test_data(mode="maximal")
    print(data)
