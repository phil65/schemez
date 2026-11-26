import pytest

from schemez.jsonschema_utils import (
    SchemaError,
    const_value,
    create_resolver,
    is_null_schema,
    merge_schemas,
    normalize_schema,
    schema_types,
)


def normalized(root: dict, part: dict | bool | None = None) -> dict:
    resolver = create_resolver(root)
    schema, _ = normalize_schema(root if part is None else part, resolver)
    return schema


def test_ref_to_defs_keeps_siblings_first() -> None:
    root = {
        "$defs": {"Pet": {"type": "object", "title": "Pet", "description": "from the def"}},
        "properties": {"pet": {"$ref": "#/$defs/Pet", "description": "from the field"}},
    }
    pet = normalized(root, root["properties"]["pet"])
    assert pet["type"] == "object"
    assert pet["title"] == "Pet"
    assert pet["description"] == "from the field"
    assert "$ref" not in pet


def test_json_pointer_anchor_and_id_scopes() -> None:
    root = {
        "$id": "https://example.com/root.json",
        "$defs": {
            "a/b": {"type": "string"},
            "named": {"$anchor": "named", "type": "integer"},
            "scoped": {
                "$id": "https://example.com/nested/scoped.json",
                "$defs": {"inner": {"type": "boolean"}},
                "properties": {"flag": {"$ref": "#/$defs/inner"}},
            },
        },
    }
    assert normalized(root, {"$ref": "#/$defs/a~1b"})["type"] == "string"
    assert normalized(root, {"$ref": "#named"})["type"] == "integer"
    resolver = create_resolver(root)
    scoped, scoped_resolver = normalize_schema({"$ref": "#/$defs/scoped"}, resolver)
    flag, _ = normalize_schema(scoped["properties"]["flag"], scoped_resolver)
    assert flag["type"] == "boolean"  # resolved against the nested $id, not the root


def test_all_of_merges_properties_and_required() -> None:
    schema = normalized({
        "allOf": [
            {"properties": {"a": {"type": "string"}}, "required": ["a"]},
            {"properties": {"b": {"type": "integer"}}, "required": ["b", "a"]},
        ],
        "title": "Both",
    })
    assert set(schema["properties"]) == {"a", "b"}
    assert schema["required"] == ["a", "b"]
    assert schema["title"] == "Both"


def test_recursive_schemas_resolve_one_level_at_a_time() -> None:
    root = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}},
            }
        },
        "$ref": "#/$defs/Node",
    }
    resolver = create_resolver(root)
    node, resolver = normalize_schema(root, resolver)
    child, _ = normalize_schema(node["properties"]["children"]["items"], resolver)
    assert child["properties"]["children"]["items"] == {"$ref": "#/$defs/Node"}


def test_errors() -> None:
    root = {"$defs": {"a": {"$ref": "#/$defs/b"}, "b": {"$ref": "#/$defs/a"}}}
    with pytest.raises(SchemaError, match="cycle"):
        normalized(root, {"$ref": "#/$defs/a"})
    with pytest.raises(SchemaError, match="resolve"):
        normalized({}, {"$ref": "#/$defs/missing"})
    with pytest.raises(SchemaError, match="false"):
        normalized({}, False)


def test_merge_combines_shared_properties() -> None:
    merged = merge_schemas(
        {"properties": {"x": {"type": "string"}}}, {"properties": {"x": {"maxLength": 3}}}
    )
    assert merged["properties"]["x"] == {"allOf": [{"type": "string"}, {"maxLength": 3}]}


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"type": "string"}, (["string"], False)),
        ({"type": ["string", "null"]}, (["string"], True)),
        ({"type": "integer", "nullable": True}, (["integer"], True)),
        ({}, ([], False)),
    ],
)
def test_schema_types(schema: dict, expected: tuple[list[str], bool]) -> None:
    assert schema_types(schema) == expected


def test_null_and_const_detection() -> None:
    assert is_null_schema({"type": "null"})
    assert is_null_schema({"const": None})
    assert is_null_schema({"enum": [None]})
    assert not is_null_schema({"type": ["string", "null"]})
    assert const_value({"const": "a"}) == (True, "a")
    assert const_value({"enum": [3]}) == (True, 3)
    assert const_value({"enum": [1, 2]}) == (False, None)
