"""OpenAPI $ref resolution: local files, nested and external-internal refs."""

import json
from typing import TYPE_CHECKING

from schemez.openapi import load_openapi_spec, resolve_openapi_refs


if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _response_schema(spec: dict) -> dict:
    op = spec["paths"]["/pets"]["get"]
    return op["responses"]["200"]["content"]["application/json"]["schema"]


def test_relative_ref_of_local_spec(tmp_path: Path) -> None:
    _write(
        tmp_path / "schemas.json",
        {
            "Pet": {"type": "object", "properties": {"owner": {"$ref": "#/User"}}},
            "User": {"type": "object", "properties": {"name": {"type": "string"}}},
        },
    )
    api = _write(
        tmp_path / "api.json",
        {
            "openapi": "3.1.0",
            "paths": {
                "/pets": {
                    "get": {
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {"schema": {"$ref": "./schemas.json#/Pet"}}
                                }
                            }
                        }
                    }
                }
            },
        },
    )
    schema = _response_schema(load_openapi_spec(api))
    assert schema["type"] == "object"
    # "#/User" inside schemas.json refers to schemas.json, not to the root spec
    assert schema["properties"]["owner"]["properties"]["name"] == {"type": "string"}


def test_nested_internal_refs_resolve_fully() -> None:
    spec = {
        "components": {
            "schemas": {
                "Pet": {"properties": {"owner": {"$ref": "#/components/schemas/User"}}},
                "User": {"properties": {"name": {"type": "string"}}},
            }
        },
        "x": {"$ref": "#/components/schemas/Pet"},
    }
    resolved = resolve_openapi_refs(spec, "http://example.invalid/api.json")
    assert resolved["x"]["properties"]["owner"] == {"properties": {"name": {"type": "string"}}}


def test_cyclic_internal_ref_terminates() -> None:
    spec = {
        "components": {
            "schemas": {"Node": {"properties": {"next": {"$ref": "#/components/schemas/Node"}}}}
        },
        "x": {"$ref": "#/components/schemas/Node"},
    }
    resolved = resolve_openapi_refs(spec, "http://example.invalid/api.json")
    assert resolved["x"]["properties"]["next"] == {"$ref": "#/components/schemas/Node"}


def test_missing_external_document_keeps_ref(tmp_path: Path) -> None:
    spec = {"x": {"$ref": "./missing.json#/Pet"}}
    resolved = resolve_openapi_refs(spec, (tmp_path / "api.json").as_uri())
    assert resolved["x"] == {"$ref": "./missing.json#/Pet"}


def test_shared_ref_resolved_once() -> None:
    spec = {
        "components": {"schemas": {"Big": {"properties": {"a": {"type": "string"}}}}},
        "x": [{"$ref": "#/components/schemas/Big"} for _ in range(3)],
    }
    resolved = resolve_openapi_refs(spec, "http://example.invalid/api.json")
    assert resolved["x"][0] is resolved["x"][1]
    assert resolved["x"][0] == {"properties": {"a": {"type": "string"}}}
