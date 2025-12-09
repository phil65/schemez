"""OpenAPI spec loading and parsing utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anyenv
import httpx
from upath import UPath
import yamling

from schemez.openapi.resolver import resolve_openapi_refs


if TYPE_CHECKING:
    from upath.types import JoinablePathLike


def load_openapi_spec(url_or_path: JoinablePathLike, timeout: float = 30.0) -> dict[str, Any]:
    """Load and fully dereference an OpenAPI spec.

    Args:
        url_or_path: URL or path to the OpenAPI spec.
        timeout: HTTP timeout for fetching.

    Returns:
        Fully dereferenced OpenAPI spec.
    """
    if (path_str := str(url_or_path)).startswith(("http://", "https://")):
        response = httpx.get(path_str, follow_redirects=True, timeout=timeout)
        response.raise_for_status()
        content = response.text
        base_url = path_str
    else:
        content = UPath(url_or_path).read_text(encoding="utf-8")
        base_url = UPath(url_or_path).resolve().as_uri()

    # Try to detect format and parse accordingly
    try:
        # First try JSON (faster to fail if not JSON)
        spec = anyenv.load_json(content)
    except anyenv.JsonLoadError:
        # Fall back to YAML
        spec = yamling.load_yaml(content, verify_type=dict)

    return resolve_openapi_refs(spec, base_url, timeout)


def parse_operations(paths: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Parse OpenAPI paths into operation configurations."""
    operations = {}
    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue

        for method, operation in path_item.items():
            if method not in {"get", "post", "put", "delete", "patch"}:
                continue
            if not isinstance(operation, dict):
                continue

            # Generate operation ID if not provided
            op_id = operation.get("operationId")
            if not op_id:
                op_id = f"{method}_{path.replace('/', '_').strip('_')}"

            # Collect all parameters
            params = operation.get("parameters", [])
            # Filter out any unresolved refs
            params = [p for p in params if isinstance(p, dict) and "$ref" not in p]

            # Handle request body
            if (
                (request_body := operation.get("requestBody"))
                and isinstance(request_body, dict)
                and (content := request_body.get("content", {}))
                and (json_schema := content.get("application/json", {}).get("schema"))
                and isinstance(json_schema, dict)
                and (properties := json_schema.get("properties", {}))
            ):
                for name, schema in properties.items():
                    if isinstance(schema, dict) and "$ref" not in schema:
                        params.append({
                            "name": name,
                            "in": "body",
                            "required": name in json_schema.get("required", []),
                            "schema": schema,
                            "description": schema.get("description", ""),
                        })

            operations[op_id] = {
                "method": method,
                "path": path,
                "description": operation.get("description", ""),
                "parameters": params,
                "responses": operation.get("responses", {}),
            }

    return operations
