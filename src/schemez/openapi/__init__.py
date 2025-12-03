"""OpenAPI toolset implementation with HTTP-aware reference resolution.

This package provides tools for loading OpenAPI specs and creating callable tools
from their operations. It includes a custom resolver that handles external HTTP
references in multi-file OpenAPI specifications.
"""

from __future__ import annotations

from schemez.openapi.callable_factory import OpenAPICallableFactory
from schemez.openapi.loader import load_openapi_spec, parse_operations
from schemez.openapi.resolver import OpenAPIResolver, resolve_openapi_refs


__all__ = [
    "OpenAPICallableFactory",
    "OpenAPIResolver",
    "load_openapi_spec",
    "parse_operations",
    "resolve_openapi_refs",
]
