"""OpenAPI reference resolver with HTTP-aware resolution."""

from __future__ import annotations

from typing import Any, Self
from urllib.parse import urljoin

import httpx2
import yamling

from schemez.log import get_logger


logger = get_logger(__name__)


class OpenAPIResolver:
    """Resolves $ref references in OpenAPI specs, including external HTTP refs."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        """Initialize resolver.

        Args:
            base_url: Base URL of the OpenAPI spec (used for relative refs).
            timeout: HTTP request timeout in seconds.
        """
        self.base_url = base_url
        self.timeout = timeout
        self._cache: dict[str, dict[str, Any]] = {}
        self._client: httpx2.Client | None = None
        self._resolving: set[str] = set()  # Track refs being resolved to detect cycles
        # Resolved targets that didn't run into a cycle (shared, not copied)
        self._resolved: dict[str, Any] = {}
        self._cycles = 0

    @property
    def client(self) -> httpx2.Client:
        if self._client is None:
            self._client = httpx2.Client(timeout=self.timeout, follow_redirects=True)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def resolve(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Fully resolve all $ref in the spec.

        Internal refs (`#/...`) resolve against the document they appear in (the spec,
        or an external document), recursively; a cyclic ref, or one whose target can't
        be found or fetched, stays a `$ref`.

        Args:
            spec: The OpenAPI spec dictionary.

        Returns:
            A new dictionary with all refs resolved.
        """
        self._resolving.clear()
        self._resolved.clear()
        self._cache[self.base_url] = spec
        return self._resolve_value(spec, self.base_url)

    def _resolve_value(self, value: Any, current_url: str) -> Any:
        """Recursively resolve refs in a value."""
        match value:
            case dict() if isinstance(value.get("$ref"), str):
                resolved = self._resolve_ref(value["$ref"], current_url)
                siblings = {k: v for k, v in value.items() if k != "$ref"}
                if siblings and isinstance(resolved, dict) and "$ref" not in resolved:
                    resolved = {**resolved, **self._resolve_value(siblings, current_url)}
                return resolved
            case dict():
                return {k: self._resolve_value(v, current_url) for k, v in value.items()}
            case list():
                return [self._resolve_value(item, current_url) for item in value]
            case _:
                return value

    def _resolve_ref(self, ref: str, current_url: str) -> Any:
        """Resolve a single $ref.

        Args:
            ref: The reference string (e.g., '../parameters/foo.yml#/codes').
            current_url: URL of the document containing this ref.

        Returns:
            The resolved schema/object.
        """
        # Split ref into URL part and JSON pointer
        url_part, pointer = ref.split("#", 1) if "#" in ref else (ref, "")
        # Same document for internal refs like #/components/schemas/Foo
        abs_url = urljoin(current_url, url_part) if url_part else current_url
        full_ref = f"{abs_url}#{pointer}"
        if full_ref in self._resolved:
            return self._resolved[full_ref]
        if full_ref in self._resolving:  # Detect circular references
            logger.debug("Circular reference detected: %s", full_ref)
            self._cycles += 1
            return {"$ref": ref}  # Keep original ref to break cycle

        doc = self._fetch_document(abs_url)
        if doc is None:
            return {"$ref": ref}
        result = _navigate_pointer(doc, pointer) if pointer else doc
        if result is _MISSING:
            logger.warning("Unresolvable reference %r in %s", ref, current_url)
            return {"$ref": ref}
        self._resolving.add(full_ref)
        cycles = self._cycles
        try:
            # Recursively resolve any refs in the target, relative to its document
            resolved = self._resolve_value(result, abs_url)
        finally:
            self._resolving.discard(full_ref)
        if self._cycles == cycles:
            self._resolved[full_ref] = resolved
        return resolved

    def _fetch_document(self, url: str) -> dict[str, Any] | None:
        """Fetch and cache an external document (HTTP or any UPath); None on failure."""
        if url in self._cache:
            return self._cache[url]

        logger.debug("Fetching external ref: %s", url)
        try:
            if url.startswith(("http://", "https://")):
                response = self.client.get(url)
                response.raise_for_status()
                content = response.text
            else:
                from upath import UPath

                content = UPath(url).read_text(encoding="utf-8")
            doc = yamling.load_yaml(content, verify_type=dict)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to fetch %s: %s", url, e)
            return None
        self._cache[url] = doc
        return doc


_MISSING: Any = object()


def _navigate_pointer(doc: dict[str, Any], pointer: str) -> Any:
    """Navigate a JSON pointer within a document.

    Args:
        doc: The document to navigate.
        pointer: JSON pointer like '/codes' or '/components/schemas/Foo'.

    Returns:
        The value at the pointer location, or `_MISSING`.
    """
    if not pointer or pointer == "/":
        return doc

    # Remove leading slash and split
    parts = pointer.lstrip("/").split("/")
    current: Any = doc

    for part in parts:
        # Handle JSON pointer escaping
        cleaned = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if cleaned not in current:
                return _MISSING
            current = current[cleaned]
        elif isinstance(current, list):
            try:
                current = current[int(cleaned)]
            except ValueError, IndexError:
                return _MISSING
        else:
            return _MISSING

    return current


def resolve_openapi_refs(
    spec: dict[str, Any],
    base_url: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Resolve all $ref in an OpenAPI spec.

    Args:
        spec: The OpenAPI spec dictionary.
        base_url: Base URL for resolving relative references.
        timeout: HTTP timeout for fetching external refs.

    Returns:
        Spec with all resolvable refs inlined (cyclic ones stay `$ref`).
    """
    with OpenAPIResolver(base_url, timeout) as resolver:
        return resolver.resolve(spec)
