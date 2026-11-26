"""Helpers for walking JSON Schemas: references, ``allOf``, types, constants.

Built on :mod:`referencing`, so ``$ref`` resolves the way JSON Schema 2020-12 says
(``$defs``, JSON pointers, ``$anchor``, ``$id`` scopes). Resolution is one level at
a time (:func:`normalize_schema`): a caller walking a schema resolves what it
reaches, so recursive schemas (a tree whose nodes contain trees) are fine.

Example:
    >>> resolver = create_resolver(schema)
    >>> part, resolver = normalize_schema(schema["properties"]["child"], resolver)
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from referencing import Registry, Resource
import referencing.exceptions
from referencing.jsonschema import DRAFT202012


if TYPE_CHECKING:
    from referencing._core import Resolver


type JsonSchema = Mapping[str, Any] | bool
"""A JSON Schema: an object, or ``true`` / ``false``."""

ROOT_URI = "urn:schemez:root"
"""Base URI for a schema without ``$id``."""


class SchemaError(ValueError):
    """A schema can't be used as asked (e.g. an unresolvable ``$ref``)."""


def create_resolver(schema: JsonSchema) -> Resolver[Any]:
    """A resolver for the ``$ref`` s of ``schema`` (its ``$id`` or :data:`ROOT_URI` as base)."""
    uri = str(schema.get("$id", ROOT_URI)) if isinstance(schema, Mapping) else ROOT_URI
    resource = Resource.from_contents(schema, default_specification=DRAFT202012)
    registry: Registry[Any] = Registry().with_resource(uri=uri, resource=resource)
    return registry.resolver(base_uri=uri)


def normalize_schema(
    schema: JsonSchema, resolver: Resolver[Any], *, _seen: frozenset[str] = frozenset()
) -> tuple[dict[str, Any], Resolver[Any]]:
    """``schema`` with its ``$ref`` resolved and its ``allOf`` merged, one level deep.

    Keywords next to a ``$ref`` take precedence over the referenced schema's (2020-12
    allows siblings; pydantic puts a field's description there). Nested schemas
    (properties, items, ...) stay as they are: normalize them when you reach them,
    with the resolver returned here (it follows ``$id`` scopes).

    Raises:
        SchemaError: For ``false``, an unresolvable ``$ref``, or a ``$ref`` cycle that
            never reaches an actual schema.
    """
    if schema is True:
        return {}, resolver
    if schema is False:
        msg = "The schema 'false' accepts no value"
        raise SchemaError(msg)
    if "$id" in schema:
        resolver = resolver.in_subresource(DRAFT202012.create_resource(schema))
    result = dict(schema)
    for keyword in ("$ref", "$dynamicRef"):
        if (ref := result.pop(keyword, None)) is None:
            continue
        if ref in _seen:
            msg = f"$ref cycle without any schema in between: {ref}"
            raise SchemaError(msg)
        try:
            resolved = resolver.lookup(ref)
        except referencing.exceptions.Unresolvable as exc:
            msg = f"Can't resolve $ref {ref!r}: {exc}"
            raise SchemaError(msg) from exc
        target, resolver = normalize_schema(
            resolved.contents, resolved.resolver, _seen=_seen | {ref}
        )
        result = merge_schemas(result, target)
    if all_of := result.pop("allOf", None):
        for part in all_of:
            part_schema, _ = normalize_schema(part, resolver, _seen=_seen)
            result = merge_schemas(result, part_schema)
    return result, resolver


def merge_schemas(base: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` with ``other``'s keywords added (an ``allOf`` of the two, as one schema).

    ``base`` wins where both set a keyword, except ``properties`` (a property in both
    becomes an ``allOf`` of the two) and ``required`` (combined).
    """
    merged = dict(base)
    for key, value in other.items():
        if key not in merged:
            merged[key] = value
        elif key == "properties":
            properties = dict(merged[key])
            for name, prop in value.items():
                properties[name] = (
                    {"allOf": [properties[name], prop]} if name in properties else prop
                )
            merged[key] = properties
        elif key == "required":
            merged[key] = list(dict.fromkeys([*merged[key], *value]))
    return merged


def schema_types(schema: Mapping[str, Any]) -> tuple[list[str], bool]:
    """The non-null types a schema names, and whether it allows ``null``.

    ``null`` counts when listed in ``type`` or when OpenAPI 3.0's ``nullable`` is set.
    """
    match schema.get("type"):
        case str(single):
            declared = [single]
        case list(several):
            declared = [str(kind) for kind in several]
        case _:
            declared = []
    nullable = "null" in declared or bool(schema.get("nullable", False))
    return [kind for kind in declared if kind != "null"], nullable


def is_null_schema(schema: Mapping[str, Any]) -> bool:
    """The schema allows nothing but ``null`` (``type: "null"``, ``const: null``, ...)."""
    if "const" in schema:
        return schema["const"] is None
    if isinstance(enum := schema.get("enum"), list):
        return enum == [None]
    return schema.get("type") in ("null", ["null"])


def const_value(schema: Mapping[str, Any]) -> tuple[bool, Any]:
    """Whether the schema allows exactly one value (``const``, one-item ``enum``), and which."""
    if "const" in schema:
        return True, schema["const"]
    if isinstance(enum := schema.get("enum"), list) and len(enum) == 1:
        return True, enum[0]
    return False, None
