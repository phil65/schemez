"""AST-based splitter that extracts pure scalar/Literal type aliases from generated modules.

Large schema-generated modules (e.g. from ``datamodel-codegen``) often mix lightweight
``Literal`` / scalar ``type`` aliases with hundreds of Pydantic ``BaseModel`` classes.
Splitting those aliases into a sibling leaf module allows downstream config packages to
import them (especially with PEP 810 ``__lazy_modules__``) without paying the import cost
of the full model module.
"""

import ast
from dataclasses import dataclass


_SCALAR_BUILTINS: frozenset[str] = frozenset({"str", "int", "float", "bool", "bytes", "None"})
_TYPING_NAMES: frozenset[str] = frozenset({
    "Annotated",
    "Any",
    "Final",
    "Literal",
    "Optional",
    "TypeAlias",
    "Union",
})
_CONTAINER_BUILTINS: frozenset[str] = frozenset({"list", "dict", "set", "frozenset", "tuple"})
_COLLECTION_ABCS: frozenset[str] = frozenset({"Mapping", "MutableMapping", "Sequence", "Set"})


@dataclass(frozen=True)
class SplitTypeAliasesResult:
    """Result of splitting scalar/literal type aliases out of a Python module."""

    types_source: str
    """Standalone source code for the extracted type-alias leaf module."""

    models_source: str
    """Rewritten source code for the original module importing the extracted names."""

    extracted_names: tuple[str, ...]
    """Names of all extracted type aliases in their original definition order."""


@dataclass(frozen=True)
class _AliasEntry:
    """A candidate top-level type alias statement and its optional docstring."""

    name: str
    value: ast.expr
    start_line: int  # 1-indexed inclusive
    end_line: int  # 1-indexed inclusive
    uses_type_alias_annotation: bool = False


def _is_constant_expr(node: ast.expr) -> bool:
    """Whether ``node`` is a literal/constant expression safe inside ``Field(...)``."""
    match node:
        case ast.Constant():
            return True
        case ast.UnaryOp(op=ast.UAdd() | ast.USub() | ast.Not(), operand=operand):
            return _is_constant_expr(operand)
        case ast.List(elts=elts) | ast.Tuple(elts=elts) | ast.Set(elts=elts):
            return all(_is_constant_expr(e) for e in elts)
        case ast.Dict(keys=keys, values=values):
            return all(k is None or _is_constant_expr(k) for k in keys) and all(
                _is_constant_expr(v) for v in values
            )
        case _:
            return False


def _is_field_call(node: ast.Call) -> bool:
    """Whether ``node`` is a ``Field(...)`` call with only constant arguments."""
    if not (isinstance(node.func, ast.Name) and node.func.id == "Field"):
        return False
    return all(_is_constant_expr(arg) for arg in node.args) and all(
        _is_constant_expr(kw.value) for kw in node.keywords
    )


def _is_extractable_expr(node: ast.expr, allowed_aliases: frozenset[str]) -> bool:
    """Whether ``node`` uses only scalars, typing constructs, ``Field``, and allowed aliases."""
    match node:
        case ast.Constant():
            ok = True
        case ast.Name(id=name):
            ok = (
                name in _SCALAR_BUILTINS
                or name in _TYPING_NAMES
                or name in _CONTAINER_BUILTINS
                or name in _COLLECTION_ABCS
                or name in allowed_aliases
            )
        case ast.BinOp(op=ast.BitOr(), left=left, right=right):
            ok = _is_extractable_expr(left, allowed_aliases) and _is_extractable_expr(
                right, allowed_aliases
            )
        case ast.Subscript(value=ast.Name(id="Literal"), slice=sub_slice):
            ok = _is_literal_slice(sub_slice)
        case ast.Subscript(value=ast.Name(id="Annotated"), slice=ast.Tuple(elts=[base, *meta])):
            ok = _is_extractable_expr(base, allowed_aliases) and all(
                _is_annotated_meta(m) for m in meta
            )
        case ast.Subscript(value=value, slice=sub_slice):
            ok = _is_extractable_expr(value, allowed_aliases) and _is_extractable_expr(
                sub_slice, allowed_aliases
            )
        case ast.Tuple(elts=elts) | ast.List(elts=elts):
            ok = all(_is_extractable_expr(e, allowed_aliases) for e in elts)
        case _:
            ok = False
    return ok


def _is_literal_slice(node: ast.expr) -> bool:
    """Whether ``node`` is a valid constant slice inside ``Literal[...]``."""
    match node:
        case ast.Tuple(elts=elts):
            return all(_is_constant_expr(e) for e in elts)
        case _:
            return _is_constant_expr(node)


def _is_annotated_meta(node: ast.expr) -> bool:
    """Whether ``node`` is safe metadata inside ``Annotated[base, ...]``."""
    match node:
        case ast.Call() as call:
            return _is_field_call(call)
        case _:
            return _is_constant_expr(node)


def _is_type_alias_annotation(annotation: ast.expr) -> bool:
    """Whether an ``AnnAssign`` annotation is ``TypeAlias`` or ``typing.TypeAlias``."""
    match annotation:
        case ast.Name(id="TypeAlias"):
            return True
        case ast.Attribute(value=ast.Name(id="typing"), attr="TypeAlias"):
            return True
        case _:
            return False


def _looks_like_plain_type_alias(value: ast.expr) -> bool:
    """Whether an unannotated assignment RHS is explicitly a typing construct."""
    match value:
        case ast.Subscript(value=ast.Name(id=name)) if name in {
            "Literal",
            "Annotated",
            "Union",
            "Optional",
        }:
            return True
        case ast.BinOp(op=ast.BitOr()):
            return True
        case _:
            return False


def _collect_candidates(body: list[ast.stmt]) -> list[_AliasEntry]:
    """Find top-level type alias statements paired with any following docstring."""
    entries: list[_AliasEntry] = []
    idx = 0
    while idx < len(body):
        stmt = body[idx]
        entry: _AliasEntry | None = None
        match stmt:
            case ast.TypeAlias(name=ast.Name(id=name), type_params=[], value=value):
                assert stmt.end_lineno is not None
                entry = _AliasEntry(
                    name=name, value=value, start_line=stmt.lineno, end_line=stmt.end_lineno
                )
            case ast.AnnAssign(
                target=ast.Name(id=name),
                annotation=annotation,
                value=value,
            ) if value is not None and _is_type_alias_annotation(annotation):
                assert stmt.end_lineno is not None
                entry = _AliasEntry(
                    name=name,
                    value=value,
                    start_line=stmt.lineno,
                    end_line=stmt.end_lineno,
                    uses_type_alias_annotation=True,
                )
            case ast.Assign(targets=[ast.Name(id=name)], value=value) if (
                _looks_like_plain_type_alias(value)
            ):
                assert stmt.end_lineno is not None
                entry = _AliasEntry(
                    name=name, value=value, start_line=stmt.lineno, end_line=stmt.end_lineno
                )
            case _:
                pass

        if entry is not None:
            # Attach immediately following string-literal docstring if present
            if idx + 1 < len(body):
                nxt = body[idx + 1]
                if (
                    isinstance(nxt, ast.Expr)
                    and isinstance(nxt.value, ast.Constant)
                    and isinstance(nxt.value.value, str)
                    and nxt.end_lineno is not None
                ):
                    entry = _AliasEntry(
                        name=entry.name,
                        value=entry.value,
                        start_line=entry.start_line,
                        end_line=nxt.end_lineno,
                        uses_type_alias_annotation=entry.uses_type_alias_annotation,
                    )
                    idx += 1
            entries.append(entry)
        idx += 1
    return entries


def _select_extractable(entries: list[_AliasEntry]) -> list[_AliasEntry]:
    """Fixed-point selection of aliases depending only on scalars and other extracted aliases."""
    # Never extract a name defined multiple times at module scope
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.name] = counts.get(entry.name, 0) + 1
    unique_entries = [e for e in entries if counts[e.name] == 1]

    selected: set[str] = set()
    changed = True
    while changed:
        changed = False
        allowed = frozenset(selected)
        for entry in unique_entries:
            if entry.name not in selected and _is_extractable_expr(entry.value, allowed):
                selected.add(entry.name)
                changed = True

    selected_entries = [e for e in unique_entries if e.name in selected]
    return _order_by_dependencies(selected_entries)


def _order_by_dependencies(entries: list[_AliasEntry]) -> list[_AliasEntry]:
    """Order aliases so dependencies come first, keeping original order as tie-breaker."""
    by_name = {e.name: e for e in entries}
    ordered: list[_AliasEntry] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(entry: _AliasEntry) -> None:
        if entry.name in visited or entry.name in visiting:
            return
        visiting.add(entry.name)
        refs = _referenced_names(entry.value)
        for dep in entries:
            if dep.name in refs and dep.name != entry.name:
                visit(by_name[dep.name])
        visiting.remove(entry.name)
        visited.add(entry.name)
        ordered.append(entry)

    for entry in entries:
        visit(entry)
    return ordered


def _referenced_names(node: ast.AST) -> set[str]:
    """Collect all ``ast.Name`` identifiers inside ``node``."""
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _import_insert_line(body: list[ast.stmt]) -> int:
    """Return 1-indexed line number immediately after the module's top-level import block."""
    last_import_end = 0
    for idx, stmt in enumerate(body):
        if (
            idx == 0
            and isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            if stmt.end_lineno is not None:
                last_import_end = stmt.end_lineno
            continue
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            if stmt.end_lineno is not None:
                last_import_end = stmt.end_lineno
            continue
        break
    return last_import_end


def _render_types_module(
    selected: list[_AliasEntry],
    lines: list[str],
    *,
    header: str | None,
) -> str:
    """Render the standalone leaf module containing all extracted type aliases."""
    used_typing: set[str] = set()
    used_abc: set[str] = set()
    uses_field = False
    for entry in selected:
        if entry.uses_type_alias_annotation:
            used_typing.add("TypeAlias")
        refs = _referenced_names(entry.value)
        used_typing |= refs & _TYPING_NAMES
        used_abc |= refs & _COLLECTION_ABCS
        if "Field" in refs:
            uses_field = True

    chunks: list[str] = []
    if header:
        chunks.append(header.rstrip("\n") + "\n")
    else:
        chunks.append('"""Extracted scalar and literal type aliases."""\n')

    import_lines: list[str] = []
    if used_abc:
        import_lines.append(f"from collections.abc import {', '.join(sorted(used_abc))}")
    if used_typing:
        import_lines.append(f"from typing import {', '.join(sorted(used_typing))}")
    if uses_field:
        if import_lines:
            import_lines.append("")
        import_lines.append("from pydantic import Field")

    if import_lines:
        chunks.append("\n" + "\n".join(import_lines) + "\n")

    for entry in selected:
        block = "".join(lines[entry.start_line - 1 : entry.end_line]).rstrip("\n")
        chunks.append("\n\n" + block + "\n")

    all_items = "\n".join(f'    "{name}",' for name in sorted(e.name for e in selected))
    chunks.append(f"\n\n__all__ = [\n{all_items}\n]\n")
    return "".join(chunks)


def _rewrite_models_module(
    source: str,
    body: list[ast.stmt],
    selected: list[_AliasEntry],
    *,
    types_module: str,
) -> str:
    """Remove extracted aliases from ``source`` and import them from ``types_module``."""
    lines = source.splitlines(keepends=True)
    removed_lines: set[int] = set()
    for entry in selected:
        removed_lines.update(range(entry.start_line, entry.end_line + 1))

    insert_after = _import_insert_line(body)
    sorted_names = sorted(e.name for e in selected)
    names_block = "\n".join(f"    {name}," for name in sorted_names)
    import_stmt = f"from {types_module} import (\n{names_block}\n)\n"

    out: list[str] = []
    inserted = False
    if insert_after == 0:
        out.append(import_stmt)
        out.append("\n")
        inserted = True

    for lineno, line in enumerate(lines, start=1):
        if lineno not in removed_lines:
            out.append(line)
        if not inserted and lineno == insert_after:
            out.append(import_stmt)
            inserted = True

    # Collapse runs of 4+ consecutive blank lines left behind by removed declarations
    result = "".join(out)
    while "\n\n\n\n" in result:
        result = result.replace("\n\n\n\n", "\n\n\n")
    return result


def split_scalar_type_aliases(
    source: str,
    *,
    types_module: str,
    header: str | None = None,
) -> SplitTypeAliasesResult:
    """Split pure scalar and ``Literal`` type aliases out of a Python module.

    Extracts top-level PEP 695 ``type`` aliases, ``TypeAlias`` annotated assignments,
    and plain ``Literal``/``Annotated``/union aliases that reference only built-in
    scalars, typing primitives, ``pydantic.Field(...)``, and other extracted aliases.
    Each extracted alias keeps any immediately following docstring.

    Args:
        source: Source code of the module to split (e.g. ``datamodel-codegen`` output).
        types_module: Dotted import path of the generated leaf module used in the
            rewritten import statement (e.g. ``"codexed.models.v2_types"``).
        header: Optional module header/docstring for the generated types module.

    Returns:
        A :class:`SplitTypeAliasesResult` containing the leaf types module source,
        the rewritten models module source, and the tuple of extracted alias names.
    """
    tree = ast.parse(source)
    candidates = _collect_candidates(tree.body)
    selected = _select_extractable(candidates)
    if not selected:
        return SplitTypeAliasesResult(
            types_source="",
            models_source=source,
            extracted_names=(),
        )

    lines = source.splitlines(keepends=True)
    types_source = _render_types_module(selected, lines, header=header)
    models_source = _rewrite_models_module(
        source,
        tree.body,
        selected,
        types_module=types_module,
    )
    return SplitTypeAliasesResult(
        types_source=types_source,
        models_source=models_source,
        extracted_names=tuple(e.name for e in selected),
    )
