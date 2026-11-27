"""Tests for AST-based scalar/Literal type alias splitting."""

import sys
import types

from pydantic import BaseModel

from schemez import SplitTypeAliasesResult, split_scalar_type_aliases


def _exec_split(
    result: SplitTypeAliasesResult,
    *,
    pkg_name: str,
    types_mod_name: str,
    models_mod_name: str,
) -> tuple[types.ModuleType, types.ModuleType]:
    """Execute both generated modules in isolated temporary modules."""
    pkg = types.ModuleType(pkg_name)
    types_mod = types.ModuleType(types_mod_name)
    models_mod = types.ModuleType(models_mod_name)
    sys.modules[pkg_name] = pkg
    sys.modules[types_mod_name] = types_mod
    sys.modules[models_mod_name] = models_mod
    try:
        exec(compile(result.types_source, types_mod_name, "exec"), types_mod.__dict__)
        exec(compile(result.models_source, models_mod_name, "exec"), models_mod.__dict__)
    finally:
        sys.modules.pop(models_mod_name, None)
        sys.modules.pop(types_mod_name, None)
        sys.modules.pop(pkg_name, None)
    return types_mod, models_mod


def test_split_pep695_literals_scalars_and_annotated_field() -> None:
    source = '''\
"""Generated models."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


type AbsolutePath = str
"""An absolute path."""


type Personality = Literal["none", "friendly", "pragmatic"]


type ReasoningEffort = Annotated[str, Field(..., min_length=1)]


type ModeOrEffort = Personality | ReasoningEffort


class Item(BaseModel):
    path: AbsolutePath
    personality: Personality = "none"
    effort: ReasoningEffort = "low"


type ItemOrStr = Item | str
'''

    result = split_scalar_type_aliases(source, types_module="_tmp_split_pkg.leaf_types")

    assert result.extracted_names == (
        "AbsolutePath",
        "Personality",
        "ReasoningEffort",
        "ModeOrEffort",
    )
    assert "ItemOrStr" not in result.extracted_names
    assert "from pydantic import Field" in result.types_source
    assert "from typing import Annotated, Literal" in result.types_source
    assert '"""An absolute path."""' in result.types_source
    assert '"""An absolute path."""' not in result.models_source
    assert "from _tmp_split_pkg.leaf_types import (" in result.models_source
    assert "type ItemOrStr = Item | str" in result.models_source

    types_mod, models_mod = _exec_split(
        result,
        pkg_name="_tmp_split_pkg",
        types_mod_name="_tmp_split_pkg.leaf_types",
        models_mod_name="_tmp_split_pkg.models",
    )
    assert types_mod.Personality is models_mod.Personality
    assert types_mod.ReasoningEffort is models_mod.ReasoningEffort
    item = models_mod.Item(path="/tmp/a", personality="friendly", effort="high")
    assert isinstance(item, BaseModel)
    assert item.personality == "friendly"


def test_split_orders_dependencies_and_handles_plain_and_typealias() -> None:
    source = """\
from typing import Literal, TypeAlias

Combined: TypeAlias = First | Second
Second = Literal["b", "c"]
First: TypeAlias = Literal["a"]
"""
    result = split_scalar_type_aliases(source, types_module="_tmp_order_pkg.types")
    assert result.extracted_names == ("Second", "First", "Combined")
    # Plain Literal/TypeAlias aliases without Field should not import pydantic
    assert "pydantic" not in result.types_source

    types_mod, models_mod = _exec_split(
        result,
        pkg_name="_tmp_order_pkg",
        types_mod_name="_tmp_order_pkg.types",
        models_mod_name="_tmp_order_pkg.models",
    )
    assert types_mod.Combined == models_mod.Combined


def test_split_returns_unchanged_when_no_extractable_aliases() -> None:
    source = """\
from pydantic import BaseModel

class Foo(BaseModel):
    x: int

type FooList = list[Foo]
"""
    result = split_scalar_type_aliases(source, types_module="pkg.types")
    assert result.extracted_names == ()
    assert result.types_source == ""
    assert result.models_source == source


def test_split_preserves_custom_header_and_multiline_literal() -> None:
    source = """\
from typing import Literal

type AuthMode = Literal[
    "api_key",
    "oauth",
    "none",
]
"""
    result = split_scalar_type_aliases(
        source,
        types_module="pkg.v2_types",
        header='# auto-generated\n"""Protocol literal types."""',
    )
    assert result.extracted_names == ("AuthMode",)
    assert result.types_source.startswith('# auto-generated\n"""Protocol literal types."""\n')
    assert '"api_key"' in result.types_source
    assert "__all__" in result.types_source
