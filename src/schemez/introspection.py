"""Annotation introspection that survives lazily evaluated annotations (PEP 649).

Since Python 3.14 annotations are evaluated on access. Names that a module imports only
under `if TYPE_CHECKING:` then raise `NameError` when a signature or the type hints of a
user callable are read. The helpers here resolve whatever can be resolved and let only
the unresolvable parts degrade to `Any` (`list[Missing]` becomes `list[Any]`).
"""

import annotationlib
import builtins
import collections.abc
import functools
import inspect
import sys
import types
import typing
from typing import Annotated, Any, ForwardRef, Literal, Union, get_args, get_origin

from schemez import log


logger = log.get_logger(__name__)

_FORWARDREF = annotationlib.Format.FORWARDREF


def get_signature(func: Any) -> inspect.Signature:
    """`inspect.signature`, with unresolvable annotation names degraded to `Any`.

    Evaluates the annotations normally first, so nothing changes for callables whose
    annotations resolve. Errors unrelated to annotations (e.g. `ValueError` for
    callables without a signature) are raised as usual.
    """
    try:
        return inspect.signature(func)
    except Exception as e:  # noqa: BLE001  (re-raised below if not annotation-related)
        try:
            sig = inspect.signature(func, annotation_format=_FORWARDREF)
        except Exception:  # noqa: BLE001
            raise e from None
    params = [
        p.replace(annotation=resolve_annotation(p.annotation, func))
        if p.annotation is not inspect.Parameter.empty
        else p
        for p in sig.parameters.values()
    ]
    ret = sig.return_annotation
    if ret is not inspect.Signature.empty:
        ret = resolve_annotation(ret, func)
    # no re-validation: a default of `Parameter.empty` looks like "no default" to it
    return type(sig)(params, return_annotation=ret, __validate_parameters__=False)


def get_resolved_hints(obj: Any, *, include_extras: bool = True) -> dict[str, Any]:
    """Evaluated type hints of a function or class.

    Like `typing.get_type_hints`, but names that can't be resolved (e.g. imports under
    `TYPE_CHECKING`) don't spoil the others: only the unresolvable parts become `Any`.
    """
    try:
        return typing.get_type_hints(obj, include_extras=include_extras)
    except Exception:  # noqa: BLE001
        try:
            hints = typing.get_type_hints(obj, include_extras=include_extras, format=_FORWARDREF)
        except Exception:  # noqa: BLE001
            return {}
    return {name: resolve_annotation(ann, obj) for name, ann in hints.items()}


def resolve_annotation(annotation: Any, owner: Any = None) -> Any:
    """Resolve forward references inside an annotation, degrading missing names to `Any`.

    Args:
        annotation: An annotation as returned in `Format.FORWARDREF` (may contain
            `ForwardRef` objects, also nested, e.g. `list[ForwardRef("X")]`).
        owner: The object the annotation belongs to (function or class), used for the
            namespace when the forward reference doesn't carry its own owner.
    """
    if not _contains_forward_ref(annotation):
        return annotation
    try:
        resolved = _substitute(annotation, owner)
    except Exception:  # noqa: BLE001
        resolved = Any
    if _contains_forward_ref(resolved):
        resolved = Any
    logger.debug(
        "Annotation %r of %r has names not available at runtime, using %r",
        annotation,
        owner,
        resolved,
    )
    return resolved


def _contains_forward_ref(annotation: Any) -> bool:
    if isinstance(annotation, ForwardRef):
        return True
    if isinstance(annotation, list | tuple):
        return any(_contains_forward_ref(a) for a in annotation)
    origin = get_origin(annotation)
    if origin is Literal:
        return False
    if origin is Annotated:
        return _contains_forward_ref(annotation.__origin__)
    return any(_contains_forward_ref(a) for a in get_args(annotation))


def _substitute(annotation: Any, owner: Any) -> Any:  # noqa: PLR0911
    if isinstance(annotation, ForwardRef):
        return _evaluate_forward_ref(annotation, owner)
    if isinstance(annotation, list):
        return [_substitute(a, owner) for a in annotation]
    origin = get_origin(annotation)
    if origin is Literal:
        return annotation
    if origin is Annotated:
        inner = _substitute(annotation.__origin__, owner)
        return Annotated[inner, *annotation.__metadata__]
    args = get_args(annotation)
    if not args:
        return annotation
    new_args = tuple(_substitute(a, owner) for a in args)
    if origin is Union or origin is types.UnionType:
        return Union[new_args]  # noqa: UP007
    if origin is collections.abc.Callable:  # args are ([params...] | ..., return)
        return collections.abc.Callable[new_args]
    if isinstance(annotation, types.GenericAlias):
        return types.GenericAlias(origin, new_args)
    if hasattr(annotation, "copy_with"):
        return annotation.copy_with(new_args)
    return origin[new_args]


def _evaluate_forward_ref(ref: ForwardRef, owner: Any) -> Any:
    ref_owner = getattr(ref, "__owner__", None) or owner
    try:
        return typing.evaluate_forward_ref(ref, owner=ref_owner)
    except Exception:  # noqa: BLE001
        pass
    globalns = _globals_of(ref_owner)
    if (module := getattr(ref, "__forward_module__", None)) and module in sys.modules:
        globalns = vars(sys.modules[module])
    localns = _AnyFallback(globalns)
    if isinstance(ref_owner, type):
        localns.update(vars(ref_owner))
    # values the FORWARDREF stringifier already resolved (e.g. `Missing | Known`)
    localns.update(getattr(ref, "__extra_names__", None) or {})
    try:
        return eval(ref.__forward_arg__, dict(globalns), localns)
    except Exception:  # noqa: BLE001
        return Any


def _globals_of(obj: Any) -> dict[str, Any]:
    obj = getattr(obj, "__func__", obj)
    if isinstance(obj, functools.partial):
        obj = obj.func
    obj = inspect.unwrap(obj) if callable(obj) else obj
    if isinstance(obj, type):
        module = sys.modules.get(obj.__module__)
        return vars(module) if module else {}
    return getattr(obj, "__globals__", {})


class _AnyFallback(dict[str, Any]):
    """Local namespace for `eval`: names found nowhere else evaluate to `Any`."""

    def __init__(self, globalns: dict[str, Any]) -> None:
        super().__init__()
        self._globalns = globalns

    def __missing__(self, key: str) -> Any:
        if key in self._globalns:
            return self._globalns[key]
        return getattr(builtins, key, Any)
