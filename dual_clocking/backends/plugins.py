"""Plugin helpers for dynamically loading dual clocking backends."""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points
from typing import Dict, Iterator, Mapping, Type

from .base import DualClockingBackend
from .ion import TrappedIonBackend
from .nmr import NMRBackend
from .superconducting import SuperconductingBackend
from .telecom import TelecomPhotonicsBackend

ENTRY_POINT_GROUP = "dual_clocking.backends"

BUILTIN_BACKENDS: Dict[str, Type[DualClockingBackend]] = {
    "superconducting": SuperconductingBackend,
    "trapped-ion": TrappedIonBackend,
    "nmr": NMRBackend,
    "telecom-photonics": TelecomPhotonicsBackend,
}


def _iter_entry_points() -> Iterator[EntryPoint]:
    """Yield registered entry points for dual clocking backends."""

    eps = entry_points()
    if hasattr(eps, "select"):
        yield from eps.select(group=ENTRY_POINT_GROUP)
        return

    group = getattr(eps, "get", lambda _name, default: default)(ENTRY_POINT_GROUP, [])
    for entry_point in group:
        yield entry_point


def available_backends() -> Mapping[str, Type[DualClockingBackend]]:
    """Return a mapping of known backend names to their classes."""

    discovered: Dict[str, Type[DualClockingBackend]] = dict(BUILTIN_BACKENDS)
    for entry_point in _iter_entry_points():
        try:
            backend_cls = entry_point.load()
        except Exception:  # pragma: no cover - defensive logging hook
            # Skip broken entry points to avoid failing discovery completely.
            continue
        if isinstance(backend_cls, type) and issubclass(backend_cls, DualClockingBackend):
            discovered.setdefault(entry_point.name, backend_cls)
    return discovered


def get_backend_class(name: str) -> Type[DualClockingBackend]:
    """Return the backend class registered under ``name``.

    The lookup checks built-in backends first, followed by plug-ins exposed via
    the :mod:`importlib.metadata` entry-point group ``dual_clocking.backends``.
    """

    if name in BUILTIN_BACKENDS:
        return BUILTIN_BACKENDS[name]

    for entry_point in _iter_entry_points():
        if entry_point.name != name:
            continue
        backend_cls = entry_point.load()
        if not isinstance(backend_cls, type) or not issubclass(backend_cls, DualClockingBackend):
            raise TypeError(
                f"Entry point '{name}' did not provide a DualClockingBackend subclass"
            )
        return backend_cls

    raise KeyError(f"No dual clocking backend named '{name}' was found")


__all__ = [
    "BUILTIN_BACKENDS",
    "ENTRY_POINT_GROUP",
    "available_backends",
    "get_backend_class",
]
