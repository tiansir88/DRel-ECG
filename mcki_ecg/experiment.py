"""Compatibility wrapper for :mod:`drel_ecg.experiment`."""

from drel_ecg import experiment as _implementation
from drel_ecg.experiment import *  # noqa: F401,F403


def __getattr__(name):
    """Forward legacy access, including private experiment helpers."""

    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
