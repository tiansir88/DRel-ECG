"""Compatibility namespace for the former MCKI-ECG package name.

New code should import :mod:`drel_ecg`. This namespace remains available so
that archived experiment scripts and serialized checkpoints continue to load.
"""

from drel_ecg import *  # noqa: F401,F403
from drel_ecg import __all__
