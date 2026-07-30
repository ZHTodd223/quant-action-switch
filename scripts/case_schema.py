"""Deprecated compatibility wrapper for quant_action_switch.schemas.case_schema."""
try:
    from ._compat import ensure_src
except ImportError:  # Direct historical execution from scripts/.
    from _compat import ensure_src
ensure_src()
from quant_action_switch.schemas.case_schema import *  # noqa: F401,F403
