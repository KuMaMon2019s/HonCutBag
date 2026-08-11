"""LangGraph orchestration foundation for HonCut.

The existing workflow remains in ``phases.pipeline_core`` until each migration
slice has passed compatibility tests.
"""

from .context import initial_state_from_config
from .state import HonCutState

__all__ = ["HonCutState", "initial_state_from_config"]
