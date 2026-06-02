"""Deterministic tools exposed to the agent.

Each tool lives in its own module and is re-exported here, alongside an
``ALL_TOOLS`` registry that nodes (or ``ChatModel.bind_tools``) can
iterate over.
"""

from app.agent.tools.legal_reference_validator import legal_reference_validator
from app.agent.tools.tao_calculator import LOSS_OFFSET_CAP, TAO_RATE, tao_calculator

ALL_TOOLS = [
    tao_calculator,
    legal_reference_validator,
]

__all__ = [
    "ALL_TOOLS",
    "LOSS_OFFSET_CAP",
    "TAO_RATE",
    "legal_reference_validator",
    "tao_calculator",
]
