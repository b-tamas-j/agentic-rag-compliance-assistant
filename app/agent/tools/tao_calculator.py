"""TAO calculator tool — Hungarian corporate income tax (Tao. tv. §19)."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Standard Hungarian corporate income tax rate (Tao. tv. §19 (1)).
TAO_RATE: float = 0.09

# A carried-forward loss can only offset up to 50% of the tax base
# (Tao. tv. §17 (2)).
LOSS_OFFSET_CAP: float = 0.50


@tool("tao_calculator", return_direct=False)
def tao_calculator(
    tax_base_huf: float,
    loss_carried_forward_huf: float = 0.0,
) -> dict[str, Any]:
    """Compute Hungarian corporate income tax (TAO) on a given tax base.

    Use this whenever the user asks for a numeric tax amount or wants to
    know the effect of carrying forward a prior-year loss. The standard
    rate is 9%. A carried-forward loss reduces the tax base, but only up
    to 50% of it (Tao. tv. §17).

    Args:
        tax_base_huf: Pre-tax positive tax base in HUF (``adóalap``).
        loss_carried_forward_huf: Optional prior-year carried-forward
            loss in HUF. Capped at 50% of the tax base.

    Returns:
        Dict with the adjusted base, the computed tax, the effective
        rate, the loss actually applied and a short human-readable
        explanation (Hungarian) suitable for chat output.
    """
    logger.debug(
        "tao_calculator called: tax_base=%s, loss=%s", tax_base_huf, loss_carried_forward_huf
    )
    if tax_base_huf < 0:
        raise ValueError("tax_base_huf must be non-negative")
    if loss_carried_forward_huf < 0:
        raise ValueError("loss_carried_forward_huf must be non-negative")

    max_offset = tax_base_huf * LOSS_OFFSET_CAP
    applied_loss = min(loss_carried_forward_huf, max_offset)
    adjusted_base = tax_base_huf - applied_loss
    tax = round(adjusted_base * TAO_RATE, 2)
    effective_rate = (tax / tax_base_huf) if tax_base_huf > 0 else 0.0

    explanation = (
        f"Adóalap: {tax_base_huf:,.0f} Ft. "
        f"Elhatárolt veszteségből levonható: {applied_loss:,.0f} Ft "
        f"(max. az adóalap 50%-a, Tao. tv. 17. §). "
        f"Módosított adóalap: {adjusted_base:,.0f} Ft. "
        f"Társasági adó (9%): {tax:,.0f} Ft."
    ).replace(",", " ")  # Hungarian thousands separator

    return {
        "tax_base_huf": tax_base_huf,
        "loss_applied_huf": applied_loss,
        "adjusted_base_huf": adjusted_base,
        "tax_huf": tax,
        "effective_rate": round(effective_rate, 4),
        "rate": TAO_RATE,
        "explanation": explanation,
    }


__all__ = ["tao_calculator", "TAO_RATE", "LOSS_OFFSET_CAP"]
