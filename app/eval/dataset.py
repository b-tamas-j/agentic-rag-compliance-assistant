"""Evaluation dataset schema and loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# Default dataset location (versioned in the repo).
DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "eval" / "questions.json"


class EvalQuestion(BaseModel):
    """A single labelled evaluation example."""

    id: str
    category: Literal["tao", "off_topic"]
    question: str
    expected_sections: list[str] = Field(default_factory=list)
    expected_terms: list[str] = Field(default_factory=list)
    topic: str = ""


def load_dataset(path: Path | str = DEFAULT_DATASET_PATH) -> list[EvalQuestion]:
    """Load and validate the eval dataset from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalQuestion.model_validate(item) for item in raw]


__all__ = ["EvalQuestion", "DEFAULT_DATASET_PATH", "load_dataset"]
