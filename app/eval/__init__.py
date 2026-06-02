"""Evaluation harness for the compliance agent."""

from app.eval.dataset import DEFAULT_DATASET_PATH, EvalQuestion, load_dataset
from app.eval.judge import JudgeVerdict, judge_answer
from app.eval.metrics import (
    aggregate,
    category_correct,
    citation_accuracy,
    expected_terms_coverage,
    retrieval_recall_at_k,
)

__all__ = [
    "DEFAULT_DATASET_PATH",
    "EvalQuestion",
    "JudgeVerdict",
    "aggregate",
    "category_correct",
    "citation_accuracy",
    "expected_terms_coverage",
    "judge_answer",
    "load_dataset",
    "retrieval_recall_at_k",
]
