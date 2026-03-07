"""T5 Reranker — same as Berkeley's for fair comparison.

This is a thin wrapper that mirrors ``models.setup_t5_reranker`` and
``models.t5_reranker_score_with_clue`` from the Berkeley codebase so
that the iterative-improvement step is identical.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from models.clue_answerer import reranker_score, setup_reranker

__all__ = ["setup_reranker", "reranker_score"]
