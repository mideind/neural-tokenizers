from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import BoolTensor, FloatTensor, LongTensor

POOLING_METHODS = ["max", "mean", "sum"]


@dataclass
class ModelInputs:
    """A model input."""

    input_ids: LongTensor
    attention_mask: FloatTensor

    @staticmethod
    def from_string(text: str) -> "ModelInputs":
        """Convert a string into a model input (for transformers-compatible models).

        Args:
            text: str

        Returns:
            ModelInputs

        """
        ids = torch.tensor(list(text.encode("utf-8")), dtype=torch.long).unsqueeze(0)
        length = ids.size(1)
        causal_mask_bool = torch.tril(torch.ones(length, length, dtype=torch.bool))
        # ∷ (Batch × Heads × Length × Length), where Batch=1, Heads=1
        causal_mask_bool = causal_mask_bool.unsqueeze(0).unsqueeze(1)
        causal_mask_float_additive = torch.zeros_like(
            causal_mask_bool, dtype=torch.float32
        ).masked_fill(~causal_mask_bool, -float("inf"))
        return ModelInputs(input_ids=ids, attention_mask=causal_mask_float_additive)


class Segment(NamedTuple):
    """A segment of a text string."""

    start: int
    length: int
    score: float
    text: str


class InnerSegmentationOutput(NamedTuple):
    """Output of the segmenter.

    Information about all segments.

    Note:
        The surprisal and entropy scores are measured in bits.

    """

    input_ids: LongTensor
    scores: FloatTensor
    segment_lengths: LongTensor
    segment_starts: LongTensor
    output_mask: BoolTensor | None
    surprisal: FloatTensor | None
    self_entropy: FloatTensor | None


class BatchedSegmentationOutput(NamedTuple):
    """Offset and length of the segments found by the segmenter in a batch of sequences."""

    offsets: LongTensor
    lengths: LongTensor
    scores: FloatTensor
    num_segments: LongTensor


class ScoredTokenIds(NamedTuple):
    """Token information with surprisal and entropy scores from a language model.

    Note:
        The surprisal and entropy scores are measured in bits.

    """

    token_ids: LongTensor
    token_surprisals: FloatTensor
    token_entropies: FloatTensor


@dataclass
class TextScores:
    """A segmentation of a string into segments of characters/tokens."""

    text: str
    chars: list[str]
    char_lens: LongTensor
    byte_ids: LongTensor
    attention_mask: BoolTensor | None = None
    # character score (composite score)
    char_scores: FloatTensor | None = None
    char_surprisals: FloatTensor | None = None
    char_entropies: FloatTensor | None = None
    # byte-level information-theoretic scores
    byte_surprisals: FloatTensor | None = None
    byte_entropies: FloatTensor | None = None

    @classmethod
    def from_string(cls, text: str) -> "TextScores":
        """Create a TextSegmentation from a string.

        Args:
            text: str

        Returns:
            TextSegmentation

        """
        model_inputs = ModelInputs.from_string(text)
        # ∷ (B × L)
        byte_ids = model_inputs.input_ids
        # ∷ (B × L)
        attention_mask = model_inputs.attention_mask
        chars = list(text)
        # ∷ (L) → (B × L)
        char_lens = torch.tensor([len(c.encode("utf-8")) for c in chars]).unsqueeze(0)
        return cls(
            text=text,
            chars=chars,
            char_lens=char_lens,
            byte_ids=byte_ids,
            attention_mask=attention_mask,
        )
