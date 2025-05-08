from dataclasses import dataclass
from typing import NamedTuple

from torch import BoolTensor, FloatTensor, LongTensor

POOLING_METHODS = ["max", "mean", "sum"]


@dataclass
class ModelInputs:
    """A model input."""

    input_ids: LongTensor
    attention_mask: FloatTensor


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
