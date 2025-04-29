from typing import NamedTuple

from torch import BoolTensor, FloatTensor, LongTensor


class Segment(NamedTuple):
    """A segment of a text string."""

    start: int
    length: int
    score: float
    text: str


class InnerSegmentationOutput(NamedTuple):
    """Output of the segmenter.

    Information about all segments.
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


class ScoredTokenIds(NamedTuple):
    """Token information with surprisal and entropy scores from a language model."""

    token_ids: LongTensor
    token_surprisals: FloatTensor
    token_entropies: FloatTensor
