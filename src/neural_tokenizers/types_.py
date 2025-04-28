from typing import NamedTuple

from torch import BoolTensor, FloatTensor, LongTensor


class ScoredTokenIds(NamedTuple):
    """Token information with surprisal and entropy scores from a language model."""

    token_ids: LongTensor
    token_surprisals: FloatTensor
    token_entropies: FloatTensor


class SegmentInfo(NamedTuple):
    """Output of the segmenter.

    Information about an individual segment.
    """

    segment_start: int
    segment_length: int
    score: float | None


class SegmentationOutput(NamedTuple):
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
