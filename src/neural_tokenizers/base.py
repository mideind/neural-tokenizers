from torch import BoolTensor, LongTensor, nn

from neural_tokenizers.types_ import SegmentationOutput, ScoredTokenIds, SegmentInfo


class Segmenter:
    """Base class for all neural tokenizers."""

    def __init__(self, model: nn.Module):
        """Initialize the segmenter with a model.

        Args:
            model: nn.Module

        """
        self.model = model

    def tokenize(self, text: str) -> SegmentationOutput:
        """Legacy method.

        Args:
            text: str

        Returns:
            list[int]

        """
        raise NotImplementedError

    def _segment_single(self, input_bytes: list[int]) -> list[dict]:
        """Segment a single text string (in byte representation) into segments.

        Args:
            input_bytes:    list[int]

        Returns:
            segment_info:  list[{start: int, length: int}]

        """
        raise NotImplementedError

    def _segment_batched(
        self, input_bytes: LongTensor, input_mask: BoolTensor, padding_value: int
    ) -> tuple[LongTensor, LongTensor, BoolTensor]:
        """Segment a batch of text strings (in byte representation) into segments.

        L₁: length in bytes
        L₂: length in segments
        L₃: (max) length of segment in bytes

        Args:
            input_bytes:               LongTensor(B×L₁)
            input_mask:                BoolTensor(B×L₁)
            padding_value:             int

        Returns:
            segment_lengths:           LongTensor(B×L₂)
            segment_starts:            LongTensor(B×L₂)
            output_mask:               BoolTensor(B×L₂)

        """
        raise NotImplementedError
