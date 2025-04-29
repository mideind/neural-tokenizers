from torch import BoolTensor, LongTensor, nn

from neural_tokenizers.types_ import BatchedSegmentationOutput, Segment


class Segmenter:
    """Base class for all neural tokenizers."""

    def __init__(self, model: nn.Module):
        """Initialize the segmenter with a model.

        Args:
            model: nn.Module

        """
        self.model = model

    def segmentize(
        # self, input_ids: LongTensor, attention_mask: BoolTensor
        self,
        input_bytes: LongTensor,
        input_mask: BoolTensor,
        padding_value: int,
    ) -> BatchedSegmentationOutput:
        """Segment a batch of text strings (in byte representation) into segments.

        L₁: length in bytes
        L₂: length in segments
        L₃: (max) length of segment in bytes

        Args:
            input_bytes:               LongTensor(B×L₁)
            input_mask:                BoolTensor(B×L₁)
            padding_value:             int

        Returns:
            BatchedSegmentationOutput

        """
        raise NotImplementedError
        pass

    def segmentize_text(self, text: str) -> list[Segment]:
        """Segment a single text string into segments.

        Args:
             text: str

        Returns:
            list[Segment]

        """
        raise NotImplementedError
        pass
