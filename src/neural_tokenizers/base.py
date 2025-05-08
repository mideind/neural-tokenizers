from torch import BoolTensor, LongTensor, nn

from neural_tokenizers.types_ import POOLING_METHODS, BatchedSegmentationOutput, Segment, TextScores


class Segmenter:
    """Base class for all neural tokenizers."""

    def __init__(self, model: nn.Module, window_len: int = 3):
        """Initialize the segmenter with a model.

        Argu
            model: nn.Module
            window_len: int
        """
        self.model = model
        self.pooling_method: str = "max"
        self.window_len: int = window_len

    def set_window_len(self, window_len: int) -> None:
        """Set the window length.

        Args:
            window_len: int

        """
        self.window_len = window_len

    def set_pooling_method(self, method_name: str) -> None:
        """Set the pooling method.

        Args:
            method_name: str

        """
        if method_name not in POOLING_METHODS:
            raise ValueError(f"Invalid pooling method: {method_name}")
        self.pooling_method = method_name

    def segmentize_ids(
        self, input_ids: LongTensor, attention_mask: BoolTensor
    ) -> BatchedSegmentationOutput:
        """Segment a batch of text strings (in byte representation) into segments.

        L₁: length in bytes
        L₂: length in segments
        L₃: (max) length of segment in bytes

        Args:
            input_ids:               LongTensor(B×L₁)
            attention_mask:          BoolTensor(B×L₁)

        Returns:
            BatchedSegmentationOutput

        """
        raise NotImplementedError

    def segmentize_text(self, text: str) -> list[Segment]:
        """Segment a single text string into segments.

        Args:
             text: str

        Returns:
            list[Segment]

        """
        raise NotImplementedError

    def score_text(self, text: str) -> TextScores:
        """Score a single text string.

        Args:
            text: str

        Returns:
            TextScores

        """
        raise NotImplementedError
