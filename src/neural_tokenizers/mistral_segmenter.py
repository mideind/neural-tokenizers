from dataclasses import dataclass

import torch
from torch import BoolTensor, FloatTensor, LongTensor, nn
from transformers import MistralForCausalLM

from neural_tokenizers.base import Segmenter
from neural_tokenizers.types_ import BatchedSegmentationOutput, ScoredTokenIds, Segment


@dataclass
class TextSegmentation:
    """A segmentation of a string into segments of characters/tokens."""

    text: str
    chars: list[str]
    char_lens: LongTensor
    byte_ids: LongTensor
    attention_mask: BoolTensor | None = None
    # character score (composite score)
    char_scores: FloatTensor | None = None
    # byte-level information-theoretic scores
    byte_surprisals: FloatTensor | None = None
    byte_entropies: FloatTensor | None = None

    @classmethod
    def from_string(cls, text: str) -> "TextSegmentation":
        """Create a TextSegmentation from a string.

        Args:
            text: str

        Returns:
            TextSegmentation

        """
        model_inputs = string_to_model_inputs(text)
        # ∷ (B × L)
        byte_ids = model_inputs["input_ids"]
        # ∷ (B × L)
        attention_mask = model_inputs["attention_mask"]
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


def string_to_model_inputs(text: str) -> dict[str, torch.Tensor]:
    """Convert a string into a model input (for transformers-compatible models).

    Args:
        text: str

    Returns:
        dict[str, torch.Tensor]

    """
    ids = torch.tensor(list(text.encode("utf-8")), dtype=torch.long).unsqueeze(0)
    length = len(ids)
    causal_mask_bool = torch.tril(torch.ones(length, length, dtype=torch.bool))
    causal_mask_bool = causal_mask_bool.unsqueeze(0).unsqueeze(1)
    causal_mask_float_additive = torch.zeros_like(
        causal_mask_bool, dtype=torch.float32
    ).masked_fill(~causal_mask_bool, -float("inf"))
    return {"input_ids": ids, "attention_mask": causal_mask_float_additive}


class MistralSegmenter(Segmenter):
    """A neural tokenizer that segments strings into segments of characters/tokens."""

    def __init__(self, model: nn.Module):
        """Initialize a MistralSegmenter.

        Args:
            model: nn.Module

        """
        # super constructor sets self.model
        super().__init__(model)

    def segmentize_text(self, text: str) -> list[Segment]:
        """Segment a string into segments using a neural tokenizer.

        Args:
            text: str

        Returns:
            SegmentationOutput

        """
        segmentation = TextSegmentation.from_string(text)

        scored_ids = self.score_ids(
            input_ids=segmentation.byte_ids, attention_mask=segmentation.attention_mask
        )
        segmentation.byte_surprisals = scored_ids.token_surprisals
        segmentation.byte_entropies = scored_ids.token_entropies

        # print("inside segment_string")
        # foo = determine_segment_boundaries(segmentation)
        return segmentation

    def segmentize(
        self, input_bytes: LongTensor, input_mask: BoolTensor, padding_value: int
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

    def score_ids(
        self,
        input_ids: LongTensor,
        attention_mask: BoolTensor,
        ignore_leading_token: bool = True,  # noqa: FBT001, FBT002
    ) -> ScoredTokenIds:
        """Score a batch of sequences (of token ids).

        Args:
            input_ids:       LongTensor(B × L)
            attention_mask:  BoolTensor(B × L)
            ignore_leading_token: bool

        Returns:
            ScoredIds(ids: LongTensor(B × L), scores: FloatTensor(B × L))

        """
        model_out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        unnormalized_logits = model_out.logits
        all_logits = unnormalized_logits.log_softmax(-1)
        all_logits = all_logits.detach().cpu()

        # compute self-entropy
        all_probs = all_logits.softmax(-1)
        # use correct sign
        entropy_per_token = -torch.sum(all_probs * all_logits, dim=-1)

        # compute surprisal
        # shift/roll left (the negative sign)
        # ∷ (B × L) → (B × L × 1)
        target_ids = input_ids.roll(-1).unsqueeze(-1)
        # ∷ (B × L × V) → (B × L × 1)
        log_probs = all_logits.gather(index=target_ids, dim=2)
        # ∷ (B × L × 1) → (B × L)
        log_probs = log_probs.squeeze(-1)
        # unroll to align input_id with its score
        log_probs = log_probs.roll(1)
        # use correct sign
        surprisal_per_token = -log_probs

        if ignore_leading_token:
            surprisal_per_token[:, 0] = 0.0

        return ScoredTokenIds(
            token_ids=input_ids,
            token_surprisals=surprisal_per_token,
            token_entropies=entropy_per_token,
        )

    def score_string(self, text: str) -> ScoredTokenIds:
        """Score a string.

        Args:
            text: str

        Returns:
            ScoredTokenIds

        """
        model_in = string_to_model_inputs(text)
        return self.score_ids(**model_in)

    @classmethod
    def from_path(cls, path: str) -> "MistralSegmenter":
        """Create a MistralSegmenter from a path to a model checkpoint (locally or Huggingface).

        Args:
            path: str

        Returns:
            MistralSegmenter

        """
        model = MistralForCausalLM.from_pretrained(path, device_map="auto")
        return cls(model)
