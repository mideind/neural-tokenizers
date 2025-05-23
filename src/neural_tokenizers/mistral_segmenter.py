from dataclasses import dataclass

import numpy as np
import torch
from torch import BoolTensor, FloatTensor, LongTensor, nn
from transformers import MistralForCausalLM

from neural_tokenizers.base import Segmenter
from neural_tokenizers.types_ import (
    BatchedSegmentationOutput,
    ModelInputs,
    ScoredTokenIds,
    Segment,
    TextScores,
)
from neural_tokenizers.utils import (
    LN_OF_TWO,
    compute_boundaries_by_spikes_in_discontig_windows,
    lengths_to_float_attention_mask_right_padding,
    pool_segments_by_lengths,
)


def encode_batch(texts: list[str]) -> ModelInputs:
    """Encode a batch of strings into a model input.

    Args:
        texts: list[str]

    Returns:
        ModelInputs(input_ids: LongTensor(B × L), attention_mask: FloatTensor(B × 1 × L × L))

    """
    # arrays directly from np.frombuffer are not writable, so we need to copy (else torch panics),
    # Python's bytes-view-object must be read as uint8 array
    byte_seqs = [
        torch.from_numpy(np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64))
        for text in texts
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        byte_seqs, batch_first=True, padding_value=256, padding_side="right"
    )
    seq_lens = torch.tensor([len(seq) for seq in byte_seqs])
    # mistral accepts both bool and float attention masks (it silently converts bool to float)
    attention_mask = lengths_to_float_attention_mask_right_padding(seq_lens)
    return ModelInputs(input_ids=input_ids, attention_mask=attention_mask)


class MistralSegmenter(Segmenter):
    """A neural tokenizer that segments strings into segments of characters/tokens."""

    def __init__(self, model: nn.Module, window_len: int = 3):
        """Initialize a MistralSegmenter.

        Args:
            model: nn.Module
            window_len: int

        """
        # super constructor sets self.model and self.window_len
        super().__init__(model, window_len)

    def segmentize_text(self, text: str) -> list[Segment]:
        """Segment a string into segments using a neural tokenizer.

        Args:
            text: str

        Returns:
            SegmentationOutput

        """
        segmentation = TextScores.from_string(text)

        scores = self._compute_surprisal(
            input_ids=segmentation.byte_ids, attention_mask=segmentation.attention_mask
        )
        segmentation.byte_surprisals = scores.token_surprisals
        segmentation.byte_entropies = scores.token_entropies

        char_surprisals = (
            pool_segments_by_lengths(
                # ∷ (B × L) → (B × L × 1)
                hidden_state=segmentation.byte_surprisals.unsqueeze(-1),
                segment_lengths=segmentation.char_lens,
                reduction=self.pooling_method,
            )
            .squeeze(-1)
            .numpy()[0]
        )

        char_entropies = (
            pool_segments_by_lengths(
                segmentation.byte_entropies.unsqueeze(-1),
                segmentation.char_lens,
                reduction="max",
            )
            .squeeze(-1)
            .numpy()[0]
        )

        # spikes in the /model's/ entropy of the conditional distribution of the next character
        selected_spikes = []
        window_offsets = list(range(0, len(char_entropies) - self.window_len, self.window_len))
        for offset in window_offsets:
            end_window = offset + self.window_len
            rel_offset = char_entropies[offset:end_window].argmax()
            abs_offset = offset + rel_offset
            selected_spikes.append(abs_offset)
        if selected_spikes[0] != 0:
            selected_spikes.insert(0, 0)
        if selected_spikes[-1] != len(char_entropies):
            selected_spikes.append(len(char_entropies))

        segments = []
        for start, end in zip(selected_spikes[:-1], selected_spikes[1:], strict=True):
            segments.append(
                Segment(
                    start=start,
                    length=end - start,
                    score=char_surprisals[start:end].sum(),
                    text=text[start:end],
                )
            )

        return segments

    def segmentize_ids(
        self, input_bytes: LongTensor, attention_mask: FloatTensor
    ) -> BatchedSegmentationOutput:
        """Segment a batch of text strings (in byte representation) into segments.

        L₁: length in bytes
        L₂: length in segments
        L₃: (max) length of segment in bytes

        Args:
            input_bytes:               LongTensor(B × L₁)
            attention_mask:            FloatTensor(B × 1 × L₁ × L₁)


        Returns:
            BatchedSegmentationOutput

        """
        scores: ScoredTokenIds = self._compute_surprisal(
            input_ids=input_bytes, attention_mask=attention_mask
        )

        sequence_lengths = input_bytes.lt(256).sum(dim=-1)
        boundaries, segment_lengths, segment_scores, num_segments = (
            compute_boundaries_by_spikes_in_discontig_windows(
                scores.token_entropies.numpy(),
                scores.token_surprisals.numpy(),
                sequence_lengths.numpy(),
                self.window_len,
            )
        )

        # clone to prevent stale pointers when switching from numba to torch
        output_segments = BatchedSegmentationOutput(
            offsets=torch.from_numpy(boundaries.copy()),
            lengths=torch.from_numpy(segment_lengths.copy()),
            scores=torch.from_numpy(segment_scores.copy()),
            num_segments=torch.from_numpy(num_segments.copy()),
        )
        return output_segments

    def _compute_surprisal(
        self,
        input_ids: LongTensor,
        attention_mask: BoolTensor,
    ) -> ScoredTokenIds:
        """Score a batch of sequences (of token ids).

        Args:
            input_ids:       LongTensor(B × L)
            attention_mask:  BoolTensor(B × L)

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
        # shift/roll (the negative sign is to roll left, [0, 1, 2] -> [1, 2, 0])
        # ∷ (B × L) → (B × L × 1)
        target_ids = input_ids.roll(-1).unsqueeze(-1)
        # ∷ (B × L × V) → (B × L × 1)
        log_probs = all_logits.gather(index=target_ids, dim=2)
        # ∷ (B × L × 1) → (B × L)
        log_probs = log_probs.squeeze(-1)
        # undo the shift/roll
        log_probs = log_probs.roll(1)
        # use correct sign
        surprisal_per_token = -log_probs / LN_OF_TWO

        # the score of the first token is meaningless (due to the shift/roll)
        surprisal_per_token[:, 0] = 0.0

        return ScoredTokenIds(
            token_ids=input_ids,
            token_surprisals=surprisal_per_token,
            token_entropies=entropy_per_token,
        )

    def score_text(self, text: str) -> TextScores:
        """Score a string.

        Args:
            text: str

        Returns:
            ScoredTokenIds

        """
        scoring = TextScores.from_string(text)
        scored_ids = self._compute_surprisal(
            input_ids=scoring.byte_ids, attention_mask=scoring.attention_mask
        )

        scoring.char_surprisals = pool_segments_by_lengths(
            # ∷ (B × L) → (B × L × 1)
            hidden_state=scored_ids.token_surprisals.unsqueeze(-1),
            segment_lengths=scoring.char_lens,
            reduction="max",
        ).squeeze(-1)[0]

        scoring.char_entropies = (
            pool_segments_by_lengths(
                scored_ids.token_entropies.unsqueeze(-1),
                scoring.char_lens,
                reduction="max",
            ).squeeze(-1)[0]
        ).roll(1)

        return scoring

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
