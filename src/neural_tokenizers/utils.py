import numba
import numpy as np
import torch

REDUCTIONS = ["none", "max", "sum", "mean"]
LN_OF_TWO = 0.6931471805599453


def lengths_to_input_mask_right(lengths: torch.Tensor) -> torch.Tensor:
    """Construct an input mask  from a batch of sequence lengths.

    The mask is 1 where elements should be kept, 0 when they should be ignored.
    The mask is assumed to be right-padded.

    Arguments:
        lengths: LongTensor ∷ (Sequences)

    Returns:
        input_mask: BoolTensor ∷ (Sequences × MaxLength)

    """
    num_sequences = lengths.shape[0]
    max_length = lengths.max()
    input_mask = (
        torch.arange(max_length, device=lengths.device).unsqueeze(0).repeat(num_sequences, 1)
    )
    input_mask = input_mask.lt(lengths.unsqueeze(-1))
    return input_mask


def lengths_to_bool_attention_mask_right_padding(lengths: torch.Tensor) -> torch.Tensor:
    """Construct a bool attention mask  from a batch of sequence lengths.

    The mask is 1 where elements should be kept, 0 when they should be ignored.
    The mask is assumed to be right-padded.

    Arguments:
        lengths: LongTensor ∷ (Sequences)

    Returns:
        causal_attn_mask: BoolTensor ∷ (Sequences × 1 × Tokens × Tokens)

    """
    bsz = lengths.shape[0]
    seq_len = lengths.max()

    # The diagonal is 1, and below is 1, rest is 0; example:
    #   [[1, 0],
    #    [1, 1]]
    # ∷ (SeqLen × SeqLen)
    causal_keep_mask = lengths.new_ones(seq_len, seq_len, dtype=torch.bool).tril()
    input_mask = lengths_to_input_mask_right(lengths)

    # ∷ (1 × SeqLen × SeqLen) ⊙ (Batch × SeqLen × 1)
    causal_keep_mask = causal_keep_mask.repeat(bsz, 1, 1) * input_mask.bool().unsqueeze(-1)

    # ∷ (Batch × SeqLen × SeqLen) → (Batch × SeqLen × SeqLen)
    causal_keep_mask = causal_keep_mask.view(bsz, 1, seq_len, seq_len)

    # ∷ (Sequences × Tokens × 1) @ (Sequences × 1 × Tokens) → (Sequences × Tokens × Tokens)
    bidir_keep_mask = input_mask.unsqueeze(-1) * input_mask.unsqueeze(1)
    # ∷ (Sequences × Tokens × Tokens) → (Sequences × 1 × Tokens × Tokens)
    bidir_keep_mask = bidir_keep_mask.unsqueeze(1)
    # element-wise multiplication (element-wise AND)
    causal_keep_mask = causal_keep_mask * bidir_keep_mask

    return causal_keep_mask


def lengths_to_float_attention_mask_right_padding(lengths: torch.Tensor) -> torch.Tensor:
    """Construct a float attention mask  from a batch of sequence lengths.

    This is a keep_mask, not a drop_mask. That means it has a value of -inf
    when the sequence is padded, and 0 otherwise.
    This mask is used by adding it to a tensor (instead of multiplying it).
    The mask is assumed to be right-padded.

    Arguments:
        lengths: LongTensor ∷ (Sequences)

    Returns:
        causal_attn_mask: FloatTensor ∷ (Sequences × 1 × Tokens × Tokens)

    """
    bsz = lengths.shape[0]
    seq_len = lengths.max()

    # The diagonal is 1, and below is 1, rest is 0; example:
    #   [[1, 0],
    #    [1, 1]]
    # ∷ (SeqLen × SeqLen)
    causal_keep_mask = lengths.new_ones(seq_len, seq_len, dtype=torch.bool).tril()
    input_mask = lengths_to_input_mask_right(lengths)

    # ∷ (1 × SeqLen × SeqLen) ⊙ (Batch × SeqLen × 1)
    causal_keep_mask = causal_keep_mask.repeat(bsz, 1, 1) * input_mask.bool().unsqueeze(-1)

    # ∷ (Batch × SeqLen × SeqLen) → (Batch × SeqLen × SeqLen)
    causal_keep_mask = causal_keep_mask.view(bsz, 1, seq_len, seq_len)

    # ∷ (Sequences × Tokens × 1) @ (Sequences × 1 × Tokens) → (Sequences × Tokens × Tokens)
    bidir_keep_mask = input_mask.unsqueeze(-1) * input_mask.unsqueeze(1)
    # ∷ (Sequences × Tokens × Tokens) → (Sequences × 1 × Tokens × Tokens)
    bidir_keep_mask = bidir_keep_mask.unsqueeze(1)
    # element-wise multiplication (element-wise AND)
    causal_keep_mask = causal_keep_mask * bidir_keep_mask

    # when this mask is added (plus) to a float tensor,
    # zero is added to elements which should be kept,
    # -inf is added to elements which should be ignored (masked)
    # ∷ (Sequences × 1 × Tokens × Tokens)
    additive_causal_keep_mask = torch.zeros_like(
        causal_keep_mask, dtype=torch.float32
    ).masked_fill_(~causal_keep_mask, -float("inf"))

    return additive_causal_keep_mask


# @numba.njit
def compute_boundaries_by_spikes_in_discontig_windows(
    token_signal: numba.float32[:, :],
    token_scores: numba.float32[:, :],
    sequence_lengths: numba.int32[:],
    window_len: int,
) -> tuple[numba.int32[:, :], numba.int32[:, :], numba.float32[:, :], numba.int32[:]]:
    """Jitted implementation of spike detection in an array of floats.

    The input array must be a contiguous numpy array.

    Arguments:
        token_signal: ∷ numpy 2d array float32 (Sequences × Tokens)
        token_scores: ∷ numpy 2d array float32 (Sequences × Tokens)
        sequence_lengths: ∷ numpy 1d array int32 (Sequences)
        window_len: int

    Returns:
        boundaries: ∷ ndarray int32 (Sequences × Tokens)
        segment_lengths: ∷ ndarray int32 (Sequences × Tokens)
        segment_scores: ∷ ndarray float32 (Sequences × Tokens)
        num_segments: ∷ ndarray int32 (Sequences)

    """
    bsz = sequence_lengths.shape[0]
    max_sequence_width = token_signal.shape[1]
    # round up to the nearest window_len
    max_segment_count = (
        (max_sequence_width // window_len) + int(max_sequence_width % window_len > 0) + 1
    )
    # the boundaries are have indices in so called "fence-post indexing",
    # this means when the index is 5, that is the end of the 5th segment,
    # so index 1 is the end of segment 1, and index 0 is the start of segment 1.
    bounds = np.zeros((bsz, max_segment_count + 1), dtype=np.int32)

    # we start with one boundary (the start)
    # when we find a new boundary and therefore start a new segment, we increment the count
    num_segments = np.full((bsz), dtype=np.int32, fill_value=0)

    for seq_idx in range(bsz):
        # make sure we don't overwrite the first boundary
        cursor = 1
        ntokens_in_sequence = sequence_lengths[seq_idx]

        for start_offset in range(0, ntokens_in_sequence, window_len):
            end = start_offset + window_len
            if start_offset == 0:
                rel_offset = token_signal[seq_idx, 1:end].argmax() + 1
            else:
                rel_offset = token_signal[seq_idx, start_offset:end].argmax()

            abs_offset = start_offset + rel_offset
            bounds[seq_idx, cursor] = abs_offset
            num_segments[seq_idx] += 1
            cursor += 1

        # terminate the segments (the last boundary is the end of the sequence)
        if bounds[seq_idx, cursor - 1] < sequence_lengths[seq_idx]:
            bounds[seq_idx, cursor] = sequence_lengths[seq_idx]
            num_segments[seq_idx] += 1
            cursor += 1

    max_segment_count = num_segments.max()

    # compute the length and score of each segment
    segment_lengths = np.zeros((bsz, max_segment_count), dtype=np.int32)
    segment_scores = np.zeros((bsz, max_segment_count), dtype=np.float32)
    for seq_idx in range(0, bsz):
        for chunk_idx in range(0, num_segments[seq_idx]):
            # for legibility
            left = bounds[seq_idx, chunk_idx]
            right = bounds[seq_idx, chunk_idx + 1]

            segment_lengths[seq_idx, chunk_idx] = right - left
            segment_scores[seq_idx, chunk_idx] = token_scores[seq_idx, left:right].sum()

    return bounds, segment_lengths, segment_scores, num_segments


def pool_segments_by_lengths(
    hidden_state: torch.Tensor,
    segment_lengths: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Map hidden state (Sequences × Tokens × Hidden) into segments (Sequences × Segments).

    Arguments:
        hidden_state:    ∷ (Sequences × Tokens × Hidden) FloatTensor
        segment_lengths: ∷ (Sequences × Segments)        LongTensor
        reduction: str,  one of {"none", "max", "sum", "mean"}

    Returns:
        segment_states  ∷ (Sequences × Segments × Hidden) FloatTensor

    """
    if reduction not in REDUCTIONS:
        raise ValueError(f"Invalid reduction: {reduction}")

    bsz, width, nfeatures = hidden_state.shape
    num_segments = segment_lengths.shape[1]
    max_segment_len = torch.max(segment_lengths)

    # Compute relative offset of tokens (relative to its target segment) then filter to get a mask
    # ∷ (Sequences × Segments × MaxSegmentLen)
    is_token_in_segment = (
        torch.arange(max_segment_len, device=hidden_state.device)
        .tile(bsz, num_segments, 1)
        .lt(segment_lengths.unsqueeze(-1))
    )

    # Compute location of tokens into the original hidden_state tensor.
    # We subtract 1 since cumsum of Trues's starts at 1.
    # ∷ (Sequences × (Segments · MaxSegmentLen))
    abs_offsets = is_token_in_segment.view(bsz, -1).cumsum(dim=-1) - 1

    # We can only gather using tensors which have the same shape,
    #   so we pad the original hidden_state then use gather to put the tokens in the right place

    extra_length_for_padding = (max_segment_len * num_segments) - width
    # ∷ (Sequences × ExtraLengthForPadding × Hidden)
    padding = hidden_state.new_zeros(bsz, extra_length_for_padding, nfeatures)

    # ∷ (Sequences × (Segments · MaxSegmentLen) × Hidden)
    hidden_state_padded = torch.cat([hidden_state, padding], dim=1)

    # The index tensor for gather has to have the exact same shape as the input tensor for gather.
    # The expand operation does not allocate new memory.
    abs_offsets_like_hidden_state = abs_offsets.unsqueeze(-1).expand(-1, -1, nfeatures)
    # out[i][j][k] = input[i][ index[i][j][k] ][k]  # when dim == 1
    hidden_state_padded_ = torch.gather(
        hidden_state_padded, dim=1, index=abs_offsets_like_hidden_state
    )

    # ∷ (Sequences × Segments × MaxSegmentLen × Hidden)
    hidden_state_segments = hidden_state_padded_.view(bsz, num_segments, max_segment_len, nfeatures)
    # Mask out the padding elements (so that sum and mean are correct)
    hidden_state_segments = hidden_state_segments * is_token_in_segment.unsqueeze(-1)

    if reduction == "none":
        return hidden_state_segments

    if reduction == "max":
        return hidden_state_segments.max(dim=2).values

    if reduction == "sum":
        return hidden_state_segments.sum(dim=2)

    if reduction == "mean":
        hidden_state_segments = hidden_state_segments.sum(dim=2)
        return hidden_state_segments / segment_lengths.unsqueeze(2).float()


def test_pool_segments_by_lengths():  # noqa
    # (Sequences × Tokens × Embed)
    #   (a) (b   .   .) (c) (d  .)
    # [[ 6,  5,  4,  3,  2,  1, 0],
    #  [13, 12, 11, 10,  9,  8, 7]]
    scores = torch.tensor([[6, 5, 4, 3, 2, 1, 0], [13, 12, 11, 10, 9, 8, 7]]).float().unsqueeze(-1)
    lengths = torch.tensor([[1, 3, 1, 2], [1, 3, 1, 2]]).long()

    # [[ 6.0,  5.0, 2.0, 1.0],
    #  [13.0, 12.0, 9.0, 8.0]]
    max_pool = pool_segments_by_lengths(
        hidden_state=scores, segment_lengths=lengths, reduction="max"
    ).squeeze(-1)

    assert torch.allclose(max_pool, torch.tensor([[6.0, 5.0, 2.0, 1.0], [13.0, 12.0, 9.0, 8.0]]))  # noqa: S101

    # [[ 6.0,  4.0, 2.0, 0.5],
    #  [13.0, 11.0, 9.0, 7.5]]
    mean_pool = pool_segments_by_lengths(
        hidden_state=scores, segment_lengths=lengths, reduction="mean"
    ).squeeze(-1)
    assert torch.allclose(mean_pool, torch.tensor([[6.0, 4.0, 2.0, 0.5], [13.0, 11.0, 9.0, 7.5]]))  # noqa: S101

    # [[ 6.0, 12.0, 2.0,  1.0],
    #  [13.0, 33.0, 9.0, 15.0]]
    sum_pool = pool_segments_by_lengths(
        hidden_state=scores, segment_lengths=lengths, reduction="sum"
    ).squeeze(-1)
    assert torch.allclose(sum_pool, torch.tensor([[6.0, 12.0, 2.0, 1.0], [13.0, 33.0, 9.0, 15.0]]))  # noqa: S101


def test_compute_boundaries_by_spikes_in_discontig_windows() -> None:
    """Test that the boundaries are computed correctly."""
    token_signal = np.array([[3, 2, 1, 0, 5, 4, 3, 2]], dtype=np.float32)
    token_scores = token_signal.copy()
    sequence_lengths = np.array([token_signal.shape[1]], dtype=np.int32)
    window_len = 3

    # fence posts   0  1  2  3  4  5  6  7  8
    # .    signal  [3, 2, 1, 0, 5, 4, 3, 2]
    # .   windows  |        |        |     |
    # .     spike  |^  ^    |   ^    |^    |^
    # .  segment    a  b  b  b  c  c  d  d  ∅  # (label of segment)
    #
    # resulting boundaries: [0, 1, 4, 6, 8]
    # and lengths can be easily read

    expected_num_segments = np.array([4], dtype=np.int32)
    expected_boundaries = np.array([[0, 1, 4, 6, 8]], dtype=np.int32)
    expected_segment_lengths = np.array([[1, 3, 2, 2]], dtype=np.int32)

    boundaries, segment_lengths, _segment_scores, num_segments = (
        compute_boundaries_by_spikes_in_discontig_windows(
            token_signal,
            token_scores,
            sequence_lengths,
            window_len,
        )
    )

    # fmt: off
    assert np.allclose(boundaries, expected_boundaries), f"{boundaries} != {expected_boundaries}"
    assert np.allclose(segment_lengths, expected_segment_lengths), f"{segment_lengths} != {expected_segment_lengths}"  # noqa
    assert np.allclose(num_segments, expected_num_segments), f"{num_segments} != {expected_num_segments}"  # noqa
    # fmt: on


if __name__ == "__main__":
    # python -m neural_tokenizers.utils
    test_pool_segments_by_lengths()
    test_compute_boundaries_by_spikes_in_discontig_windows()
