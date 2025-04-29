import torch

REDUCTIONS = ["none", "max", "sum", "mean"]


def lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Construct an input mask from a batch of sequence lengths.

    Arguments:
        lengths: LongTensor ∷ (Sequences)

    Returns:
        padding_mask: BoolTensor ∷ (Sequences × MaxLength)

    """
    num_sequences = lengths.shape[0]
    max_length = lengths.max()
    input_mask = (
        torch.arange(max_length, device=lengths.device).unsqueeze(0).repeat(num_sequences, 1)
    )
    input_mask = input_mask.lt(lengths.unsqueeze(-1))
    return input_mask


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


if __name__ == "__main__":
    # python -m neural_tokenizers.utils
    test_pool_segments_by_lengths()
