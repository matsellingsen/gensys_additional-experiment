import torch

from .scan_helpers import PAD


def collate_scan_batch(batch):
    """
    Collate function for SCAN batches.

    Args:
        batch: List of tuples:
            (
                source_sequence: LongTensor [source_length],
                target_input_sequence: LongTensor [target_length + 1],
                target_output_sequence: LongTensor [target_length + 1],
            )

    Returns:
        source_batch        : LongTensor [batch_size, max_source_length]
        target_input_batch  : LongTensor [batch_size, max_target_length]
        target_output_batch : LongTensor [batch_size, max_target_length]
    """
    (
        source_sequences,
        target_input_sequences,
        target_output_sequences,
    ) = zip(*batch)

    batch_size = len(batch)

    max_source_length = max(seq.size(0) for seq in source_sequences)
    max_target_length = max(seq.size(0) for seq in target_input_sequences)

    source_batch = torch.full(
        (batch_size, max_source_length),
        PAD,
        dtype=torch.long,
    )

    target_input_batch = torch.full(
        (batch_size, max_target_length),
        PAD,
        dtype=torch.long,
    )

    target_output_batch = torch.full(
        (batch_size, max_target_length),
        PAD,
        dtype=torch.long,
    )

    for batch_index in range(batch_size):
        source_length = source_sequences[batch_index].size(0)
        target_length = target_input_sequences[batch_index].size(0)

        source_batch[batch_index, :source_length] = source_sequences[batch_index]
        target_input_batch[batch_index, :target_length] = target_input_sequences[batch_index]
        target_output_batch[batch_index, :target_length] = target_output_sequences[batch_index]

    return source_batch, target_input_batch, target_output_batch
