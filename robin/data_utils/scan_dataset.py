import torch
from torch.utils.data import Dataset

from .scan_helpers import encode, make_decoder_inputs


class ScanDataset(Dataset):
    """
    SCAN Dataset.

    Each item returns:
      source_sequence        : LongTensor [source_length]
      target_input_sequence  : LongTensor [target_length + 1]  (BOS + target)
      target_output_sequence : LongTensor [target_length + 1]  (target + EOS)
    """

    def __init__(
        self,
        input_sequences,
        output_sequences,
        source_vocabulary,
        target_vocabulary,
    ):
        """
        Args:
            input_sequences   : List[List[str]]
            output_sequences  : List[List[str]]
            source_vocabulary : Dict[str, int]
            target_vocabulary : Dict[str, int]
        """
        assert len(input_sequences) == len(output_sequences), (
            "Input and output sequences must have the same length"
        )

        self.input_sequences = input_sequences
        self.output_sequences = output_sequences
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        # Encode source tokens
        source_ids = encode(
            self.input_sequences[index],
            self.source_vocabulary,
        )

        # Encode target tokens
        target_ids = encode(
            self.output_sequences[index],
            self.target_vocabulary,
        )

        # Prepare decoder input/output sequences
        target_input_ids, target_output_ids = make_decoder_inputs(target_ids)

        return (
            torch.tensor(source_ids, dtype=torch.long),
            torch.tensor(target_input_ids, dtype=torch.long),
            torch.tensor(target_output_ids, dtype=torch.long),
        )
