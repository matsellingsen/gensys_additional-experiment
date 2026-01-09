import torch
from torch.utils.data import Dataset


class dataHandler(Dataset):
    def __init__(self, path: str):
        self.data = self.load_data(path)
        self.src, self.tgt = self.split_data()

        self.src_token = [t.split() for t in self.src]
        self.tgt_token = [t.split() for t in self.tgt]
        
        self.vocab_src = self.create_vocab(self.src_token)
        self.vocab_tgt = self.create_vocab(self.tgt_token)
        self.src_indexes = [self.text_to_indexes(seq, self.vocab_src) for seq in self.src_token]
        self.tgt_indexes = [self.text_to_indexes(seq, self.vocab_tgt) for seq in self.tgt_token]

        self.samples = list(zip(self.src_indexes, self.tgt_indexes))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def load_data(self, path):
        # load data from path
        with open(path, 'r') as f:
            data_lines = f.readlines()
        return [line.strip() for line in data_lines]
    
    
    def split_data(self):
        src = []
        tgt = []
        for line in self.data:
            source, target = line.split('OUT: ')
            source = source.replace('IN: ', '').strip()
            target = target.strip()

            src.append(source)
            tgt.append(target)

        return src, tgt
    
    def create_vocab(self, type: list):   

        unique_tokens = set([token for seq in type for token in seq])

        vocab = {token: idx for idx, token in enumerate(sorted(unique_tokens), start=2)}  # start from 2 to reserve 0 and 1
        print("type: ", type)
        print("vocab: ", vocab)
        vocab['<PAD>'] = 0
        vocab['<BOS>'] = 1
        return vocab
    
    def vocab_sizes(self):
        return len(self.vocab_src), len(self.vocab_tgt)
    
    def text_to_indexes(self, text, vocab):
        index_list = [vocab[token] for token in text]
        tensor = torch.tensor(index_list, dtype=torch.long)
        return tensor



    



