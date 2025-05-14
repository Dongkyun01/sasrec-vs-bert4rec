import pandas as pd
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import torch

def load_sasrec_dataset(path, min_len=2, max_len=100):
    data = pd.read_csv(path, sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    data = data[data['rating'] >= 3]
    data = data.sort_values(['user', 'timestamp'])

    unique_items = sorted(data['item'].unique())
    item2idx = {item: idx + 1 for idx, item in enumerate(unique_items)}
    data['item'] = data['item'].map(item2idx)
    num_items = len(item2idx)
    all_items = list(range(1, num_items + 1))

    user_seqs = defaultdict(list)
    for row in data.itertuples():
        user_seqs[row.user].append(row.item)

    users = list(user_seqs.keys())
    np.random.seed(42)
    np.random.shuffle(users)
    split = int(len(users) * 0.8)
    train_users, test_users = users[:split], users[split:]

    train_seqs, train_labels = {}, {}
    test_seqs, test_labels = {}, {}
    for u in train_users:
        seq = user_seqs[u]
        if len(seq) >= min_len:
            train_seqs[u] = seq[:-1]
            train_labels[u] = seq[-1]
    for u in test_users:
        seq = user_seqs[u]
        if len(seq) >= min_len:
            test_seqs[u] = seq[:-1]
            test_labels[u] = seq[-1]

    return train_seqs, train_labels, test_seqs, test_labels, num_items, all_items


def pad_sequence(seq, max_len):
    seq = seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


class SASRecDataset(Dataset):
    def __init__(self, user_ids, sequences, labels, max_len=100):
        self.inputs = [pad_sequence(sequences[u], max_len) for u in user_ids]
        self.labels = [labels[u] for u in user_ids]

    def __len__(self): return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


