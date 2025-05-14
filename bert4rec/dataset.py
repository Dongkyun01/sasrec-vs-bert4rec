import pandas as pd
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
import random

def load_dataset_bert4rec(path, max_len=100, seed=42):
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
    np.random.seed(seed)
    np.random.shuffle(users)
    split = int(len(users) * 0.8)
    train_users, test_users = users[:split], users[split:]

    train_seqs = {u: user_seqs[u] for u in train_users if len(user_seqs[u]) >= 5}
    test_seqs = {u: user_seqs[u] for u in test_users if len(user_seqs[u]) >= 5}

    return train_seqs, test_seqs, num_items, all_items


class BERT4RecDataset(Dataset):
    def __init__(self, user_seqs, max_len=100, mask_prob=0.15):
        self.inputs, self.labels = [], []
        self.mask_token = max([item for seq in user_seqs.values() for item in seq]) + 1
        for seq in user_seqs.values():
            seq = seq[-max_len:]
            tokens = [0] * (max_len - len(seq)) + seq
            labels = [0] * max_len
            for i in range(max_len):
                if tokens[i] != 0 and random.random() < mask_prob:
                    labels[i] = tokens[i]
                    tokens[i] = self.mask_token
            self.inputs.append(tokens)
            self.labels.append(labels)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )
__all__ = ["BERT4RecDataset", "load_dataset_bert4rec"]