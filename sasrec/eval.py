import torch
import random
import numpy as np
from sasrec.dataset import pad_sequence

@torch.no_grad()
def evaluate_full(model, user_ids, sequences, labels, seen_seqs, max_len=100, k=10):
    model.eval()
    device = next(model.parameters()).device
    hits, ndcgs = 0, 0
    for u in user_ids:
        x = torch.tensor(pad_sequence(sequences[u], max_len)).unsqueeze(0).to(device)
        logits = model(x)
        for seen in seen_seqs[u]:
            logits[0, seen] = -float('inf')
        top_k = logits[0].topk(k).indices.tolist()
        target = labels[u]
        if target in top_k:
            hits += 1
            rank = top_k.index(target)
            ndcgs += 1 / torch.log2(torch.tensor(rank + 2.0)).item()
    return hits / len(user_ids), ndcgs / len(user_ids)

@torch.no_grad()
def evaluate_sampled(model, user_ids, sequences, labels, all_items, max_len=100, k=10, num_neg=100):
    model.eval()
    device = next(model.parameters()).device
    hits, ndcgs = 0, 0
    for u in user_ids:
        x = torch.tensor(pad_sequence(sequences[u], max_len)).unsqueeze(0).to(device)
        target = labels[u]
        seen = set(sequences[u]) | {target}
        negatives = random.sample([i for i in all_items if i not in seen], num_neg)
        test_items = [target] + negatives
        test_tensor = torch.tensor(test_items).to(device)
        logits = model(x)[0][test_tensor]
        top_k = logits.topk(k).indices.tolist()
        if 0 in top_k:
            hits += 1
            rank = top_k.index(0)
            ndcgs += 1 / torch.log2(torch.tensor(rank + 2.0)).item()
    return hits / len(user_ids), ndcgs / len(user_ids)
