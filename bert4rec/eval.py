import torch
import random

@torch.no_grad()
def evaluate_bert4rec(model, user_seqs, all_items, full=True, k=10, num_neg=100):
    model.eval()
    device = next(model.parameters()).device
    hits, ndcgs = 0, 0

    for u, seq in user_seqs.items():
        if len(seq) < 2: continue
        input_seq = seq[:-1][-100:]
        input_seq = [0] * (100 - len(input_seq)) + input_seq
        input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)
        logits = model(input_tensor)[0, -1]

        target = seq[-1]

        if full:
            for s in set(seq[:-1]):
                logits[s] = -float('inf')
            top_k = logits.topk(k).indices.tolist()
            if target in top_k:
                hits += 1
                rank = top_k.index(target)
                ndcgs += 1 / torch.log2(torch.tensor(rank + 2.0)).item()
        else:
            seen = set(seq)
            candidates = [i for i in all_items if i not in seen]
            negatives = random.sample(candidates, num_neg)
            sample = [target] + negatives
            sample_tensor = torch.tensor(sample).to(device)
            scores = logits[sample_tensor]
            top_k = scores.topk(k).indices.tolist()
            if 0 in top_k:
                hits += 1
                rank = top_k.index(0)
                ndcgs += 1 / torch.log2(torch.tensor(rank + 2.0)).item()

    return hits / len(user_seqs), ndcgs / len(user_seqs)
