import torch
from torch.utils.data import DataLoader
from bert4rec.dataset import load_dataset_bert4rec, BERT4RecDataset
from bert4rec.model import BERT4Rec
from bert4rec.train import train
from bert4rec.eval import evaluate_bert4rec
import json

# Load data
train_seqs, test_seqs, num_items, all_items = load_dataset_bert4rec(r"C:\Users\USER\Desktop\추천시스템연습\ml-100k\u.data")

# Prepare dataset & dataloader
dataset = BERT4RecDataset(train_seqs)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Model & optimizer setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERT4Rec(num_items=num_items).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Logging dict
bert_log = {
    "epoch": list(range(1, 101)),
    "hit_sampled": [],
    "ndcg_sampled": [],
    "hit_full": [],
    "ndcg_full": []
}

# Train and evaluate
for epoch in range(1, 101):
    loss = train(model, dataloader, optimizer, device)
    hit_s, ndcg_s = evaluate_bert4rec(model, test_seqs, all_items, full=False)
    hit_f, ndcg_f = evaluate_bert4rec(model, test_seqs, all_items, full=True)

    print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Sampled Hit@10: {hit_s:.4f} | NDCG@10: {ndcg_s:.4f} | Full Hit@10: {hit_f:.4f} | NDCG@10: {ndcg_f:.4f}")

    bert_log["hit_sampled"].append(hit_s)
    bert_log["ndcg_sampled"].append(ndcg_s)
    bert_log["hit_full"].append(hit_f)
    bert_log["ndcg_full"].append(ndcg_f)

# Save log
with open("realbert4rec_log.json", "w") as f:
    json.dump(bert_log, f, indent=2)

# Final evaluation
print("\nFinal Evaluation on Test Set")
print(f"Full Ranking => Hit@10: {hit_f:.4f} | NDCG@10: {ndcg_f:.4f}")
print(f"Sampled Ranking => Hit@10: {hit_s:.4f} | NDCG@10: {ndcg_s:.4f}")
