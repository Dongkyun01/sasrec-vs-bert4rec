# main_sasrec.py
import torch
from torch.utils.data import DataLoader
from sasrec.dataset import load_sasrec_dataset, SASRecDataset
from sasrec.model import SASRec
from sasrec.train import train
from sasrec.eval import evaluate_full, evaluate_sampled
import json

# 1. 데이터 로드
path = r"C:\Users\USER\Desktop\추천시스템연습\ml-100k\u.data"
train_seqs, train_labels, test_seqs, test_labels, num_items, all_items = load_sasrec_dataset(path)

# 2. 데이터로더 준비
train_dataset = SASRecDataset(list(train_seqs.keys()), train_seqs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3. 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SASRec(num_items=num_items).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# 4. 로그 초기화
log_history = []

# 5. 학습 및 평가 루프
for epoch in range(1, 101):
    loss = train(model, train_loader, optimizer, device)

    hit_f, ndcg_f = evaluate_full(model, list(test_seqs.keys()), test_seqs, test_labels, test_seqs)
    hit_s, ndcg_s = evaluate_sampled(model, list(test_seqs.keys()), test_seqs, test_labels, all_items)

    print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Sampled Hit@10: {hit_s:.4f} | NDCG@10: {ndcg_s:.4f} | Full Hit@10: {hit_f:.4f} | NDCG@10: {ndcg_f:.4f}")

    log_history.append({
        "epoch": epoch,
        "loss": loss,
        "hit@10_sampled": hit_s,
        "ndcg@10_sampled": ndcg_s,
        "hit@10_full": hit_f,
        "ndcg@10_full": ndcg_f
    })

# 6. 로그 저장
with open("real_sasrec_log.json", "w") as f:
    json.dump(log_history, f, indent=2)

# 7. 최종 평가 출력
print("\nFinal Evaluation on Test Set")
print(f"Full Ranking => Hit@10: {hit_f:.4f} | NDCG@10: {ndcg_f:.4f}")
print(f"Sampled Ranking => Hit@10: {hit_s:.4f} | NDCG@10: {ndcg_s:.4f}")
