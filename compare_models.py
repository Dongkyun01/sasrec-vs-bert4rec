import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
# 로그 파일 로드
with open("real_sasrec_log.json") as f:
    sasrec = json.load(f)
with open("realbert4rec_log.json") as f:
    bert = json.load(f)

# 에포크 범위 설정
epochs = list(range(1, 101))

# SASRec 결과 추출
sas_hit_sampled = [entry["hit@10_sampled"] for entry in sasrec]
sas_ndcg_sampled = [entry["ndcg@10_sampled"] for entry in sasrec]
sas_hit_full = [entry["hit@10_full"] for entry in sasrec]
sas_ndcg_full = [entry["ndcg@10_full"] for entry in sasrec]

# BERT4Rec 결과 추출
bert_hit_sampled = bert["hit_sampled"]
bert_ndcg_sampled = bert["ndcg_sampled"]
bert_hit_full = bert["hit_full"]
bert_ndcg_full = bert["ndcg_full"]

# 1. Sampled Ranking 시각화
plt.figure(figsize=(10, 5))
plt.plot(epochs, sas_hit_sampled, label="SASRec Hit@10 (Sampled)", linestyle="--")
plt.plot(epochs, bert_hit_sampled, label="BERT4Rec Hit@10 (Sampled)")
plt.plot(epochs, sas_ndcg_sampled, label="SASRec NDCG@10 (Sampled)", linestyle="--")
plt.plot(epochs, bert_ndcg_sampled, label="BERT4Rec NDCG@10 (Sampled)")
plt.title("Sampled Ranking - Hit@10 / NDCG@10")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Sampled_Ranking.png")

# 2. Full Ranking 시각화
plt.figure(figsize=(10, 5))
plt.plot(epochs, sas_hit_full, label="SASRec Hit@10 (Full)", linestyle="--")
plt.plot(epochs, bert_hit_full, label="BERT4Rec Hit@10 (Full)")
plt.plot(epochs, sas_ndcg_full, label="SASRec NDCG@10 (Full)", linestyle="--")
plt.plot(epochs, bert_ndcg_full, label="BERT4Rec NDCG@10 (Full)")
plt.title("Full Ranking - Hit@10 / NDCG@10")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Full_Ranking.png")

# 3. 마지막 에포크 비교 막대그래프
final_labels = ["SASRec", "BERT4Rec"]
final_hit_sampled = [sas_hit_sampled[-1], bert_hit_sampled[-1]]
final_ndcg_sampled = [sas_ndcg_sampled[-1], bert_ndcg_sampled[-1]]
final_hit_full = [sas_hit_full[-1], bert_hit_full[-1]]
final_ndcg_full = [sas_ndcg_full[-1], bert_ndcg_full[-1]]

x = range(len(final_labels))
width = 0.2

plt.figure(figsize=(10, 6))
plt.bar([i - width*1.5 for i in x], final_hit_sampled, width=width, label="Hit@10 Sampled")
plt.bar([i - width*0.5 for i in x], final_ndcg_sampled, width=width, label="NDCG@10 Sampled")
plt.bar([i + width*0.5 for i in x], final_hit_full, width=width, label="Hit@10 Full")
plt.bar([i + width*1.5 for i in x], final_ndcg_full, width=width, label="NDCG@10 Full")
plt.xticks(x, final_labels)
plt.ylabel("Score")
plt.title("Final Performance Comparison")
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("Final_Performance_Comparison.png")
