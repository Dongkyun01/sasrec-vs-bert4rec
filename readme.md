# SASRec vs BERT4Rec MovieLens100K 비교 프로젝트

이 프로젝트는 **Sequential Recommendation System**의 대표 모델인 `SASRec`과 `BERT4Rec`을 **MovieLens 100K 데이터셋** 기반으로 구현하고 성능을 비교한 실험입니다.

---

## 📁 프로젝트 구조

```
├── data/
│   └── ml-100k/u.data
├── sasrec/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── eval.py
├── bert4rec/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── eval.py
├── main_sasrec.py
├── main_bert4rec.py
├── compare_models.py
└── requirements.txt
```

---

## 💡 사용한 모델

### SASRec

* 단방향 트랜스포머 기반 시계열 추천 모델
* 아이템 임베딩 + 포지셔널 임베딩 사용
* 미래 아이템 예측을 위해 캐주얼 마스킹 적용

### BERT4Rec

* 양방향 트랜스포머 기반의 마스크드 모델
* 아이템 시퀀스 중 일부를 마스크하고 해당 위치의 아이템을 복원하도록 학습
* 모든 위치에서 예측 가능하므로 더 많은 문맥 정보 활용 가능

---

## 🧪 실험 환경

* 데이터: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
* 평가 방식:

  * Hit\@10, NDCG\@10
  * Sampled Ranking vs Full Ranking
* 에폭: 100
* Batch Size: 64
* Optimizer: Adam (lr=0.0005)
* Embedding dim: 128
* Transformer Layer: 3

---

## 📊 실험 결과

| Model    | Hit\@10 (Sampled) | NDCG\@10 (Sampled) | Hit\@10 (Full) | NDCG\@10 (Full) |
| -------- | ----------------- | ------------------ | -------------- | --------------- |
| SASRec   | 0.2698            | 0.1266             | 0.0265         | 0.0158          |
| BERT4Rec | **0.3069**        | **0.1445**         | 0.0265         | 0.0112          |

> Sampled Ranking 기준으로는 BERT4Rec이 SASRec보다 우수함.
> 하지만 Full Ranking에서는 SASRec이 NDCG에서 더 높음.

그래프는 `compare_models.py`를 실행하여 확인할 수 있습니다.

---

## 🚀 실행 방법

### 모델 학습

```bash
# SASRec 학습
python main_sasrec.py

# BERT4Rec 학습
python main_bert4rec.py
```

### 모델 비교 시각화

```bash
python compare_models.py
```

---



## 🙋 참고

* SASRec: Self-Attentive Sequential Recommendation (WWW 2018)
* BERT4Rec: BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer (CIKM 2019)