"""영화로 학습한 선형 모델 -> 영화 Test / 도서 Test 평가 + 유저별 Top-K 추천

영화·책 평점은 8:1:1 (train / validation / test).
영화 유저 프로필은 train 평점(npz에 저장됨)을 그대로 쓴다.
가중치는 공통 Validation A로 학습한다. (hybrid_validation)
Validation B 예측은 하이브리드 α 선택용으로 저장한다.
같은 도메인 채점은 영화 test, 교차 도메인은 도서 test.
책 train·validation은 채점에 넣지 않는다.

피처 4개 (클수록 유사): 장르 코사인, 줄거리 코사인, 러닝타임/페이지 근접도, 연도 근접도

실행: python scripts/cross_domain_eval.py
결과는 results/content_based/cross_domain_eval/ 에 저장한다.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# ---- 경로 설정 (파일 위치에 맞게 조정) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "content_based" / "cross_domain_eval"
HYBRID_SPLIT_NPZ = DATA_DIR / "hybrid_validation" / "validation_split_indices.npz"
MOVIE_ITEM_NPZ = DATA_DIR / "movie_item_user_data.npz"

# ---- 추천 설정 ----
REC_DOMAIN = "book"   # 추천받을 도메인: "movie" 또는 "book"
REC_USER_ID = None    # None이면 해당 도메인 Test의 첫 번째 유저
TOP_K = 10

FEATURE_NAMES = ["genre_cos", "plot_cos", "length_closeness", "year_closeness"]
CHUNK = 50_000  # 줄거리 임베딩이 (N, 1024)라 청크 단위로 계산


# ------------------------------------------------------------
# 로딩
# ------------------------------------------------------------

def load_sparse(data, prefix):
    """npz에 CSR 구성요소로 저장된 희소행렬 복원."""
    return csr_matrix(
        (data[f"{prefix}_data"], data[f"{prefix}_indices"], data[f"{prefix}_indptr"]),
        shape=tuple(data[f"{prefix}_shape"]),
    )


def load_domain(name, item_key):
    """한 도메인(movie/book)의 아이템 특성과 유저 프로필을 하나의 dict로 묶는다."""
    with np.load(DATA_DIR / f"{name}_content.npz", allow_pickle=True) as f:
        content = {k: f[k] for k in f.files}
    with np.load(DATA_DIR / f"{name}_user_profiles.npz", allow_pickle=True) as f:
        profile = {k: f[k] for k in f.files}

    return {
        "item_key": item_key,                       # 'movie_ids' 또는 'book_ids'
        "item_ids": content[item_key].astype(np.int64),
        # 제목 (없으면 빈 문자열)
        "item_titles": (
            content["titles"].astype(str)
            if "titles" in content
            else np.full(len(content[item_key]), "", dtype=str)
        ),
        "item_genres": load_sparse(content, "genres"),
        "item_plot": content["plot_embeddings"].astype(np.float32),
        "item_numeric": content["numeric_features"].astype(np.float32),
        "user_ids": profile["user_ids"].astype(np.int64),
        "user_genres": load_sparse(profile, "user_genre_profiles"),
        "user_plot": profile["plot_profiles"].astype(np.float32),
        "user_numeric": profile["numeric_profiles"].astype(np.float32),
        "profile": profile,
    }


# ------------------------------------------------------------
# 피처 계산
# ------------------------------------------------------------

def pair_features(domain, u_idx, i_idx):
    """(유저 행 위치, 아이템 행 위치) 쌍마다 피처 4개를 계산. 평가/추천이 공통으로 사용."""
    X = np.zeros((len(u_idx), 4), dtype=np.float32)
    for s in range(0, len(u_idx), CHUNK):
        sl = slice(s, s + CHUNK)
        u, i = u_idx[sl], i_idx[sl]

        # x1: 장르 코사인 (양쪽 L2 정규화 완료 -> 원소곱의 합 = 코사인)
        X[sl, 0] = np.asarray(domain["user_genres"][u].multiply(domain["item_genres"][i]).sum(axis=1)).ravel()

        # x2: 줄거리 임베딩 코사인
        eu, ei = domain["user_plot"][u], domain["item_plot"][i]
        norm = np.linalg.norm(eu, axis=1) * np.linalg.norm(ei, axis=1)
        X[sl, 1] = np.einsum("ij,ij->i", eu, ei) / (norm + 1e-8)

        # x3: 러닝타임(영화) / 페이지(도서) 근접도,  x4: 연도 근접도
        X[sl, 2] = 1.0 - np.abs(domain["user_numeric"][u, 0] - domain["item_numeric"][i, 0])
        X[sl, 3] = 1.0 - np.abs(domain["user_numeric"][u, 1] - domain["item_numeric"][i, 1])
    return X


def build_features(domain, split):
    """split의 (유저, 아이템, 평점) 한 건마다 피처를 계산해 (X, y) 반환."""
    prof = domain["profile"]
    user_ids = prof[f"{split}_user_ids"].astype(np.int64)
    item_ids = prof[f"{split}_{domain['item_key']}"].astype(np.int64)
    y = prof[f"{split}_ratings"].astype(np.float32)

    # ID -> 행 위치 (ID 배열은 정렬되어 있다고 가정)
    u_idx = np.searchsorted(domain["user_ids"], user_ids)
    i_idx = np.searchsorted(domain["item_ids"], item_ids)
    return pair_features(domain, u_idx, i_idx), y, user_ids, item_ids


def load_hybrid_val_part(domain, part):
    """movie_item_user_data.npz Validation 행을 A/B 인덱스로 고른다."""
    with np.load(HYBRID_SPLIT_NPZ) as f:
        indices = f[f"{part}_indices"]
    with np.load(MOVIE_ITEM_NPZ, allow_pickle=False) as src:
        user_ids = src["validation_user_ids"][indices].astype(np.int64)
        item_ids = src["validation_movie_ids"][indices].astype(np.int64)
        y = src["validation_ratings"][indices].astype(np.float32)
    u_idx = np.searchsorted(domain["user_ids"], user_ids)
    i_idx = np.searchsorted(domain["item_ids"], item_ids)
    if np.any(domain["user_ids"][u_idx] != user_ids):
        raise ValueError("Validation 유저가 콘텐츠 프로필에 없습니다.")
    if np.any(domain["item_ids"][i_idx] != item_ids):
        raise ValueError("Validation 아이템이 콘텐츠 목록에 없습니다.")
    return u_idx, i_idx, y, user_ids, item_ids


def save_predictions(path, user_ids, item_ids, rating, predicted):
    pd.DataFrame({
        "user_id": np.asarray(user_ids, np.int64),
        "item_id": np.asarray(item_ids, np.int64),
        "rating": np.asarray(rating, float),
        "predicted_rating": np.clip(np.asarray(predicted, float), 0.5, 5.0),
    }).to_csv(path, index=False)


# ------------------------------------------------------------
# 추천
# ------------------------------------------------------------

def recommend_top_k(model, domain, user_id, k=TOP_K):
    """한 유저에 대해 전체 아이템 점수를 계산하고, 점수가 높은(=유사한) 상위 k개를 반환.
    이미 Train에서 평점을 남긴 아이템은 제외한다."""
    # 유저 행 위치 확인
    u_pos = int(np.searchsorted(domain["user_ids"], user_id))
    if u_pos >= len(domain["user_ids"]) or domain["user_ids"][u_pos] != user_id:
        raise ValueError(f"유저 {user_id}의 프로필이 없습니다.")

    # 모든 아이템과의 피처 -> 모델 점수
    n_items = len(domain["item_ids"])
    X = pair_features(domain, np.full(n_items, u_pos), np.arange(n_items))
    scores = model.predict(X)

    # 이미 본(Train 평점) 아이템 제외
    prof = domain["profile"]
    seen_mask = prof["train_user_ids"].astype(np.int64) == user_id
    seen_items = prof[f"train_{domain['item_key']}"].astype(np.int64)[seen_mask]
    scores[np.isin(domain["item_ids"], seen_items)] = -np.inf

    # 상위 k개 (점수 내림차순)
    top = np.argsort(-scores)[:k]
    out = pd.DataFrame(X[top], columns=FEATURE_NAMES)
    out.insert(0, "item_id", domain["item_ids"][top])
    out.insert(1, "title", domain["item_titles"][top])
    out.insert(2, "score", scores[top])
    return out.reset_index(drop=True)


# ------------------------------------------------------------
# 평가
# ------------------------------------------------------------

def metrics(y_true, y_pred):
    """RMSE / MAE / Pearson."""
    err = y_pred - y_true
    corr = pearsonr(y_pred, y_true)[0] if np.std(y_pred) > 0 else float("nan")
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "pearson": float(corr),
    }


def report(title, m):
    print(f"  {title:<28} RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  pearson={m['pearson']:.4f}")


def json_ready(obj):
    """numpy / NaN 값을 JSON으로 쓸 수 있게 바꾼다."""
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def save_run(output_dir, coefficients, metrics_rows, recommendations, summary):
    """계수, 지표, 추천, 실행 요약을 results/content_based 아래에 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coefficients).to_csv(output_dir / "coefficients.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics.csv", index=False)
    recommendations.to_csv(
        output_dir / f"recommendations_user_{summary['user_id']}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "run_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n저장: {output_dir}")


def main():
    movie = load_domain("movie", "movie_ids")
    book = load_domain("book", "book_ids")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 영화 Validation A로 가중치 학습. 프로필은 train 그대로.
    u_fit, i_fit, y_fit, _, _ = load_hybrid_val_part(movie, "a")
    model = LinearRegression(positive=True).fit(pair_features(movie, u_fit, i_fit), y_fit)
    coefs = {n: round(float(w), 3) for n, w in zip(FEATURE_NAMES, model.coef_)}
    print("계수:", coefs, "b =", round(float(model.intercept_), 3))
    print("학습 행:", len(y_fit), "(영화 Validation A)")

    u_b, i_b, y_b, uid_b, iid_b = load_hybrid_val_part(movie, "b")
    pred_b = model.predict(pair_features(movie, u_b, i_b))
    save_predictions(OUTPUT_DIR / "movie_val_b_predictions.csv", uid_b, iid_b, y_b, pred_b)

    # 2) 영화 Test / 도서 Test 평가 (predict만, 재학습 없음)
    X_mt, y_mt, uid_mt, iid_mt = build_features(movie, "test")
    X_bt, y_bt, uid_bt, iid_bt = build_features(book, "test")
    movie_mean = float(movie["profile"]["train_ratings"].mean())
    book_mean = float(book["profile"]["train_ratings"].mean())
    pred_mt = model.predict(X_mt)
    pred_bt = model.predict(X_bt)
    save_predictions(OUTPUT_DIR / "movie_test_predictions.csv", uid_mt, iid_mt, y_mt, pred_mt)
    save_predictions(OUTPUT_DIR / "book_test_predictions.csv", uid_bt, iid_bt, y_bt, pred_bt)
    shift_mt = pred_mt - pred_mt.mean() + movie_mean
    shift_bt = pred_bt - pred_bt.mean() + book_mean

    print("\n[평균 편향 확인] (예측 평균 / 실제 Test 평균 / 도메인 Train 평균)")
    print(f"  영화: {pred_mt.mean():.3f} / {y_mt.mean():.3f} / {movie_mean:.3f}")
    print(f"  도서: {pred_bt.mean():.3f} / {y_bt.mean():.3f} / {book_mean:.3f}")

    metric_rows = [
        {"name": "영화 Validation B (모델)", **metrics(y_b, pred_b)},
        {"name": "영화 Test (모델)", **metrics(y_mt, pred_mt)},
        {"name": "영화 Test (모델, 평균 보정)", **metrics(y_mt, shift_mt)},
        {"name": "영화 Test (평균 베이스라인)", **metrics(y_mt, np.full_like(y_mt, movie_mean))},
        {"name": "도서 Test (모델)", **metrics(y_bt, pred_bt)},
        {"name": "도서 Test (모델, 평균 보정)", **metrics(y_bt, shift_bt)},
        {"name": "도서 Test (평균 베이스라인)", **metrics(y_bt, np.full_like(y_bt, book_mean))},
    ]
    print("\n[영화 Validation A 학습 결과 비교]")
    for row in metric_rows:
        report(row["name"], row)

    # 3) Top-K 추천
    domain = movie if REC_DOMAIN == "movie" else book
    user_id = REC_USER_ID
    if user_id is None:
        user_id = int(domain["profile"]["test_user_ids"][0])
    recs = recommend_top_k(model, domain, user_id)
    print(f"\n[{REC_DOMAIN} 추천 Top-{TOP_K}] 유저 {user_id}")
    print(recs.round(3).to_string())

    coefficients = [{"feature": n, "weight": float(w)} for n, w in zip(FEATURE_NAMES, model.coef_)]
    coefficients.append({"feature": "intercept", "weight": float(model.intercept_)})
    save_run(
        OUTPUT_DIR,
        coefficients,
        metric_rows,
        recs,
        {
            "script": "cross_domain_eval.py",
            "n_fit": int(len(y_fit)),
            "n_val_b": int(len(y_b)),
            "fit_split": "movie validation A",
            "alpha_split": "movie validation B",
            "eval": "movie test + book test",
            "movie_mean": movie_mean,
            "coefficients": coefs,
            "intercept": float(model.intercept_),
            "pred_mean": float(pred_bt.mean()),
            "true_mean": float(y_bt.mean()),
            "book_train_mean": book_mean,
            "rec_domain": REC_DOMAIN,
            "user_id": int(user_id),
            "top_k": TOP_K,
            "metrics": metric_rows,
        },
    )


if __name__ == "__main__":
    main()

"""
계수: {'genre_cos': 0.707, 'plot_cos': 0.866, 'length_closeness': 0.0, 'year_closeness': 0.0} b = 2.77
학습 행: 1087586 (영화 Validation A)

[평균 편향 확인] (예측 평균 / 실제 Test 평균 / 도메인 Train 평균)
  영화: 3.569 / 3.571 / 3.571
  도서: 3.807 / 3.921 / 3.920
/Users/chaehee/Documents/school/ML/machine_learning_term_project/scripts/cross_domain_eval.py:182: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  corr = pearsonr(y_pred, y_true)[0] if np.std(y_pred) > 0 else float("nan")

[영화 Validation A 학습 결과 비교]
  영화 Validation B (모델)         RMSE=0.9877  MAE=0.7813  pearson=0.3525
  영화 Test (모델)                 RMSE=0.9861  MAE=0.7798  pearson=0.3535
  영화 Test (모델, 평균 보정)          RMSE=0.9861  MAE=0.7797  pearson=0.3535
  영화 Test (평균 베이스라인)           RMSE=1.0541  MAE=0.8360  pearson=nan
  도서 Test (모델)                 RMSE=0.9716  MAE=0.7848  pearson=0.2209
  도서 Test (모델, 평균 보정)          RMSE=0.9649  MAE=0.7604  pearson=0.2209
  도서 Test (평균 베이스라인)           RMSE=0.9893  MAE=0.7733  pearson=nan

[book 추천 Top-10] 유저 13478
   item_id                                         title  score  genre_cos  plot_cos  length_closeness  year_closeness
0      212                                     Atonement  3.995      0.820     0.746             0.989           0.752
1     6335                             Oscar and Lucinda  3.981      0.820     0.731             0.979           0.796
2     9627                 The Hand That First Held Mine  3.972      0.785     0.748             0.987           0.690
3     6090                          The Baker's Daughter  3.966      0.785     0.742             0.975           0.672
4      727                                    The Reader  3.962      0.761     0.757             0.963           0.805
5      689                               Beautiful Ruins  3.959      0.785     0.733             0.987           0.672
6     1566                                   Ethan Frome  3.954      0.761     0.747             0.941           0.734
7     1565                      Death Comes to Pemberley  3.953      0.783     0.728             0.978           0.681
8     4438                            Dear Life: Stories  3.953      0.720     0.780             0.986           0.672
9     6948  Sense & Sensibility (The Austen Project, #1)  3.953      0.776     0.734             0.991           0.663

저장: /Users/chaehee/Documents/school/ML/machine_learning_term_project/results/content_based/cross_domain_eval
"""