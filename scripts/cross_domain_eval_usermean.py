"""영화로 학습한 선형 모델 -> 영화 Test / 도서 val+test 평가 + 유저별 Top-K 추천

원본 scripts/cross_domain_eval.py 복제. 바꾼 점 하나:
  예측 = 그 유저 train 평균 + b + w·유사도
  w는 영화 validation에서 (평점 − 영화유저 평균) 잔차를 맞춘다.
  같은 도메인 채점은 영화 test, 교차 도메인은 도서 val+test.
  책에서는 책 유저 train 평균을 더한다. w는 그대로.

실행: python scripts/cross_domain_eval_usermean.py
결과는 results/content_based/cross_domain_eval_usermean/ 에 저장한다.
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
OUTPUT_DIR = PROJECT_ROOT / "results" / "content_based" / "cross_domain_eval_usermean"

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
    return pair_features(domain, u_idx, i_idx), y, u_idx


def train_user_means(domain):
    """train 평점의 유저별 평균. 평점 없는 유저는 도메인 train 평균."""
    prof = domain["profile"]
    raw_u = prof["train_user_ids"].astype(np.int64)
    r = prof["train_ratings"].astype(np.float64)
    u_idx = np.searchsorted(domain["user_ids"], raw_u)
    n = len(domain["user_ids"])
    s = np.zeros(n, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    np.add.at(s, u_idx, r)
    np.add.at(c, u_idx, 1.0)
    out = np.full(n, float(r.mean()))
    ok = c > 0
    out[ok] = s[ok] / c[ok]
    return out


# ------------------------------------------------------------
# 추천
# ------------------------------------------------------------

def recommend_top_k(model, domain, user_id, means, k=TOP_K):
    """한 유저에 대해 전체 아이템 점수를 계산하고, 점수가 높은(=유사한) 상위 k개를 반환.
    이미 Train에서 평점을 남긴 아이템은 제외한다."""
    # 유저 행 위치 확인
    u_pos = int(np.searchsorted(domain["user_ids"], user_id))
    if u_pos >= len(domain["user_ids"]) or domain["user_ids"][u_pos] != user_id:
        raise ValueError(f"유저 {user_id}의 프로필이 없습니다.")

    # 모든 아이템과의 피처 -> 모델 점수
    n_items = len(domain["item_ids"])
    X = pair_features(domain, np.full(n_items, u_pos), np.arange(n_items))
    scores = means[u_pos] + model.predict(X)

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

    # 1) 영화 validation으로 잔차 학습. 프로필·평균은 train.
    m_mean = train_user_means(movie)
    X_fit, y_fit_raw, u_fit = build_features(movie, "validation")
    y_fit = y_fit_raw - m_mean[u_fit]
    model = LinearRegression(positive=True).fit(X_fit, y_fit)

    coefs = {n: round(float(w), 3) for n, w in zip(FEATURE_NAMES, model.coef_)}
    print("예측 = 유저 train 평균 + b + w·유사도")
    print("계수:", coefs, "b =", round(float(model.intercept_), 3))
    print("학습 행:", len(y_fit), "(영화 validation)")

    # 2) 영화 test(같은 도메인) / 도서 val+test(교차 도메인). w는 고정.
    b_mean = train_user_means(book)
    X_mt, y_mt, u_mt = build_features(movie, "test")
    X_bv, y_bv, u_bv = build_features(book, "validation")
    X_bt, y_bt, u_bt = build_features(book, "test")
    pred_mt = m_mean[u_mt] + model.predict(X_mt)
    pred_bv = b_mean[u_bv] + model.predict(X_bv)
    pred_bt = b_mean[u_bt] + model.predict(X_bt)
    y_be = np.concatenate([y_bv, y_bt])
    pred_be = np.concatenate([pred_bv, pred_bt])
    movie_mean = float(movie["profile"]["train_ratings"].mean())
    book_mean = float(book["profile"]["train_ratings"].mean())
    shift_mt = pred_mt - pred_mt.mean() + movie_mean
    shift_be = pred_be - pred_be.mean() + book_mean
    user_mean_base = np.concatenate([b_mean[u_bv], b_mean[u_bt]]).astype(np.float32)

    print("\n[평균 편향 확인] (예측 평균 / 실제 평균 / 도메인 Train 평균)")
    print(f"  영화 test: {pred_mt.mean():.3f} / {y_mt.mean():.3f} / {movie_mean:.3f}")
    print(f"  도서 val+test: {pred_be.mean():.3f} / {y_be.mean():.3f} / {book_mean:.3f}")

    metric_rows = [
        {"name": "영화 test (모델)", **metrics(y_mt, pred_mt)},
        {"name": "영화 test (평균 보정)", **metrics(y_mt, shift_mt)},
        {"name": "영화 test (평균 베이스라인)", **metrics(y_mt, np.full_like(y_mt, movie_mean))},
        {"name": "영화 test (유저 평균 베이스라인)", **metrics(y_mt, m_mean[u_mt].astype(np.float32))},
        {"name": "도서 val (모델)", **metrics(y_bv, pred_bv)},
        {"name": "도서 test (모델)", **metrics(y_bt, pred_bt)},
        {"name": "도서 val+test (모델)", **metrics(y_be, pred_be)},
        {"name": "도서 val+test (평균 보정)", **metrics(y_be, shift_be)},
        {"name": "도서 val+test (평균 베이스라인)", **metrics(y_be, np.full_like(y_be, book_mean))},
        {"name": "도서 val+test (유저 평균 베이스라인)", **metrics(y_be, user_mean_base)},
    ]
    print("\n[영화 validation 잔차 학습 → 영화 test / 도서 val+test]")
    for row in metric_rows:
        report(row["name"], row)

    # 3) Top-K 추천
    domain = movie if REC_DOMAIN == "movie" else book
    rec_means = m_mean if REC_DOMAIN == "movie" else b_mean
    user_id = REC_USER_ID
    if user_id is None:
        user_id = int(domain["profile"]["test_user_ids"][0])
    recs = recommend_top_k(model, domain, user_id, rec_means)
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
            "script": "cross_domain_eval_usermean.py",
            "n_fit": int(len(y_fit)),
            "fit_split": "movie validation residual",
            "eval": "movie test + book val+test",
            "movie_mean": movie_mean,
            "coefficients": coefs,
            "intercept": float(model.intercept_),
            "pred_mean": float(pred_be.mean()),
            "true_mean": float(y_be.mean()),
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
예측 = 유저 train 평균 + b + w·유사도
계수: {'genre_cos': 0.639, 'plot_cos': 0.0, 'length_closeness': 0.0, 'year_closeness': 0.0} b = -0.328
학습 행: 2175173 (영화 validation)

[평균 편향 확인] (예측 평균 / 실제 평균 / 도메인 Train 평균)
  영화 test: 3.570 / 3.572 / 3.571
  도서 val+test: 4.022 / 3.921 / 3.920
/Users/chaehee/Documents/school/ML/machine_learning_term_project/scripts/cross_domain_eval_usermean.py:168: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  corr = pearsonr(y_pred, y_true)[0] if np.std(y_pred) > 0 else float("nan")

[영화 validation 잔차 학습 → 영화 test / 도서 val+test]
  영화 test (모델)                 RMSE=0.9358  MAE=0.7215  pearson=0.4607
  영화 test (평균 보정)              RMSE=0.9358  MAE=0.7214  pearson=0.4607
  영화 test (평균 베이스라인)           RMSE=1.0538  MAE=0.8358  pearson=nan
  영화 test (유저 평균 베이스라인)        RMSE=0.9442  MAE=0.7279  pearson=0.4447
  도서 val (모델)                  RMSE=0.9004  MAE=0.6966  pearson=0.4332
  도서 test (모델)                 RMSE=0.8970  MAE=0.6946  pearson=0.4363
  도서 val+test (모델)             RMSE=0.8987  MAE=0.6956  pearson=0.4348
  도서 val+test (평균 보정)          RMSE=0.8930  MAE=0.7003  pearson=0.4348
  도서 val+test (평균 베이스라인)       RMSE=0.9903  MAE=0.7740  pearson=nan
  도서 val+test (유저 평균 베이스라인)    RMSE=0.8932  MAE=0.7007  pearson=0.4325

[book 추천 Top-10] 유저 13478
   item_id                           title  score  genre_cos  plot_cos  length_closeness  year_closeness
0     6335               Oscar and Lucinda  3.704      0.820     0.731             0.979           0.796
1     3465              Rich Man, Poor Man  3.704      0.820     0.629             0.950           0.965
2      212                       Atonement  3.704      0.820     0.746             0.989           0.752
3     2769  Z: A Novel of Zelda Fitzgerald  3.690      0.797     0.704             0.989           0.663
4      198                The Color Purple  3.687      0.792     0.674             0.979           0.743
5      689                 Beautiful Ruins  3.682      0.785     0.733             0.987           0.672
6     2029                 On Chesil Beach  3.682      0.785     0.707             0.954           0.717
7     5241                            Silk  3.682      0.785     0.688             0.939           0.796
8     5050     The Story of Beautiful Girl  3.682      0.785     0.723             0.988           0.681
9     6090            The Baker's Daughter  3.682      0.785     0.742             0.975           0.672

저장: /Users/chaehee/Documents/school/ML/machine_learning_term_project/results/content_based/cross_domain_eval_usermean
"""