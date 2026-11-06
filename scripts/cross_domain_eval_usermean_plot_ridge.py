"""영화로 학습한 Ridge -> 영화 Test / 도서 Test 평가 + 유저별 Top-K 추천

scripts/cross_domain_eval_usermean.py 복제. 바꾼 점:
  줄거리 코사인 1칸을, 정규화 후 원소곱 1024칸으로 펼친다.
  장르 코사인 / 길이 근접 / 연도 근접은 그대로다.
  w는 영화 validation 잔차로 비음수 Ridge (배치 X'X, w≥0, 절편은 자유)로 맞춘다.
  같은 도메인 채점은 영화 test, 교차 도메인은 도서 test.

실행: python scripts/cross_domain_eval_usermean_plot_ridge.py
결과는 results/content_based/cross_domain_eval_usermean_plot_ridge/ 에 저장한다.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import lsq_linear
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr

# ---- 경로 설정 (파일 위치에 맞게 조정) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "content_based" / "cross_domain_eval_usermean_plot_ridge"

# ---- 추천 설정 ----
REC_DOMAIN = "book"   # 추천받을 도메인: "movie" 또는 "book"
REC_USER_ID = None    # None이면 해당 도메인 Test의 첫 번째 유저
TOP_K = 10

FEATURE_NAMES = ["genre_cos", "plot_cos", "length_closeness", "year_closeness"]
CHUNK = 50_000  # 줄거리 임베딩이 (N, 1024)라 청크 단위로 계산
RIDGE_ALPHA = 10.0


class RidgeModel:
    """비음수 Ridge. sklearn과 같은 predict 인터페이스."""

    def __init__(self, coef, intercept):
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = float(intercept)

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_


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

def n_plot_dim(domain):
    return int(domain["item_plot"].shape[1])


def pair_features(domain, u_idx, i_idx):
    """(유저 행 위치, 아이템 행 위치) 쌍마다 피처를 계산.

    칸: [genre_cos | plot 원소곱 1024칸 | length_closeness | year_closeness]
    줄거리 1024칸의 합 = 원래 plot_cos.
    """
    n_plot = n_plot_dim(domain)
    X = np.zeros((len(u_idx), 3 + n_plot), dtype=np.float32)
    for s in range(0, len(u_idx), CHUNK):
        sl = slice(s, s + CHUNK)
        u, i = u_idx[sl], i_idx[sl]

        # x1: 장르 코사인 (양쪽 L2 정규화 완료 -> 원소곱의 합 = 코사인)
        X[sl, 0] = np.asarray(domain["user_genres"][u].multiply(domain["item_genres"][i]).sum(axis=1)).ravel()

        # 줄거리: 코사인의 항을 더하지 않고 칸마다 둠. 합하면 원래 plot_cos.
        eu, ei = domain["user_plot"][u], domain["item_plot"][i]
        norm = (np.linalg.norm(eu, axis=1) * np.linalg.norm(ei, axis=1))[:, None]
        X[sl, 1:1 + n_plot] = eu * ei / (norm + 1e-8)

        # 러닝타임(영화) / 페이지(도서) 근접도, 연도 근접도
        X[sl, 1 + n_plot] = 1.0 - np.abs(domain["user_numeric"][u, 0] - domain["item_numeric"][i, 0])
        X[sl, 2 + n_plot] = 1.0 - np.abs(domain["user_numeric"][u, 1] - domain["item_numeric"][i, 1])
    return X


def split_rows(domain, split):
    """split의 (유저 위치, 아이템 위치, 평점). X는 만들지 않는다."""
    prof = domain["profile"]
    user_ids = prof[f"{split}_user_ids"].astype(np.int64)
    item_ids = prof[f"{split}_{domain['item_key']}"].astype(np.int64)
    y = prof[f"{split}_ratings"].astype(np.float32)
    u_idx = np.searchsorted(domain["user_ids"], user_ids)
    i_idx = np.searchsorted(domain["item_ids"], item_ids)
    return u_idx, i_idx, y


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


def fit_ridge_batched(domain, u_idx, i_idx, y, alpha=RIDGE_ALPHA):
    """전체 X를 만들지 않고 비음수 Ridge를 푼다.

    손실: ||y - b - Xw||^2 + α||w||^2, 제약: w ≥ 0.
    절편 b는 규제하지 않고 음수도 허용한다. (LinearRegression(positive=True)와 같음)
    """
    n_features = 3 + n_plot_dim(domain)
    dim = n_features + 1
    xtx = np.zeros((dim, dim), dtype=np.float64)
    xty = np.zeros(dim, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    for s in range(0, len(u_idx), CHUNK):
        sl = slice(s, s + CHUNK)
        X = pair_features(domain, u_idx[sl], i_idx[sl]).astype(np.float64, copy=False)
        xtx[0, 0] += X.shape[0]
        xtx[0, 1:] += X.sum(axis=0)
        xtx[1:, 1:] += X.T @ X
        xty[0] += y[sl].sum()
        xty[1:] += X.T @ y[sl]
    xtx[1:, 0] = xtx[0, 1:]
    xtx.flat[:: dim + 1] += alpha
    xtx[0, 0] -= alpha
    # min ||R θ - rhs||^2  s.t. w≥0.  H = R'R, 제약 없는 Ridge는 Hθ = X'y.
    R = cholesky(xtx, lower=False, check_finite=False)
    rhs = solve_triangular(R, xty, trans="T", lower=False, check_finite=False)
    lb = np.zeros(dim)
    lb[0] = -np.inf
    sol = lsq_linear(R, rhs, bounds=(lb, np.inf), lsmr_tol="auto")
    return RidgeModel(sol.x[1:], sol.x[0])


def predict_residual(model, domain, u_idx, i_idx):
    """잔차 예측을 청크로 이어 붙인다."""
    out = np.empty(len(u_idx), dtype=np.float64)
    for s in range(0, len(u_idx), CHUNK):
        sl = slice(s, s + CHUNK)
        out[sl] = model.predict(pair_features(domain, u_idx[sl], i_idx[sl]))
    return out


def display_features(X):
    """학습 피처 1027칸을 Top-K용 4칸으로 접는다. plot_cos = 줄거리 칸의 합."""
    n_plot = X.shape[1] - 3
    return np.column_stack([
        X[:, 0],
        X[:, 1:1 + n_plot].sum(axis=1),
        X[:, 1 + n_plot],
        X[:, 2 + n_plot],
    ])


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

    # 모든 아이템과의 피처 -> 모델 점수 (1027칸)
    n_items = len(domain["item_ids"])
    X = pair_features(domain, np.full(n_items, u_pos), np.arange(n_items))
    scores = means[u_pos] + model.predict(X)

    # 이미 본(Train 평점) 아이템 제외
    prof = domain["profile"]
    seen_mask = prof["train_user_ids"].astype(np.int64) == user_id
    seen_items = prof[f"train_{domain['item_key']}"].astype(np.int64)[seen_mask]
    scores[np.isin(domain["item_ids"], seen_items)] = -np.inf

    # 상위 k개 (점수 내림차순). 표는 기존 4칸만.
    top = np.argsort(-scores)[:k]
    out = pd.DataFrame(display_features(X[top]), columns=FEATURE_NAMES)
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
    n_plot = n_plot_dim(movie)

    # 1) 영화 validation으로 잔차 학습. 프로필·평균은 train.
    m_mean = train_user_means(movie)
    u_fit, i_fit, y_fit_raw = split_rows(movie, "validation")
    y_fit = y_fit_raw - m_mean[u_fit]
    model = fit_ridge_batched(movie, u_fit, i_fit, y_fit)

    w = model.coef_
    w_plot = w[1:1 + n_plot]
    coefs = {
        "genre_cos": round(float(w[0]), 3),
        "plot_w_mean": round(float(w_plot.mean()), 4),
        "plot_w_l2": round(float(np.linalg.norm(w_plot)), 3),
        "length_closeness": round(float(w[1 + n_plot]), 3),
        "year_closeness": round(float(w[2 + n_plot]), 3),
    }
    print("예측 = 유저 train 평균 + b + w·(장르 + 줄거리칸 + 길이 + 연도),  w ≥ 0")
    print("계수:", coefs, "b =", round(float(model.intercept_), 3))
    print(f"줄거리 칸: {n_plot}  alpha={RIDGE_ALPHA}  positive=True (절편 제외)")
    print("학습 행:", len(y_fit), "(영화 validation)")

    # 2) 영화 Test / 도서 Test. w는 고정, 평균은 각 도메인 유저 train.
    b_mean = train_user_means(book)
    u_mt, i_mt, y_mt = split_rows(movie, "test")
    u_bt, i_bt, y_bt = split_rows(book, "test")
    pred_mt = m_mean[u_mt] + predict_residual(model, movie, u_mt, i_mt)
    pred_bt = b_mean[u_bt] + predict_residual(model, book, u_bt, i_bt)
    movie_mean = float(movie["profile"]["train_ratings"].mean())
    book_mean = float(book["profile"]["train_ratings"].mean())
    shift_mt = pred_mt - pred_mt.mean() + movie_mean
    shift_bt = pred_bt - pred_bt.mean() + book_mean

    print("\n[평균 편향 확인] (예측 평균 / 실제 Test 평균 / 도메인 Train 평균)")
    print(f"  영화: {pred_mt.mean():.3f} / {y_mt.mean():.3f} / {movie_mean:.3f}")
    print(f"  도서: {pred_bt.mean():.3f} / {y_bt.mean():.3f} / {book_mean:.3f}")

    metric_rows = [
        {"name": "영화 Test (모델)", **metrics(y_mt, pred_mt)},
        {"name": "영화 Test (모델, 평균 보정)", **metrics(y_mt, shift_mt)},
        {"name": "영화 Test (평균 베이스라인)", **metrics(y_mt, np.full_like(y_mt, movie_mean))},
        {"name": "영화 Test (유저 평균 베이스라인)", **metrics(y_mt, m_mean[u_mt].astype(np.float32))},
        {"name": "도서 Test (모델)", **metrics(y_bt, pred_bt)},
        {"name": "도서 Test (모델, 평균 보정)", **metrics(y_bt, shift_bt)},
        {"name": "도서 Test (평균 베이스라인)", **metrics(y_bt, np.full_like(y_bt, book_mean))},
        {"name": "도서 Test (유저 평균 베이스라인)", **metrics(y_bt, b_mean[u_bt].astype(np.float32))},
    ]
    print("\n[영화(validation) Ridge 학습 결과 비교]")
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

    coefficients = [
        {"feature": "genre_cos", "weight": float(w[0])},
        {"feature": "plot_w_mean", "weight": float(w_plot.mean())},
        {"feature": "plot_w_l2", "weight": float(np.linalg.norm(w_plot))},
        {"feature": "length_closeness", "weight": float(w[1 + n_plot])},
        {"feature": "year_closeness", "weight": float(w[2 + n_plot])},
        {"feature": "intercept", "weight": float(model.intercept_)},
    ]
    save_run(
        OUTPUT_DIR,
        coefficients,
        metric_rows,
        recs,
        {
            "script": "cross_domain_eval_usermean_plot_ridge.py",
            "n_fit": int(len(y_fit)),
            "n_plot": int(n_plot),
            "ridge_alpha": RIDGE_ALPHA,
            "positive_coef": True,
            "fit_split": "movie validation residual",
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
예측 = 유저 train 평균 + b + w·(장르 + 줄거리칸 + 길이 + 연도),  w ≥ 0
계수: {'genre_cos': 0.618, 'plot_w_mean': 0.3119, 'plot_w_l2': 21.384, 'length_closeness': 0.0, 'year_closeness': 0.0} b = -0.355
줄거리 칸: 1024  alpha=10.0  positive=True (절편 제외)
학습 행: 2175173 (영화 validation)

[평균 편향 확인] (예측 평균 / 실제 Test 평균 / 도메인 Train 평균)
  영화: 3.570 / 3.571 / 3.571
  도서: 4.055 / 3.921 / 3.920
/Users/chaehee/Documents/school/ML/machine_learning_term_project/scripts/cross_domain_eval_usermean_plot_ridge.py:239: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  corr = pearsonr(y_pred, y_true)[0] if np.std(y_pred) > 0 else float("nan")

[영화(validation) Ridge 학습 결과 비교]
  영화 Test (모델)                 RMSE=0.9333  MAE=0.7188  pearson=0.4662
  영화 Test (모델, 평균 보정)          RMSE=0.9333  MAE=0.7188  pearson=0.4662
  영화 Test (평균 베이스라인)           RMSE=1.0541  MAE=0.8360  pearson=nan
  영화 Test (유저 평균 베이스라인)        RMSE=0.9443  MAE=0.7279  pearson=0.4449
  도서 Test (모델)                 RMSE=0.8999  MAE=0.6937  pearson=0.4398
  도서 Test (모델, 평균 보정)          RMSE=0.8898  MAE=0.6977  pearson=0.4398
  도서 Test (평균 베이스라인)           RMSE=0.9893  MAE=0.7733  pearson=nan
  도서 Test (유저 평균 베이스라인)        RMSE=0.8917  MAE=0.6999  pearson=0.4336

[book 추천 Top-10] 유저 13478
   item_id                           title  score  genre_cos  plot_cos  length_closeness  year_closeness
0      212                       Atonement  3.764      0.820     0.746             0.989           0.752
1     3465              Rich Man, Poor Man  3.759      0.820     0.629             0.950           0.965
2     6335               Oscar and Lucinda  3.755      0.820     0.731             0.979           0.796
3     2769  Z: A Novel of Zelda Fitzgerald  3.751      0.797     0.704             0.989           0.663
4     1957            I Capture the Castle  3.744      0.776     0.708             1.000           0.796
5      867             The History of Love  3.744      0.785     0.706             0.971           0.725
6     1758                        Birdsong  3.742      0.761     0.705             0.985           0.805
7     9483                 Any Human Heart  3.740      0.761     0.654             0.986           0.743
8     6467     The Women of Brewster Place  3.739      0.761     0.686             0.959           0.929
9     9627   The Hand That First Held Mine  3.738      0.785     0.748             0.987           0.690

저장: /Users/chaehee/Documents/school/ML/machine_learning_term_project/results/content_based/cross_domain_eval_usermean_plot_ridge
"""