"""영화로 학습한 선형 모델 -> 영화 Test / 도서 Test 비교 + 유저별 Top-K 추천

비교 대상
  A) 영화 학습 -> 영화 Test  (같은 도메인)
  B) 영화 학습 -> 도서 Test  (교차 도메인)
추천
  학습된 모델로 한 유저의 전체 아이템 점수를 계산해 상위 K개를 뽑는다.

피처 4개 (클수록 유사): 장르 코사인, 줄거리 코사인, 러닝타임/페이지 근접도, 연도 근접도

실행: python cross_domain_eval_recommend.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# ---- 경로 설정 (파일 위치에 맞게 조정) ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# 가중치를 학습할 영화 split.
# 유저 프로필이 Train 평점으로 만들어지므로, "train"을 쓰면 피처에 정답이 섞여 누수가 생긴다.
FIT_SPLIT = "validation"

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
    return pair_features(domain, u_idx, i_idx), y


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


def main():
    movie = load_domain("movie", "movie_ids")
    book = load_domain("book", "book_ids")

    # 1) 영화로 가중치 학습
    X_fit, y_fit = build_features(movie, FIT_SPLIT)
    model = LinearRegression(positive=True).fit(X_fit, y_fit)
    print("계수:", {n: round(float(w), 3) for n, w in zip(FEATURE_NAMES, model.coef_)},
          "b =", round(float(model.intercept_), 3))

    # 2) 영화 Test / 도서 Test 평가 (predict만, 재학습 없음)
    X_mt, y_mt = build_features(movie, "test")
    X_bt, y_bt = build_features(book, "test")

    # 베이스라인: 각 도메인 Train 평점 평균으로 상수 예측 (모델이 이보다 나아야 의미 있음)
    movie_mean = float(movie["profile"]["train_ratings"].mean())
    book_mean = float(book["profile"]["train_ratings"].mean())

    pred_mt = model.predict(X_mt)
    pred_bt = model.predict(X_bt)

    # 평균 보정: 예측 평균을 해당 도메인 Train 평점 평균으로 옮김 (Test 정답은 사용하지 않음)
    # -> 영화로 학습한 절편 때문에 생기는 도메인 간 평균 오프셋을 제거해 순수한 순위/변동 성능을 본다.
    shift_mt = pred_mt - pred_mt.mean() + movie_mean
    shift_bt = pred_bt - pred_bt.mean() + book_mean

    # 편향 확인: 예측 평균 vs 실제 평균
    print("\n[평균 편향 확인] (예측 평균 / 실제 Test 평균 / 도메인 Train 평균)")
    print(f"  영화: {pred_mt.mean():.3f} / {y_mt.mean():.3f} / {movie_mean:.3f}")
    print(f"  도서: {pred_bt.mean():.3f} / {y_bt.mean():.3f} / {book_mean:.3f}")

    print(f"\n[영화({FIT_SPLIT}) 학습 결과 비교]")
    report("영화 Test (모델)", metrics(y_mt, pred_mt))
    report("영화 Test (모델, 평균 보정)", metrics(y_mt, shift_mt))
    report("영화 Test (평균 베이스라인)", metrics(y_mt, np.full_like(y_mt, movie_mean)))
    report("도서 Test (모델)", metrics(y_bt, pred_bt))
    report("도서 Test (모델, 평균 보정)", metrics(y_bt, shift_bt))
    report("도서 Test (평균 베이스라인)", metrics(y_bt, np.full_like(y_bt, book_mean)))

    # 3) Top-K 추천
    domain = movie if REC_DOMAIN == "movie" else book
    user_id = REC_USER_ID
    if user_id is None:
        user_id = int(domain["profile"]["test_user_ids"][0])
    print(f"\n[{REC_DOMAIN} 추천 Top-{TOP_K}] 유저 {user_id}")
    print(recommend_top_k(model, domain, user_id).round(3).to_string())


if __name__ == "__main__":
    main()

""" 결과
계수: {'genre_cos': 0.727, 'plot_cos': 0.867, 'length_closeness': -0.554, 'year_closeness': -0.94} b = 4.184

[평균 편향 확인] (예측 평균 / 실제 Test 평균 / 도메인 Train 평균)
  영화: 3.579 / 3.571 / 3.571
  도서: 3.787 / 3.921 / 3.920

[영화(validation) 학습 결과 비교]
  영화 Test (모델)                 RMSE=0.9846  MAE=0.7767  pearson=0.3572
  영화 Test (모델, 평균 보정)          RMSE=0.9846  MAE=0.7777  pearson=0.3572
  영화 Test (평균 베이스라인)           RMSE=1.0541  MAE=0.8360  pearson=nan
  도서 Test (모델)                 RMSE=0.9740  MAE=0.7914  pearson=0.2212
  도서 Test (모델, 평균 보정)          RMSE=0.9648  MAE=0.7612  pearson=0.2212
  도서 Test (평균 베이스라인)           RMSE=0.9893  MAE=0.7733  pearson=nan

[book 추천 Top-10] 유저 13478
   item_id                                                  title  score  genre_cos  plot_cos  length_closeness  year_closeness
0     4068                  Daddy-Long-Legs (Daddy-Long-Legs, #1)  4.343      0.743     0.645             0.970           0.428
1     7737                                    Ozma of Oz (Oz, #3)  4.292      0.579     0.676             0.946           0.399
2     1772            The Black Stallion (The Black Stallion, #1)  4.152      0.675     0.671             0.975           0.601
3     2879                     Mike Mulligan and His Steam Shovel  4.136      0.649     0.635             0.933           0.589
4      289                    Watership Down (Watership Down, #1)  4.125      0.579     0.729             0.986           0.601
5     2571            These Happy Golden Years (Little House, #8)  4.113      0.743     0.649             0.977           0.672
6     1603                                 The Story of Ferdinand  4.113      0.609     0.625             0.936           0.571
7     2446                                     The Story of Babar  4.112      0.579     0.652             0.931           0.577
8     2527  A Song of Ice and Fire (A Song of Ice and Fire, #1-5)  4.111      0.484     0.626             0.078           0.982
9     3348                                       The Little House  4.100      0.657     0.605             0.930           0.607




계수: {'genre_cos': 0.705, 'plot_cos': 0.863, 'length_closeness': 0.0, 'year_closeness': 0.0} b = 2.773
계수가 음수이던 요소들을 제거하고 "장르, 줄거리"만 비교한 결과. 음수라고 역효과를 가진게 아니라 영화 테스트에는 약간 도움이된다.

[평균 편향 확인] (예측 평균 / 실제 Test 평균 / 도메인 Train 평균)
  영화: 3.571 / 3.571 / 3.571
  도서: 3.807 / 3.921 / 3.920

[영화(validation) 학습 결과 비교]
  영화 Test (모델)                 RMSE=0.9861  MAE=0.7797  pearson=0.3535
  영화 Test (모델, 평균 보정)          RMSE=0.9861  MAE=0.7797  pearson=0.3535
  영화 Test (평균 베이스라인)           RMSE=1.0541  MAE=0.8360  pearson=nan
  도서 Test (모델)                 RMSE=0.9715  MAE=0.7847  pearson=0.2209
  도서 Test (모델, 평균 보정)          RMSE=0.9649  MAE=0.7604  pearson=0.2209
  도서 Test (평균 베이스라인)           RMSE=0.9893  MAE=0.7733  pearson=nan

[book 추천 Top-10] 유저 13478
   item_id                                         title  score  genre_cos  plot_cos  length_closeness  year_closeness
0      212                                     Atonement  3.995      0.820     0.746             0.989           0.970
1     6335                             Oscar and Lucinda  3.981      0.820     0.731             0.979           0.940
2     9627                 The Hand That First Held Mine  3.972      0.785     0.748             0.987           0.988
3     6090                          The Baker's Daughter  3.966      0.785     0.742             0.975           0.976
4      727                                    The Reader  3.962      0.761     0.757             0.963           0.934
5      689                               Beautiful Ruins  3.959      0.785     0.733             0.987           0.976
6     1566                                   Ethan Frome  3.954      0.761     0.747             0.941           0.982
7     1565                      Death Comes to Pemberley  3.953      0.783     0.728             0.978           0.982
8     4438                            Dear Life: Stories  3.953      0.720     0.780             0.986           0.976
9     6948  Sense & Sensibility (The Austen Project, #1)  3.953      0.776     0.734             0.991           0.971


"""