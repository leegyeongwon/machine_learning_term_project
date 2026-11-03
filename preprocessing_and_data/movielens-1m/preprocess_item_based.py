
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path


# ============================================================
# 기본 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed"
DATA_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

RATINGS_PATH = DATA_PATH / "movie_ratings.csv"
OUTPUT_PATH = DATA_PATH / "item_user_data.npz"


# ============================================================
# 1. 데이터 로드 및 8:1:1 분할
# ============================================================

print("평점 데이터 로딩 중...")

ratings = pd.read_csv(
    RATINGS_PATH,
    usecols=[
        "userId",
        "movieId",
        "rating",
        "timestamp",
        "tmdbId"
    ]
)

# ID 자료형 통일
ratings["userId"] = ratings["userId"].astype(np.int64)
ratings["movieId"] = ratings["movieId"].astype(np.int64)
ratings["tmdbId"] = ratings["tmdbId"].astype(np.int64)

print("전체 평점 수:", len(ratings))
print("전체 유저 수:", ratings["userId"].nunique())
print("전체 영화 수:", ratings["movieId"].nunique())

# 기존 데이터와 동일한 정렬 기준
ratings = ratings.sort_values(
    ["userId", "movieId"]
).reset_index(drop=True)


# ------------------------------------------------------------
# 8:1:1 분할
# ------------------------------------------------------------

print("\n평점 데이터 분할 중...")

# DataFrame 전체를 복사하는 대신 행 인덱스만 분할
all_indices = np.arange(len(ratings))

train_indices, temp_indices = train_test_split(
    all_indices,
    test_size=0.2,
    random_state=RANDOM_STATE,
    shuffle=True
)

validation_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.5,
    random_state=RANDOM_STATE,
    shuffle=True
)

# 필요한 열만 NumPy 배열로 추출
user_ids_all = ratings["userId"].to_numpy()
movie_ids_all = ratings["movieId"].to_numpy()
tmdb_ids_all = ratings["tmdbId"].to_numpy()
rating_values_all = ratings["rating"].to_numpy(dtype=np.float32)

# 각 데이터셋 분리
train_user_ids = user_ids_all[train_indices]
train_movie_ids = movie_ids_all[train_indices]
train_tmdb_ids = tmdb_ids_all[train_indices]
train_ratings = rating_values_all[train_indices]

validation_user_ids = user_ids_all[validation_indices]
validation_movie_ids = movie_ids_all[validation_indices]
validation_tmdb_ids = tmdb_ids_all[validation_indices]
validation_ratings = rating_values_all[validation_indices]

test_user_ids = user_ids_all[test_indices]
test_movie_ids = movie_ids_all[test_indices]
test_tmdb_ids = tmdb_ids_all[test_indices]
test_ratings = rating_values_all[test_indices]

print("Train:", len(train_indices))
print("Validation:", len(validation_indices))
print("Test:", len(test_indices))


# ============================================================
# 2. 전체 유저 및 영화 ID 정렬
# ============================================================

print("\n유저 및 영화 ID 정렬 중...")

# 전체 데이터에 등장하는 ID 유지
all_user_ids = np.sort(np.unique(user_ids_all))
all_movie_ids = np.sort(np.unique(movie_ids_all))

# 영화별 TMDB ID 연결
movie_mapping = ratings[
    ["movieId", "tmdbId"]
].drop_duplicates()

# 하나의 movieId에 여러 tmdbId가 연결되는 경우 방지
if movie_mapping["movieId"].duplicated().any():
    raise ValueError("하나의 movieId에 여러 tmdbId가 연결되어 있습니다.")

movie_tmdb_mapping = (
    movie_mapping
    .set_index("movieId")
    .reindex(all_movie_ids)["tmdbId"]
    .to_numpy(dtype=np.int64)
)

print("유저 수:", len(all_user_ids))
print("영화 수:", len(all_movie_ids))


# ============================================================
# 3. Train 데이터의 행렬 인덱스 생성
# ============================================================

# 실제 ID를 CSR 행렬의 정수 인덱스로 변환
train_user_indices = np.searchsorted(
    all_user_ids,
    train_user_ids
)

train_movie_indices = np.searchsorted(
    all_movie_ids,
    train_movie_ids
)

# 전체 행렬 크기
matrix_shape = (
    len(all_user_ids),
    len(all_movie_ids)
)

print("Train 평점 행렬 크기:", matrix_shape)


# ============================================================
# 4. Train 기준 유저별 Z-score 정규화
# ============================================================

print("\nTrain 기준 유저별 Z-score 정규화 중...")

# 유저별 Train 평점 개수
user_counts = np.bincount(
    train_user_indices,
    minlength=len(all_user_ids)
).astype(np.float64)

# 유저별 Train 평점 합계
user_sums = np.bincount(
    train_user_indices,
    weights=train_ratings,
    minlength=len(all_user_ids)
)

# 유저별 평균
user_means = np.divide(
    user_sums,
    user_counts,
    out=np.zeros_like(user_sums),
    where=user_counts > 0
)

# 각 평점에서 해당 유저의 평균을 뺀 값
centered_ratings = (
    train_ratings - user_means[train_user_indices]
)

# 유저별 편차 제곱 합계
user_squared_deviations = np.bincount(
    train_user_indices,
    weights=centered_ratings ** 2,
    minlength=len(all_user_ids)
)

# 모집단 표준편차 (ddof=0)
user_variances = np.divide(
    user_squared_deviations,
    user_counts,
    out=np.zeros_like(user_squared_deviations),
    where=user_counts > 0
)

user_stds = np.sqrt(user_variances)

# 표준편차가 0인 유저 처리
user_stds[user_stds == 0] = 1

# Train 평점 Z-score 정규화
normalized_train_ratings = (
    train_ratings - user_means[train_user_indices]
) / user_stds[train_user_indices]

normalized_train_ratings = normalized_train_ratings.astype(
    np.float32
)

print("Z-score 정규화 완료")


# ============================================================
# 5. CSR 희소행렬 생성
# ============================================================

print("\nCSR 희소행렬 생성 중...")

# Train 정규화 평점 행렬
rating_sparse = csr_matrix(
    (
        normalized_train_ratings,
        (train_user_indices, train_movie_indices)
    ),
    shape=matrix_shape,
    dtype=np.float32
)

# 정규화 결과가 0인 명시적 저장 원소 제거
rating_sparse.eliminate_zeros()

# Train 평가 여부 마스크
mask_sparse = csr_matrix(
    (
        np.ones(len(train_indices), dtype=np.int8),
        (train_user_indices, train_movie_indices)
    ),
    shape=matrix_shape,
    dtype=np.int8
)

rating_sparse.sort_indices()
mask_sparse.sort_indices()

print("CSR 변환 완료")
print("평점 행렬 저장 원소 수:", rating_sparse.nnz)
print("평가 마스크 저장 원소 수:", mask_sparse.nnz)


# ============================================================
# 6. 데이터 저장
# ============================================================

print("\nNPZ 파일 저장 중...")

np.savez_compressed(
    OUTPUT_PATH,

    # Train 정규화 평점 CSR
    rating_data=rating_sparse.data,
    rating_indices=rating_sparse.indices,
    rating_indptr=rating_sparse.indptr,
    rating_shape=np.asarray(rating_sparse.shape),

    # Train 평가 여부 CSR
    mask_data=mask_sparse.data,
    mask_indices=mask_sparse.indices,
    mask_indptr=mask_sparse.indptr,
    mask_shape=np.asarray(mask_sparse.shape),

    # 전체 유저 및 영화 ID
    user_ids=all_user_ids,
    movie_ids=all_movie_ids,
    tmdb_ids=movie_tmdb_mapping,

    # Train 기준 유저별 통계량
    user_means=user_means,
    user_stds=user_stds,

    # Train 원본 평점
    train_user_ids=train_user_ids,
    train_movie_ids=train_movie_ids,
    train_tmdb_ids=train_tmdb_ids,
    train_ratings=train_ratings,

    # Validation 원본 평점
    validation_user_ids=validation_user_ids,
    validation_movie_ids=validation_movie_ids,
    validation_tmdb_ids=validation_tmdb_ids,
    validation_ratings=validation_ratings,

    # Test 원본 평점
    test_user_ids=test_user_ids,
    test_movie_ids=test_movie_ids,
    test_tmdb_ids=test_tmdb_ids,
    test_ratings=test_ratings,

    random_state=np.asarray(RANDOM_STATE)
)

print("item_user_data.npz 저장 완료")


# ============================================================
# 7. 저장된 파일 로드 및 CSR 복원
# ============================================================

print("\n저장된 NPZ 파일 복원 중...")

with np.load(
    OUTPUT_PATH,
    allow_pickle=False
) as loaded:

    # 정규화 평점 행렬 복원
    restored_rating_sparse = csr_matrix(
        (
            loaded["rating_data"],
            loaded["rating_indices"],
            loaded["rating_indptr"]
        ),
        shape=tuple(loaded["rating_shape"])
    )

    # 평가 여부 마스크 복원
    restored_mask_sparse = csr_matrix(
        (
            loaded["mask_data"],
            loaded["mask_indices"],
            loaded["mask_indptr"]
        ),
        shape=tuple(loaded["mask_shape"])
    )

    # ID 및 통계량 복원
    user_ids = loaded["user_ids"].copy()
    movie_ids = loaded["movie_ids"].copy()
    tmdb_ids = loaded["tmdb_ids"].copy()

    restored_user_means = loaded["user_means"].copy()
    restored_user_stds = loaded["user_stds"].copy()

    # Validation 복원
    validation_user_ids = loaded["validation_user_ids"].copy()
    validation_movie_ids = loaded["validation_movie_ids"].copy()
    validation_tmdb_ids = loaded["validation_tmdb_ids"].copy()
    validation_ratings = loaded["validation_ratings"].copy()

    # Test 복원
    test_user_ids = loaded["test_user_ids"].copy()
    test_movie_ids = loaded["test_movie_ids"].copy()
    test_tmdb_ids = loaded["test_tmdb_ids"].copy()
    test_ratings = loaded["test_ratings"].copy()

print("CSR 행렬 복원 완료")


# ============================================================
# 8. 추천 알고리즘 입력 데이터 확인
# ============================================================

print("\n[추천 알고리즘 입력 데이터]")

print("유저 수:", len(user_ids))
print("영화 수:", len(movie_ids))

print("평점 행렬 크기:", restored_rating_sparse.shape)
print("평점 행렬 저장 원소 수:", restored_rating_sparse.nnz)

print("평가 마스크 저장 원소 수:", restored_mask_sparse.nnz)

print("Validation 평점 수:", len(validation_ratings))
print("Test 평점 수:", len(test_ratings))

print("추천 알고리즘 입력 데이터 준비 완료")


# ============================================================
# 사용 예시 1: Train 데이터 확인
# ============================================================

# 주의: 전체 행렬을 Dense로 변환하면 메모리를 많이 사용함.
# 일부 유저와 영화만 확인할 것.

sample_size = min(5, len(user_ids))

sample_rating = restored_rating_sparse[:sample_size, :10].toarray()
sample_mask = restored_mask_sparse[:sample_size, :10].toarray()

rating_df = pd.DataFrame(
    sample_rating,
    index=user_ids[:sample_size],
    columns=movie_ids[:10]
)

mask_df = pd.DataFrame(
    sample_mask,
    index=user_ids[:sample_size],
    columns=movie_ids[:10]
)

print("\n[Train 정규화 평점 데이터 일부]")
print(rating_df)

print("\n[Train 평점 관측 여부 일부]")
print(mask_df)


# ============================================================
# 사용 예시 2: 영화 간 유사도 비교
# ============================================================

# 두 영화의 Train 정규화 평점 벡터 추출
r0 = restored_rating_sparse[:, 0].toarray()
r1 = restored_rating_sparse[:, 1].toarray()

# 두 영화의 평가 여부 마스크 추출
m0 = restored_mask_sparse[:, 0].toarray()
m1 = restored_mask_sparse[:, 1].toarray()

# 양쪽 모두 평가한 유저만 남기는 공동 마스크
common_mask = m0 * m1

# 공통 유저의 평점만 추출
r0_common = r0 * common_mask
r1_common = r1 * common_mask

# 이후 r0_common과 r1_common으로 코사인 유사도 연산 진행

print("\nItem-based 전처리 완료!")