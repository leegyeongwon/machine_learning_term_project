
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
from ast import literal_eval
from sklearn.model_selection import train_test_split


# ============================================================
# 기본 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed"

DATA_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# 1. 데이터 로드
# ============================================================

books_df = pd.read_csv(
    'https://raw.githubusercontent.com/malcolmosh/goodbooks-10k/master/books_enriched.csv',
    index_col=[0],
    converters={"genres": literal_eval}
)

# 유저별 도서 평점 데이터
ratings = pd.read_csv(
    'https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv'
)

print("메모리 적재 완료")


# ============================================================
# 2. Train / Validation / Test 분할
# ============================================================

def split_ratings(ratings_df):

    train_df, temp_df = train_test_split(
        ratings_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


train_df, validation_df, test_df = split_ratings(ratings)

print("데이터 분할 완료")
print("Train:", len(train_df))
print("Validation:", len(validation_df))
print("Test:", len(test_df))


# ============================================================
# 3. 전체 유저 및 도서 ID 인덱스 생성
# ============================================================

# 평가 데이터에 등장하는 유저와 도서도 포함
user_ids = np.sort(ratings['user_id'].unique())
book_ids = np.sort(ratings['book_id'].unique())

user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
book_to_idx = {bid: idx for idx, bid in enumerate(book_ids)}

# Train 데이터만 인덱스로 변환
row_indices = train_df['user_id'].map(user_to_idx).to_numpy()
col_indices = train_df['book_id'].map(book_to_idx).to_numpy()

rating_values = train_df['rating'].to_numpy(dtype=np.float32)


# ============================================================
# 4. Train 유저-도서 평점 희소행렬 생성
# ============================================================

rating_sparse = csr_matrix(
    (rating_values, (row_indices, col_indices)),
    shape=(len(user_ids), len(book_ids)),
    dtype=np.float32
)

print("Train 희소행렬 크기:", rating_sparse.shape)
print("Train 실제 평점 개수:", rating_sparse.nnz)


# ============================================================
# 5. Train 기준 유저별 Z-score 정규화
# ============================================================

# 유저별 평점 합계와 제곱합
user_rating_sums = np.asarray(
    rating_sparse.sum(axis=1)
).ravel()

user_rating_sq_sums = np.asarray(
    rating_sparse.power(2).sum(axis=1)
).ravel()

# 유저별 평점 개수
user_counts = np.diff(rating_sparse.indptr)

# 유저별 평균
user_means = np.divide(
    user_rating_sums,
    user_counts,
    out=np.zeros_like(user_rating_sums, dtype=np.float64),
    where=user_counts > 0
)

# 모집단 분산 (ddof=0)
user_variances = np.divide(
    user_rating_sq_sums,
    user_counts,
    out=np.zeros_like(user_rating_sq_sums, dtype=np.float64),
    where=user_counts > 0
) - user_means ** 2

user_stds = np.sqrt(np.maximum(user_variances, 0))

# 평점이 없거나 표준편차가 0인 유저는 1로 대체
user_stds[user_stds == 0] = 1.0

# CSR에 저장된 실제 평점만 정규화
normalized_data = rating_sparse.data.astype(
    np.float32,
    copy=True
)

for user_idx in range(len(user_ids)):

    start = rating_sparse.indptr[user_idx]
    end = rating_sparse.indptr[user_idx + 1]

    normalized_data[start:end] = (
        normalized_data[start:end] - user_means[user_idx]
    ) / user_stds[user_idx]

# 정규화 평점 희소행렬
rating_sparse = csr_matrix(
    (
        normalized_data,
        rating_sparse.indices.copy(),
        rating_sparse.indptr.copy()
    ),
    shape=rating_sparse.shape
)


# ============================================================
# 6. Train 평가 여부 마스크 생성
# ============================================================

mask_sparse = csr_matrix(
    (
        np.ones(rating_sparse.nnz, dtype=np.int8),
        rating_sparse.indices.copy(),
        rating_sparse.indptr.copy()
    ),
    shape=rating_sparse.shape
)

rating_sparse.sort_indices()
mask_sparse.sort_indices()

print("Z-score 정규화 및 CSR 변환 완료")
print("Train 평점 저장 원소 수:", rating_sparse.nnz)
print("Train 평가 마스크 저장 원소 수:", mask_sparse.nnz)


# ============================================================
# 7. 하나의 .npz 파일로 압축 저장
# ============================================================

np.savez_compressed(
    DATA_PATH / "book_item_user_data.npz",

    # Train 정규화 평점 CSR 구성 요소
    rating_data=rating_sparse.data,
    rating_indices=rating_sparse.indices,
    rating_indptr=rating_sparse.indptr,
    rating_shape=np.asarray(rating_sparse.shape),

    # Train 평가 여부 CSR 구성 요소
    mask_data=mask_sparse.data,
    mask_indices=mask_sparse.indices,
    mask_indptr=mask_sparse.indptr,
    mask_shape=np.asarray(mask_sparse.shape),

    # 전체 유저 및 도서 ID
    user_ids=user_ids,
    book_ids=book_ids,

    # Train 기준 유저별 통계량
    user_means=user_means,
    user_stds=user_stds,

    # Train 원본 평점
    train_user_ids=train_df['user_id'].to_numpy(),
    train_book_ids=train_df['book_id'].to_numpy(),
    train_ratings=train_df['rating'].to_numpy(),

    # Validation 원본 평점
    validation_user_ids=validation_df['user_id'].to_numpy(),
    validation_book_ids=validation_df['book_id'].to_numpy(),
    validation_ratings=validation_df['rating'].to_numpy(),

    # Test 원본 평점
    test_user_ids=test_df['user_id'].to_numpy(),
    test_book_ids=test_df['book_id'].to_numpy(),
    test_ratings=test_df['rating'].to_numpy(),

    # 재현성을 위한 랜덤 시드
    random_state=np.asarray(RANDOM_STATE)
)

print("book_item_user_data.npz 저장 완료")


# ============================================================
# 8. 저장 데이터 로드 및 CSR 복원
# ============================================================

with np.load(
    DATA_PATH / "book_item_user_data.npz",
    allow_pickle=False
) as loaded:

    restored_rating_sparse = csr_matrix(
        (
            loaded['rating_data'],
            loaded['rating_indices'],
            loaded['rating_indptr']
        ),
        shape=tuple(loaded['rating_shape'])
    )

    restored_mask_sparse = csr_matrix(
        (
            loaded['mask_data'],
            loaded['mask_indices'],
            loaded['mask_indptr']
        ),
        shape=tuple(loaded['mask_shape'])
    )

    user_ids = loaded['user_ids'].copy()
    book_ids = loaded['book_ids'].copy()

    restored_user_means = loaded['user_means'].copy()
    restored_user_stds = loaded['user_stds'].copy()

    # 원본 분할 평점 데이터 복원
    restored_train_user_ids = loaded['train_user_ids'].copy()
    restored_train_book_ids = loaded['train_book_ids'].copy()
    restored_train_ratings = loaded['train_ratings'].copy()

    restored_validation_user_ids = loaded['validation_user_ids'].copy()
    restored_validation_book_ids = loaded['validation_book_ids'].copy()
    restored_validation_ratings = loaded['validation_ratings'].copy()

    restored_test_user_ids = loaded['test_user_ids'].copy()
    restored_test_book_ids = loaded['test_book_ids'].copy()
    restored_test_ratings = loaded['test_ratings'].copy()

    restored_random_state = loaded['random_state'].copy()

print("CSR 행렬 및 원본 평점 데이터 복원 완료")


# ============================================================
# 9. 결과 확인
# ============================================================

print("\n[데이터 분할 결과]")
print("Train:", len(restored_train_ratings))
print("Validation:", len(restored_validation_ratings))
print("Test:", len(restored_test_ratings))

print("\n[전체 데이터]")
print("유저 수:", len(user_ids))
print("도서 수:", len(book_ids))

print("\n[Train 희소행렬]")
print("행렬 크기:", restored_rating_sparse.shape)
print("실제 평점 개수:", restored_rating_sparse.nnz)

# 일부 데이터만 확인
rating_preview = pd.DataFrame(
    restored_rating_sparse[:5, :10].toarray(),
    index=user_ids[:5],
    columns=book_ids[:10]
)

mask_preview = pd.DataFrame(
    restored_mask_sparse[:5, :10].toarray(),
    index=user_ids[:5],
    columns=book_ids[:10]
)

print("\n[Train 정규화 평점 데이터]")
print(rating_preview)

print("\n[Train 평가 여부: 1이면 실제 평점 존재]")
print(mask_preview)

print("\n추천 알고리즘 입력 데이터 준비 완료")
