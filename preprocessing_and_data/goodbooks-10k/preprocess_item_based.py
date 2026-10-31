import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
from ast import literal_eval

# 현재 파이썬 파일 기준으로 깃 루트 찾기
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 저장 파일 경로
DATA_PATH = PROJECT_ROOT / "data" / "processed"

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
# 2. 유저-도서 평점 희소행렬 생성
# ============================================================

# 유저 ID와 도서 ID를 행렬 인덱스로 변환
user_ids = np.sort(ratings['user_id'].unique())
book_ids = np.sort(ratings['book_id'].unique())

user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
book_to_idx = {bid: idx for idx, bid in enumerate(book_ids)}

row_indices = ratings['user_id'].map(user_to_idx).to_numpy()
col_indices = ratings['book_id'].map(book_to_idx).to_numpy()
rating_values = ratings['rating'].to_numpy(dtype=np.float32)

# 실제 평점이 있는 항목만 저장
rating_sparse = csr_matrix(
    (rating_values, (row_indices, col_indices)),
    shape=(len(user_ids), len(book_ids)),
    dtype=np.float32
)

print("원본 희소행렬 크기:", rating_sparse.shape)
# nnz는 희소 행렬에서 실제 데이터가 몇개인지 알려줌 
print("실제 평점 개수:", rating_sparse.nnz)

# ============================================================
# 3. 유저별 Z-score 정규화
# ============================================================

# 유저별 평점 합계와 제곱합
user_rating_sums = np.asarray(
    rating_sparse.sum(axis=1)
).ravel()

user_rating_sq_sums = np.asarray(
    rating_sparse.power(2).sum(axis=1)
).ravel()

# 유저별 평점 준 개수
user_counts = np.diff(rating_sparse.indptr)

# 유저별 평균
user_means = np.divide(
    user_rating_sums,
    user_counts,
    out=np.zeros_like(user_rating_sums, dtype=np.float64),
    where=user_counts > 0
)

# 모집단 표준편차 (ddof=0)
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
normalized_data = rating_sparse.data.astype(np.float32, copy=True)

# 각 유저가 저장한 평점들의 위치를 이용해 정규화
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
# 4. 평가 여부 마스크 생성
# ============================================================

# 평점이 존재하는 위치에만 1 저장
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
print("평점 행렬 저장 원소 수:", rating_sparse.nnz)
print("평가 마스크 저장 원소 수:", mask_sparse.nnz)

# ============================================================
# 5. 하나의 .npz 파일로 압축 저장
# ============================================================

np.savez_compressed(
    DATA_PATH / "book_item_user_data.npz",

    # 정규화 평점 CSR 구성 요소
    rating_data=rating_sparse.data,
    rating_indices=rating_sparse.indices,
    rating_indptr=rating_sparse.indptr,
    rating_shape=np.asarray(rating_sparse.shape),

    # 평가 여부 CSR 구성 요소
    mask_data=mask_sparse.data,
    mask_indices=mask_sparse.indices,
    mask_indptr=mask_sparse.indptr,
    mask_shape=np.asarray(mask_sparse.shape),

    # 행과 열의 실제 ID
    user_ids=user_ids,
    book_ids=book_ids,

    # 역변환 및 평점 예측에 사용할 유저별 통계량
    user_means=user_means,
    user_stds=user_stds
)

print("book_item_user_data.npz 저장 완료")


# ============================================================
# 6. 저장 데이터 로드 및 CSR 복원
# ============================================================

with np.load(DATA_PATH / "book_item_user_data.npz", allow_pickle=False) as loaded:

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

print("CSR 행렬 복원 완료")

# ============================================================
# 7. 결과 확인
# ============================================================

print("유저 수:", len(user_ids))
print("도서 수:", len(book_ids))
print("추천 알고리즘 입력 데이터 준비 완료")

# 일부 데이터만 확인하기
# 이 데이터셋은 양이 많아서 전체 행렬을 toarray()로 변환하면 메모리를 많이 사용할 수 있음
# 유저 5명과 책 10개만 출력
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

print("\n[정규화된 평점 데이터]")
print(rating_preview)

print("\n[평가 여부: 1이면 실제 평점 존재]")
print(mask_preview)