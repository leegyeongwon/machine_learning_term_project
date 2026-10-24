import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path

# 현재 파이썬 파일 기준으로 깃 루트 찾기
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 저장 파일 경로
DATA_PATH = PROJECT_ROOT / "data" / "processed"

# ============================================================
# 1. 데이터 로드
# ============================================================

rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

ratings = pd.read_csv(
    './ratings.dat',
    sep='::',
    names=rating_cols,
    engine='python',
    encoding='ISO-8859-1'
)

# 타임스탬프 제거
ratings = ratings.drop(columns=['timestamp'])

# ============================================================
# 2. 유저-영화 평점 행렬 생성
# ============================================================

# 행: 유저 / 열: 영화
# 미평가 항목은 NaN으로 유지
rating_matrix = ratings.pivot(
    index='user_id',
    columns='movie_id',
    values='rating'
)

print("원본 행렬 크기:", rating_matrix.shape)


# ============================================================
# 3. 평가 여부 마스크 생성
# ============================================================

# 실제로 평가한 항목은 True
# 미평가 항목은 False
observed_mask = rating_matrix.notna()

print("실제 평가 개수:", int(observed_mask.to_numpy().sum()))


# ============================================================
# 4. 유저별 Z-score 정규화
# ============================================================

# 각 유저가 실제로 평가한 영화들의 평균
user_means = rating_matrix.mean(axis=1)

# 각 유저가 실제로 평가한 영화들의 표준편차
user_stds = rating_matrix.std(axis=1, ddof=0)

# 표준편차가 0인 유저는 1로 대체
# 모든 영화에 같은 평점을 준 경우 0으로 나누지 않도록 처리
user_stds = user_stds.replace(0, 1)

# 유저별 평균을 빼고 표준편차로 나누기
normalized_matrix = rating_matrix.sub(
    user_means, axis=0
).div(
    user_stds, axis=0
)

# 미평가 항목을 계산용 0으로 변환
# 미평가 여부는 observed_mask에서 별도로 확인
normalized_matrix = normalized_matrix.fillna(0)

print("Z-score 정규화 완료")


# ============================================================
# 5. CSR 희소행렬 변환
# ============================================================

# 정규화 평점 행렬
rating_sparse = csr_matrix(
    normalized_matrix.to_numpy(dtype=np.float32)
)

# 평가 여부 행렬
# True=1, False=0
mask_sparse = csr_matrix(
    observed_mask.to_numpy(dtype=np.int8)
)

# 행렬 연산의 효율성을 위해 인덱스 정리
rating_sparse.sort_indices()
mask_sparse.sort_indices()

print("CSR 변환 완료")
print("평점 행렬 저장 원소 수:", rating_sparse.nnz)
print("평가 마스크 저장 원소 수:", mask_sparse.nnz)


# ============================================================
# 6. 하나의 .npz 파일로 압축 저장
# ============================================================

np.savez_compressed(
    DATA_PATH / "movie_item_based/item_user_data.npz",

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
    user_ids=rating_matrix.index.to_numpy(),
    movie_ids=rating_matrix.columns.to_numpy(),

    # 역변환 및 평점 예측에 사용할 유저별 통계량
    user_means=user_means.to_numpy(dtype=np.float64),
    user_stds=user_stds.to_numpy(dtype=np.float64)
)

print("item_user_data.npz 저장 완료")


# ============================================================
# 7. 파일 로드 및 CSR 행렬 복원
# ============================================================

with np.load(DATA_PATH / "movie_item_based/item_user_data.npz", allow_pickle=False) as loaded:

    # 정규화 평점 행렬 복원
    restored_rating_sparse = csr_matrix(
        (
            loaded['rating_data'],
            loaded['rating_indices'],
            loaded['rating_indptr']
        ),
        shape=tuple(loaded['rating_shape'])
    )

    # 평가 여부 행렬 복원
    restored_mask_sparse = csr_matrix(
        (
            loaded['mask_data'],
            loaded['mask_indices'],
            loaded['mask_indptr']
        ),
        shape=tuple(loaded['mask_shape'])
    )

    # ID 및 유저별 통계량 복원
    user_ids = loaded['user_ids'].copy()
    movie_ids = loaded['movie_ids'].copy()
    restored_user_means = loaded['user_means'].copy()
    restored_user_stds = loaded['user_stds'].copy()

print("CSR 행렬 복원 완료")



# ============================================================
# 8. 추천 알고리즘에서 사용할 데이터
# ============================================================

# 아래 두 행렬을 이후 아이템 기반 협업 필터링에 사용
# restored_rating_sparse: 정규화 평점
# restored_mask_sparse: 실제 평가 여부

print("유저 수:", len(user_ids))
print("영화 수:", len(movie_ids))
print("추천 알고리즘 입력 데이터 준비 완료")




# ============================================================
# 사용 예시 1: 로드한 데이터 확인
# ============================================================

# 1. 전체 희소 행렬을 DataFrame으로 한 번에 변환 (행: 사용자, 열: 영화)
rating_df = pd.DataFrame(restored_rating_sparse.toarray(), index=user_ids, columns=movie_ids)
mask_df = pd.DataFrame(restored_mask_sparse.toarray(), index=user_ids, columns=movie_ids)

# 2. .head()를 사용해 상위 5개 데이터 확인
print("\n[정규화된 평점 데이터]")
print(rating_df.head())

print("\n[평점 관측 여부: 1이면 실제 평점 존재]")
print(mask_df.head())


# ============================================================
# 사용 예시 2: 유사도 비교
# ============================================================

# 두 영화(예: 0번 영화와 1번 영화)의 정규화된 평점 벡터 벡터 추출
r0 = restored_rating_sparse[:, 0].toarray()
r1 = restored_rating_sparse[:, 1].toarray()

# 두 영화의 평가 여부 마스크 벡터 추출 (1 또는 0)
m0 = restored_mask_sparse[:, 0].toarray()
m1 = restored_mask_sparse[:, 1].toarray()

# 양쪽 모두 1인 칸(동시에 평가한 유저)만 남기는 공동 마스크 생성
common_mask = m0 * m1  # 원소별 곱셈 (Element-wise)

# 공통 유저의 평점만 추출하여 코사인 유사도 계산
r0_common = r0 * common_mask
r1_common = r1 * common_mask

# 이후 r0_common과 r1_common으로 코사인 유사도 연산 진행
