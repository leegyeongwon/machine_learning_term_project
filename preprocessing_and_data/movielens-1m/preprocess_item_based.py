import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path

# 현재 파이썬 파일 기준으로 깃 루트 찾기
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 저장 파일 경로
DATA_PATH = PROJECT_ROOT / "data" / "processed"
DATA_PATH.mkdir(parents=True, exist_ok=True)

# Content-based 파일과 동일한 설정
RANDOM_STATE = 42


# ==========================================
# 공통 평점 데이터 분할 함수
# ==========================================
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


# ==========================================
# 1. 데이터 로드 및 8:1:1 분할
# ==========================================
print("평점 데이터 로딩 및 분할 중...")

rating_cols = [
    'user_id',
    'movie_id',
    'rating',
    'timestamp'
]

ratings = pd.read_csv(
    './ratings.dat',
    sep='::',
    names=rating_cols,
    engine='python',
    encoding='ISO-8859-1'
)

# Content-based 파일과 동일한 정렬 기준
ratings = ratings.sort_values(
    ['user_id', 'movie_id']
).reset_index(drop=True)

train_df, validation_df, test_df = split_ratings(ratings)

print("전체 평점 수:", len(ratings))
print("Train:", len(train_df))
print("Validation:", len(validation_df))
print("Test:", len(test_df))

# 타임스탬프 제거
train_df = train_df.drop(columns=['timestamp'])
validation_df = validation_df.drop(columns=['timestamp'])
test_df = test_df.drop(columns=['timestamp'])


# ==========================================
# 2. Train 유저-영화 평점 행렬 생성
# ==========================================

# 원본 전체 데이터에 등장하는 유저와 영화 ID 유지
all_user_ids = np.sort(ratings['user_id'].unique())
all_movie_ids = np.sort(ratings['movie_id'].unique())

# Train 평점만으로 피벗
rating_matrix = train_df.pivot(
    index='user_id',
    columns='movie_id',
    values='rating'
)

# 전체 유저 및 영화 ID 기준으로 행렬 확장
rating_matrix = rating_matrix.reindex(
    index=all_user_ids,
    columns=all_movie_ids
)

print("Train 평점 행렬 크기:", rating_matrix.shape)


# ==========================================
# 3. 평가 여부 마스크 생성
# ==========================================

# 실제 Train에서 평가한 항목은 True
# 미평가 항목은 False
observed_mask = rating_matrix.notna()

print(
    "Train 실제 평가 개수:",
    int(observed_mask.to_numpy().sum())
)


# ==========================================
# 4. Train 기준 유저별 Z-score 정규화
# ==========================================

# Train에서 실제로 평가한 영화들의 평균
user_means = rating_matrix.mean(axis=1)

# Train에서 실제로 평가한 영화들의 표준편차
user_stds = rating_matrix.std(axis=1, ddof=0)

# Train 평점이 없는 유저는 평균 0으로 처리
user_means = user_means.fillna(0)

# 표준편차가 0이거나 NaN이면 1로 처리
user_stds = user_stds.fillna(1)
user_stds = user_stds.replace(0, 1)

# Z-score 정규화
normalized_matrix = rating_matrix.sub(
    user_means,
    axis=0
).div(
    user_stds,
    axis=0
)

# 미평가 항목을 계산용 0으로 변환
normalized_matrix = normalized_matrix.fillna(0)

print("Train 기준 Z-score 정규화 완료")


# ==========================================
# 5. CSR 희소행렬 변환
# ==========================================

rating_sparse = csr_matrix(
    normalized_matrix.to_numpy(dtype=np.float32)
)

mask_sparse = csr_matrix(
    observed_mask.to_numpy(dtype=np.int8)
)

rating_sparse.sort_indices()
mask_sparse.sort_indices()

print("CSR 변환 완료")
print("평점 행렬 저장 원소 수:", rating_sparse.nnz)
print("평가 마스크 저장 원소 수:", mask_sparse.nnz)


# ==========================================
# 6. 데이터 저장
# ==========================================

np.savez_compressed(
    DATA_PATH / "item_user_data.npz",

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
    user_ids=rating_matrix.index.to_numpy(),
    movie_ids=rating_matrix.columns.to_numpy(),

    # Train 기준 유저별 통계량
    user_means=user_means.to_numpy(dtype=np.float64),
    user_stds=user_stds.to_numpy(dtype=np.float64),

    # Train 원본 평점
    train_user_ids=train_df['user_id'].to_numpy(),
    train_movie_ids=train_df['movie_id'].to_numpy(),
    train_ratings=train_df['rating'].to_numpy(),

    # Validation 원본 평점
    validation_user_ids=validation_df['user_id'].to_numpy(),
    validation_movie_ids=validation_df['movie_id'].to_numpy(),
    validation_ratings=validation_df['rating'].to_numpy(),

    # Test 원본 평점
    test_user_ids=test_df['user_id'].to_numpy(),
    test_movie_ids=test_df['movie_id'].to_numpy(),
    test_ratings=test_df['rating'].to_numpy(),

    random_state=np.asarray(RANDOM_STATE)
)

print("item_user_data.npz 저장 완료")


# ==========================================
# 7. 저장된 파일 로드 및 CSR 복원
# ==========================================

with np.load(
    DATA_PATH / "item_user_data.npz",
    allow_pickle=False
) as loaded:

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

    # ID 및 통계량 복원
    user_ids = loaded['user_ids'].copy()
    movie_ids = loaded['movie_ids'].copy()

    restored_user_means = loaded['user_means'].copy()
    restored_user_stds = loaded['user_stds'].copy()

    # Validation 복원
    validation_user_ids = loaded['validation_user_ids'].copy()
    validation_movie_ids = loaded['validation_movie_ids'].copy()
    validation_ratings = loaded['validation_ratings'].copy()

    # Test 복원
    test_user_ids = loaded['test_user_ids'].copy()
    test_movie_ids = loaded['test_movie_ids'].copy()
    test_ratings = loaded['test_ratings'].copy()

print("CSR 행렬 복원 완료")


# ==========================================
# 8. 추천 알고리즘 입력 데이터 확인
# ==========================================

# restored_rating_sparse: Train 정규화 평점
# restored_mask_sparse: Train 실제 평가 여부
# validation_*: Validation 원본 평점
# test_*: Test 원본 평점

print("유저 수:", len(user_ids))
print("영화 수:", len(movie_ids))

print("Validation 평점 수:", len(validation_ratings))
print("Test 평점 수:", len(test_ratings))

print("추천 알고리즘 입력 데이터 준비 완료")


# ==========================================
# 사용 예시 1: Train 데이터 확인
# ==========================================

rating_df = pd.DataFrame(
    restored_rating_sparse.toarray(),
    index=user_ids,
    columns=movie_ids
)

mask_df = pd.DataFrame(
    restored_mask_sparse.toarray(),
    index=user_ids,
    columns=movie_ids
)

print("\n[Train 정규화 평점 데이터]")
print(rating_df.head())

print("\n[Train 평점 관측 여부]")
print(mask_df.head())


# ==========================================
# 사용 예시 2: 영화 간 유사도 비교
# ==========================================

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
