
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from pathlib import Path

# 현재 파이썬 파일 기준으로 깃 루트 찾기
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 저장 파일 경로
DATA_PATH = PROJECT_ROOT / "data" / "processed"
DATA_PATH.mkdir(parents=True, exist_ok=True)

# 공통 데이터 분할 설정 (두 파일 동일하게 유지)
RANDOM_STATE = 42


# ==========================================
# 공통 평점 데이터 분할 함수
# ==========================================
def split_ratings(ratings_df):

    # Train 80%, 나머지 20%
    train_df, temp_df = train_test_split(
        ratings_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # 나머지 20%를 Validation 10%, Test 10%로 분할
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
# 1. 영화 데이터 로드 및 멀티 핫 인코딩
# ==========================================
print("영화 데이터 로딩 및 인코딩 중...")

movies_df = pd.read_csv(
    './movies.dat',
    sep='::',
    engine='python',
    encoding='ISO-8859-1',
    names=['movie_id', 'title', 'genres']
)

genre_lists = movies_df['genres'].str.split('|').tolist()

mlb = MultiLabelBinarizer()
movie_multihot = mlb.fit_transform(genre_lists)

movie_feature_df = pd.DataFrame(
    movie_multihot,
    index=movies_df['movie_id'],
    columns=mlb.classes_
)

# 영화 멀티 핫 정규화
movie_norms = np.linalg.norm(
    movie_multihot,
    axis=1,
    keepdims=True
)

movie_norms[movie_norms == 0] = 1.0

movie_multihot_norm = movie_multihot / movie_norms


# ==========================================
# 2. 평점 데이터 로드 및 8:1:1 분할
# ==========================================
print("평점 데이터 로딩 및 분할 중...")

ratings_df = pd.read_csv(
    './ratings.dat',
    sep='::',
    engine='python',
    encoding='ISO-8859-1',
    names=['user_id', 'movie_id', 'rating', 'timestamp']
)

# 두 파일의 입력 순서를 동일하게 보장
ratings_df = ratings_df.sort_values(
    ['user_id', 'movie_id']
).reset_index(drop=True)

train_df, validation_df, test_df = split_ratings(ratings_df)

print("전체 평점 수:", len(ratings_df))
print("Train:", len(train_df))
print("Validation:", len(validation_df))
print("Test:", len(test_df))


# ==========================================
# 3. Train 평점만으로 유저 프로필 생성
# ==========================================
print("Train 데이터로 유저 프로필 생성 중...")

# 평점 중심화 (3점 기준)
train_df = train_df.copy()
train_df['adjusted_rating'] = train_df['rating'] - 3.0

# Train 평점과 영화 장르 결합
merged_df = pd.merge(
    train_df,
    movie_feature_df,
    left_on='movie_id',
    right_index=True,
    how='inner'
)

genre_cols = list(mlb.classes_)

# 장르 벡터에 중심화된 평점 적용
for col in genre_cols:
    merged_df[col] = (
        merged_df[col] * merged_df['adjusted_rating']
    )

# 유저별 장르 점수 합산
user_profiles_raw = merged_df.groupby('user_id')[genre_cols].sum()

# L2 정규화
user_profiles_values = user_profiles_raw.to_numpy()

user_profile_norms = np.linalg.norm(
    user_profiles_values,
    axis=1,
    keepdims=True
)

user_profile_norms[user_profile_norms == 0] = 1.0

user_profiles_norm = (
    user_profiles_values / user_profile_norms
)

# 유저 프로필 DataFrame
user_profiles_df = pd.DataFrame(
    {
        'user_profile_raw': list(user_profiles_values),
        'user_profile_norm': list(user_profiles_norm)
    },
    index=user_profiles_raw.index
)

# ==========================================
# 4. 영화 정보 저장
# ==========================================
print("영화 데이터 저장 중...")

np.savez_compressed(
    DATA_PATH / 'movies_multihot_data.npz',

    multi_hot_norm=movie_multihot_norm,
    classes=mlb.classes_,
    movie_ids=movies_df['movie_id'].to_numpy(),
    titles=movies_df['title'].to_numpy()
)


# ==========================================
# 5. 유저 프로필 및 데이터 분할 결과 저장
# ==========================================
print("유저 프로필 및 분할 데이터 저장 중...")

np.savez_compressed(
    DATA_PATH / 'user_profiles_data.npz',

    # Train으로 생성한 유저 프로필
    user_ids=user_profiles_df.index.to_numpy(),
    user_profiles_norm=user_profiles_norm,
    genre_classes=mlb.classes_,

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

print("저장 완료!")


# ==========================================
# 6. 저장 데이터 복원 확인
# ==========================================
print("\n--- 유저 프로필 복원 테스트 ---")

with np.load(
    DATA_PATH / 'user_profiles_data.npz',
    allow_pickle=False
) as users_npz:

    restored_user_df = pd.DataFrame(
        {
            'user_profile_norm': list(
                users_npz['user_profiles_norm']
            )
        },
        index=users_npz['user_ids']
    )

    print(restored_user_df.head())

    print("\n분할 데이터 개수")
    print("Train:", len(users_npz['train_ratings']))
    print("Validation:", len(users_npz['validation_ratings']))
    print("Test:", len(users_npz['test_ratings']))


print("\n--- 영화 데이터 복원 테스트 ---")

with np.load(
    DATA_PATH / 'movies_multihot_data.npz',
    allow_pickle=False
) as movies_npz:

    restored_movie_df = pd.DataFrame(
        {
            'multi_hot_norm': list(
                movies_npz['multi_hot_norm']
            )
        },
        index=movies_npz['movie_ids']
    )

    print(restored_movie_df.head())

print("\nContent-based 전처리 완료!")
