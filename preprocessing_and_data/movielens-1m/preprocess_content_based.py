import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path

# 현재 파이썬 파일 기준으로 깃 루트 찾기
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 저장 파일 경로
DATA_PATH = PROJECT_ROOT / "data" / "processed"


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
movie_norms = np.linalg.norm(movie_multihot, axis=1, keepdims=True)
movie_norms[movie_norms == 0] = 1.0
movie_multihot_norm = movie_multihot / movie_norms

# ==========================================
# 2. 평점 데이터 로드 및 유저 프로필 생성 (단일 컬럼화)
# ==========================================
print("평점 데이터 로딩 및 유저 프로필 생성 중...")
ratings_df = pd.read_csv(
    './ratings.dat', 
    sep='::', 
    engine='python', 
    encoding='ISO-8859-1', 
    names=['user_id', 'movie_id', 'rating', 'timestamp']
)

# 평점 중심화 (Centering: 3점 기준)
ratings_df['adjusted_rating'] = ratings_df['rating'] - 3.0

merged_df = pd.merge(ratings_df, movie_feature_df, left_on='movie_id', right_index=True)

genre_cols = list(mlb.classes_)
for col in genre_cols:
    merged_df[col] = merged_df[col] * merged_df['adjusted_rating']

# 유저별로 그룹화하여 장르 점수 합산
user_profiles_raw = merged_df.groupby('user_id')[genre_cols].sum()

# L2 정규화 수행
user_profiles_values = user_profiles_raw.values
user_profile_norms = np.linalg.norm(user_profiles_values, axis=1, keepdims=True)
user_profile_norms[user_profile_norms == 0] = 1.0
user_profiles_norm = user_profiles_values / user_profile_norms

# 유저 프로필을 1개의 열에 리스트 형태로 담은 데이터프레임 구축
user_profiles_df = pd.DataFrame({
    'user_profile_raw': list(user_profiles_values),
    'user_profile_norm': list(user_profiles_norm)
}, index=user_profiles_raw.index)


# ==========================================
# 3. .npz 파일로 압축 저장
# ==========================================
print("\n데이터 저장 중...")
np.savez_compressed(
    DATA_PATH / 'movies_multihot_data.npz',
    multi_hot_norm=movie_multihot_norm,
    classes=mlb.classes_,
    movie_ids=movies_df['movie_id'].values,
    titles=movies_df['title'].values,
)

np.savez_compressed(
    DATA_PATH / 'user_profiles_data.npz',
    user_ids=user_profiles_df.index.values,
    user_profiles_norm=user_profiles_norm,
    genre_classes=mlb.classes_
)
print("저장 완료!")


# ==========================================
# 4. 저장된 .npz 파일 로드 후 단일 컬럼 DataFrame 복원 확인 (유저 & 영화)
# ==========================================
print("\n--- 유저 프로필 .npz 파일 로드 및 복원 테스트 ---")
users_npz = np.load(DATA_PATH / 'user_profiles_data.npz', allow_pickle=True)

restored_user_df = pd.DataFrame({
    'user_profile_norm': list(users_npz['user_profiles_norm'])
}, index=users_npz['user_ids'])

print(restored_user_df.head())


print("\n--- 영화 데이터 .npz 파일 로드 및 복원 테스트 ---")
movies_npz = np.load(DATA_PATH / 'movies_multihot_data.npz', allow_pickle=True)

restored_movie_df = pd.DataFrame({
    'multi_hot_norm': list(movies_npz['multi_hot_norm']),
}, index=movies_npz['movie_ids'])

print(restored_movie_df.head())