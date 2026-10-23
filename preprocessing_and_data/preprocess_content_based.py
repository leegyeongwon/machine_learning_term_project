import pandas as pd
import numpy as np

# pip install "huggingface-hub>=0.25.0,<2.0"
# pip install sentence_transformers
from sentence_transformers import SentenceTransformer
"""
# 1. 데이터 로드 (예시 데이터프레임)
data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': [
        'Toy Story (1995)', 
        'Jumanji (1995)', 
        'Grumpier Old Men (1995)', 
        'Waiting to Exhale (1995)', 
        'Father of the Bride Part II (1995)'
    ],
    'genres': [
        'Animation|Children\'s|Comedy', 
        'Adventure|Children\'s|Fantasy', 
        'Comedy|Romance', 
        'Comedy|Drama', 
        'Comedy'
    ]
}
df = pd.DataFrame(data)

# 실제 파일에서 로드하는 경우:
# df = pd.read_csv('movies.dat', sep='::', engine='python', names=['movie_id', 'title', 'genres'])

# 2. BGE-M3 모델 로드 및 장르 임베딩 생성
model = SentenceTransformer('BAAI/bge-m3')
print("모델 로드 완료")

# 장르 텍스트 전처리 (파이프 '|'를 쉼표로 변환해 임베딩 품질 향상)
genre_texts = df['genres'].str.replace('|', ', ', regex=False).tolist()

print("장르 임베딩 생성 중...")
embeddings = model.encode(genre_texts, normalize_embeddings=True)  # 크기: (영화 수, 1024)

# 3. .npz 파일로 압축 저장
np.savez_compressed(
    './movies_bge_m3_data.npz',
    embeddings=embeddings,              # 1024차원 임베딩 행렬 (Dense Matrix)
    movie_ids=df['movie_id'].values,    # 영화 ID 배열
    titles=df['title'].values,          # 영화 제목 배열
    genres=df['genres'].values          # 장르 원본 텍스트 배열
)
print("단일 .npz 파일로 압축 저장 완료\n")
"""

# --- [ 파일 로드 및 복원 ] ---
# 1. BGE-M3 임베딩 .npz 파일 로드
loaded_bge = np.load('./movies_bge_m3_data.npz', allow_pickle=True)

# 2. 임베딩 벡터를 하나의 열('embedding')로 묶어서 데이터프레임으로 복원
embeddings_df = pd.DataFrame({
    'title': loaded_bge['titles'],
    'genres': loaded_bge['genres'],
    'embedding': list(loaded_bge['embeddings'])  # 1024차원 벡터를 리스트로 변환해 한 열에 대입
}, index=loaded_bge['movie_ids'])

# 3. head() 사용해서 확인하기
print("--- BGE-M3 임베딩 데이터프레임 상위 5줄 ---")
print("행은 영화 id, 열은 특징들(장르 임베딩, )")
print(embeddings_df.head())