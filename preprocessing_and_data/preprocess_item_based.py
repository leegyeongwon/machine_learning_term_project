import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

# 데이터 로드 및 피벗 테이블 생성. (유저 번호/영화 번호/별점) 으로 구성된 데이터입니다.
rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('./ratings.dat', sep='::', names=rating_cols, engine='python', encoding='ISO-8859-1')

ratings_filtered = ratings.drop(columns=['timestamp'])
item_user_matrix = ratings_filtered.pivot_table(
    index='user_id', 
    columns='movie_id', 
    values='rating'
).fillna(0)

# 스파스 행렬 변환
# 이 데이터는 0이 매우 많습니다. 즉 희소행렬입니다.
# 희소행렬을 가공해서 0점을 다 날립니다.
sparse_matrix = csr_matrix(item_user_matrix.values)


# 그냥 저장하려니까 너무 무겁습니다.

# --- [ 1. 하나의 .npz 파일로 압축 저장 ] ---
# sparse matrix의 구성 요소(data, indices, indptr, shape)와 index/columns를 한 번에 저장합니다.
np.savez_compressed(
    './item_user_data.npz',
    data=sparse_matrix.data,
    indices=sparse_matrix.indices,
    indptr=sparse_matrix.indptr,
    shape=sparse_matrix.shape,
    movie_ids=item_user_matrix.index.values,
    user_ids=item_user_matrix.columns.values
)
print("단일 .npz 파일로 압축 저장 완료\n")


# --- [ 2. 파일 로드해서 복원하기 ] ---
loaded = np.load('./item_user_data.npz', allow_pickle=True)

# 스파스 행렬 재조립
restored_sparse = csr_matrix(
    (loaded['data'], loaded['indices'], loaded['indptr']), 
    shape=loaded['shape']
)

# 데이터프레임으로 복원 및 상위 5줄 출력
restored_df = pd.DataFrame(
    restored_sparse.toarray(),
    index=loaded['movie_ids'],
    columns=loaded['user_ids']
)

print("--- 복원된 데이터프레임 상위 5줄 ---")
print(restored_df.head())