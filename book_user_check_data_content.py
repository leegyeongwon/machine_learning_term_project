import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix

file_path = Path("data/processed/book_user_profiles.npz")

with np.load(file_path, allow_pickle=True) as data:
    user_ids = data["user_ids"]

    # 모든 프로필 특성 컬럼을 유지
    genre_names = data["genre_names"]
    author_names = data["author_names"]
    numeric_feature_names = data["numeric_feature_names"]
    numeric_profiles = data["numeric_profiles"]
    plot_profiles = data["plot_profiles"]

    # 희소 유저 프로필 행렬 복원
    user_genre_profiles = csr_matrix((
        data["user_genre_profiles_data"],
        data["user_genre_profiles_indices"],
        data["user_genre_profiles_indptr"],
    ), shape=tuple(data["user_genre_profiles_shape"]))

    user_author_profiles = csr_matrix((
        data["user_author_profiles_data"],
        data["user_author_profiles_indices"],
        data["user_author_profiles_indptr"],
    ), shape=tuple(data["user_author_profiles_shape"]))

    print("===== 유저 프로필 =====")
    print("전체 유저 수:", len(user_ids))
    print("장르 선호 컬럼 수:", len(genre_names))
    print("저자 선호 컬럼 수:", len(author_names))
    print("수치형 컬럼:", numeric_feature_names.tolist())
    print("플롯 프로필 차원:", plot_profiles.shape[1])

    sample_indices = np.arange(min(5, len(user_ids)))

    for i in sample_indices:
        print(f"\n--- User {i + 1} ---")
        print("User ID:", user_ids[i])

        # 모든 특성 컬럼을 출력
        genre_row = user_genre_profiles[i].toarray().ravel()
        author_row = user_author_profiles[i].toarray().ravel()

        print("Genre profile (전체 컬럼):")
        print(dict(zip(genre_names, genre_row)))

        #print("Author profile (전체 컬럼):")
        #print(dict(zip(author_names, author_row)))

        print("Numeric profile:")
        print(dict(zip(numeric_feature_names, numeric_profiles[i])))

        print("Plot profile (앞 10개 값):")
        print(plot_profiles[i, :10])