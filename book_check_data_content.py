import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix

file_path = Path("data/processed/book_content.npz")

with np.load(file_path, allow_pickle=True) as data:
    book_ids = data["book_ids"]
    titles = data["titles"]

    # 모든 특성 컬럼을 유지
    genre_names = data["genre_names"]
    author_names = data["author_names"]
    numeric_feature_names = data["numeric_feature_names"]
    numeric_features = data["numeric_features"]
    plot_embeddings = data["plot_embeddings"]
    plot_valid_mask = data["plot_valid_mask"]

    # 희소 행렬 복원
    genres = csr_matrix((
        data["genres_data"],
        data["genres_indices"],
        data["genres_indptr"],
    ), shape=tuple(data["genres_shape"]))

    authors = csr_matrix((
        data["authors_data"],
        data["authors_indices"],
        data["authors_indptr"],
    ), shape=tuple(data["authors_shape"]))

    print("===== 도서 콘텐츠 =====")
    print("전체 도서 수:", len(book_ids))
    print("장르 컬럼 수:", len(genre_names))
    print("저자 컬럼 수:", len(author_names))
    print("수치형 컬럼:", numeric_feature_names.tolist())
    print("플롯 임베딩 차원:", plot_embeddings.shape[1])

    # 출력할 행만 선택
    sample_indices = np.arange(min(5, len(book_ids)))

    
    for i in sample_indices:
        print(f"\n--- 도서 {i + 1} ---")
        print("Book ID:", book_ids[i])
        print("Title:", titles[i])

        # 전체 장르/저자 컬럼을 유지하고 값만 출력
        genre_row = genres[i].toarray().ravel()
        author_row = authors[i].toarray().ravel()

        print("Genres (전체 컬럼):")
        print(dict(zip(genre_names, genre_row)))


#        print("Authors (전체 컬럼):")
#        print(dict(zip(author_names, author_row)))

        
        print("Numeric features:")
        print(dict(zip(numeric_feature_names, numeric_features[i])))
        

        
        print("Plot valid:", bool(plot_valid_mask[i]))
        print("Plot embedding (앞 10개 값):")
        print(plot_embeddings[i, :10])
        
        