import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from FlagEmbedding import BGEM3FlagModel

"""
# 유사도 계산 방법: 줄거리,작가,장르 -> 코사인 유사도.
#                  페이지, 출판 연도 -> abs((유저 선호 페이지수, 연도) - (실제 책의 페이지수, 연도)). 작을 수록 좋음

# 유사도 계산을 하면 줄거리(-1~1), 작가(-1~1), 장르(-1~1), 페이지(0~1), 연도(0~1)값이 나옵니다.

# 계산 후에 줄거리, 작가, 장르에 (A+1)/2 계산을 해서 0~1로 바꾸고, 페이지, 연도는 각각 (1-A)계산을 해서 1에 가까우면 좋도록 바꿉니다.
# 전체를 더하면 0점~5점이라는 점수를 가집니다. 0점일수록 유사도가 낮고 5점이면 유사도가 높습니다.

# 마지막으로 각 요소에 가중치를 곱해서 더합니다. 
# ex) 최종 점수 = (줄거리*w1 + 작가*w2 + 장르*w3 + 페이지*w4 + 연도*w5) 
# w1, w2, w3, w4, w5는 하이퍼파라미터로, validation set을 나눠 실험(머신러닝 학습) 후에 가장 좋은 것을 채택해 test합니다.

# 추가로 plot_embedding요소는 제대로 작동하는지 확인이 어렵습니다. 실험 후에 머신러닝 모델이 제대로 수렴하는지 보고 안되면 수정해야할 듯 합니다.
"""

# ============================================================
# 1. 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOOKS_URL = (
    "https://raw.githubusercontent.com/"
    "malcolmosh/goodbooks-10k-extended/master/books_enriched.csv"
)
RATINGS_URL = (
    "https://raw.githubusercontent.com/"
    "malcolmosh/goodbooks-10k-extended/master/ratings.csv"
)

# 최종 결과 파일은 이 두 개
BOOK_CONTENT_PATH = OUTPUT_DIR / "book_content.npz"
USER_PROFILES_PATH = OUTPUT_DIR / "book_user_profiles.npz"


# ============================================================
# 2. 데이터 불러오기
# ============================================================

books_df = pd.read_csv(
    BOOKS_URL,
    index_col=0,
    converters={"genres": ast.literal_eval},
)

ratings_df = pd.read_csv(RATINGS_URL)

# 도서 ID가 인덱스로 들어온 경우 일반 열로 변환
books_df = books_df.reset_index()

if "book_id" not in books_df.columns:
    books_df = books_df.rename(
        columns={books_df.columns[0]: "book_id"}
    )

required_book_columns = {
    "book_id", "genres", "authors", "pages", "publishDate", "description"
}
required_rating_columns = {"user_id", "book_id", "rating"}

missing_book_columns = required_book_columns - set(books_df.columns)
missing_rating_columns = required_rating_columns - set(ratings_df.columns)

if missing_book_columns:
    raise ValueError(f"도서 데이터에 없는 열: {missing_book_columns}")

if missing_rating_columns:
    raise ValueError(f"평점 데이터에 없는 열: {missing_rating_columns}")

# ID와 평점 정리
books_df["book_id"] = pd.to_numeric(
    books_df["book_id"], errors="coerce"
)
ratings_df["book_id"] = pd.to_numeric(
    ratings_df["book_id"], errors="coerce"
)
ratings_df["user_id"] = pd.to_numeric(
    ratings_df["user_id"], errors="coerce"
)
ratings_df["rating"] = pd.to_numeric(
    ratings_df["rating"], errors="coerce"
)

books_df = books_df.dropna(subset=["book_id"]).copy()
ratings_df = ratings_df.dropna(
    subset=["user_id", "book_id", "rating"]
).copy()

books_df["book_id"] = books_df["book_id"].astype(np.int64)
ratings_df["book_id"] = ratings_df["book_id"].astype(np.int64)
ratings_df["user_id"] = ratings_df["user_id"].astype(np.int64)

ratings_df = ratings_df[
    ratings_df["rating"].between(1, 5)
].copy()

# 도서 데이터에 존재하는 책의 평점만 사용
valid_book_ids = set(books_df["book_id"])
ratings_df = ratings_df[
    ratings_df["book_id"].isin(valid_book_ids)
].copy()

# 같은 유저가 같은 책을 여러 번 평가했다면 평균 사용
ratings_df = (
    ratings_df.groupby(["user_id", "book_id"], as_index=False)
    ["rating"].mean()
)

# 인덱스 매핑을 일관되게 유지
books_df = (
    books_df.sort_values("book_id")
    .drop_duplicates("book_id")
    .reset_index(drop=True)
)

book_ids = books_df["book_id"].to_numpy(dtype=np.int64)
user_ids = np.sort(ratings_df["user_id"].unique()).astype(np.int64)

book_to_idx = {book_id: i for i, book_id in enumerate(book_ids)}
user_to_idx = {user_id: i for i, user_id in enumerate(user_ids)}

ratings_df["book_idx"] = ratings_df["book_id"].map(book_to_idx)
ratings_df["user_idx"] = ratings_df["user_id"].map(user_to_idx)

print(f"도서 수: {len(book_ids):,}")
print(f"유저 수: {len(user_ids):,}")
print(f"평점 수: {len(ratings_df):,}")


# ============================================================
# 3. 공통 함수
# ============================================================

def parse_list(value):
    """리스트 또는 리스트 형태 문자열을 Python 리스트로 변환."""
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]

    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    # 문자열이 ['A', 'B'] 같은 리스트 형태인 경우
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [
                str(x).strip()
                for x in parsed
                if str(x).strip()
            ]
    except (ValueError, SyntaxError):
        pass

    # 데이터가 구분자 형태로 저장된 경우
    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]

    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]

    # 쉼표는 저자 이름 자체에 들어갈 수 있어 자동 분리하지 않음
    return [text]


def add_sparse(data_dict, prefix, matrix):
    """CSR 행렬을 하나의 NPZ 파일에 넣을 수 있도록 분해."""
    matrix = csr_matrix(matrix)

    data_dict[f"{prefix}_data"] = matrix.data
    data_dict[f"{prefix}_indices"] = matrix.indices
    data_dict[f"{prefix}_indptr"] = matrix.indptr
    data_dict[f"{prefix}_shape"] = np.asarray(
        matrix.shape, dtype=np.int64
    )


# ============================================================
# 4. 도서별 콘텐츠 특성
# ============================================================

n_books = len(books_df)

# ---------- 4-1. 장르: 멀티핫 + 행별 L2 정규화 ----------

genre_lists = books_df["genres"].apply(parse_list).tolist()

genre_encoder = MultiLabelBinarizer(sparse_output=True)
genres = genre_encoder.fit_transform(genre_lists).astype(np.float32)
genres = normalize(genres, norm="l2", axis=1)

# ---------- 4-2. 저자: 멀티핫 + 행별 L2 정규화 ----------

author_lists = books_df["authors"].apply(parse_list).tolist()

author_encoder = MultiLabelBinarizer(sparse_output=True)
authors = author_encoder.fit_transform(author_lists).astype(np.float32)
authors = normalize(authors, norm="l2", axis=1)

# ---------- 4-3. 페이지 수와 출판 연도: Min-Max 정규화 ----------

pages = pd.to_numeric(
    books_df["pages"], errors="coerce"
).to_numpy(dtype=np.float32)

pages_missing = ~np.isfinite(pages) | (pages <= 0)
pages[pages_missing] = np.nan

publish_dates = books_df["publishDate"].astype("string")
publish_year = pd.to_datetime(
    publish_dates, errors="coerce"
).dt.year.to_numpy(dtype=np.float32)

# 날짜 전체를 파싱하지 못하는 경우 문자열에서 연도 추출
for i, value in enumerate(publish_dates):
    if not np.isfinite(publish_year[i]):
        match = re.search(r"\b(1000|1\d{3}|20\d{2})\b", str(value))
        if match:
            publish_year[i] = float(match.group(1))

year_missing = ~np.isfinite(publish_year)
publish_year[year_missing] = np.nan


def minmax_feature(values):
    """결측값은 중앙값으로 채운 뒤 0~1로 변환."""
    values = values.astype(np.float32).copy()
    missing = ~np.isfinite(values)

    valid = values[~missing]
    if len(valid) == 0:
        return np.full_like(values, 0.5), missing, 0.0, 1.0

    median = float(np.median(valid))
    values[missing] = median

    min_value = float(values.min())
    max_value = float(values.max())

    if max_value == min_value:
        scaled = np.full_like(values, 0.5)
    else:
        scaled = (values - min_value) / (max_value - min_value)

    return scaled, missing, min_value, max_value


pages_scaled, pages_missing, pages_min, pages_max = minmax_feature(pages)
year_scaled, year_missing, year_min, year_max = minmax_feature(publish_year)

numeric_features = np.column_stack(
    [pages_scaled, year_scaled]
).astype(np.float32)

# ---------- 4-4. 줄거리: BGE-M3 임베딩 ----------
# GPU 환경을 권장합니다. 데이터 양이 많아서 cpu로 하니까 5시간 걸림

descriptions = (
    books_df["description"]
    .fillna("")
    .astype(str)
    .str.strip()
    .tolist()
)

valid_plot_mask = np.array(
    [bool(text) for text in descriptions], dtype=bool
)

valid_plot_indices = np.flatnonzero(valid_plot_mask)

plot_embeddings = None
plot_dim = 0

if len(valid_plot_indices) > 0:
    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=True,
        devices=["cuda:0"] 
    )

    valid_descriptions = [
        descriptions[i] for i in valid_plot_indices
    ]

    embedding_batches = []

    for start in range(0, len(valid_descriptions), 16):
        batch = valid_descriptions[start:start + 16]

        result = model.encode(
            batch,
            batch_size=16,
            max_length=2048,
        )

        embedding_batches.append(
            np.asarray(result["dense_vecs"], dtype=np.float32)
        )

    valid_embeddings = np.concatenate(
        embedding_batches, axis=0
    )

    valid_embeddings = normalize(
        valid_embeddings, norm="l2", axis=1
    ).astype(np.float32)

    plot_dim = valid_embeddings.shape[1]

    # 설명이 없는 책은 0 벡터로 두고 유효 여부를 별도 저장
    plot_embeddings = np.zeros(
        (n_books, plot_dim), dtype=np.float32
    )
    plot_embeddings[valid_plot_indices] = valid_embeddings

else:
    plot_embeddings = np.zeros((n_books, 0), dtype=np.float32)

# ---------- 4-5. 도서 콘텐츠를 단 하나의 파일에 저장 ----------

book_content = {
    "book_ids": book_ids,
    "titles": (
        books_df["title"].fillna("").astype(str).to_numpy()
        if "title" in books_df.columns
        else np.full(n_books, "", dtype=str)
    ),
    "genre_names": genre_encoder.classes_,
    "author_names": author_encoder.classes_,
    "numeric_feature_names": np.array(["pages", "publish_year"]),
    "numeric_features": numeric_features,
    "pages_missing": pages_missing,
    "year_missing": year_missing,
    "pages_min": np.array(pages_min, dtype=np.float32),
    "pages_max": np.array(pages_max, dtype=np.float32),
    "year_min": np.array(year_min, dtype=np.float32),
    "year_max": np.array(year_max, dtype=np.float32),
    "plot_embeddings": plot_embeddings,
    "plot_valid_mask": valid_plot_mask,
    "plot_dim": np.array(plot_dim, dtype=np.int64),
}

add_sparse(book_content, "genres", genres)
add_sparse(book_content, "authors", authors)

np.savez_compressed(BOOK_CONTENT_PATH, **book_content)
print(f"도서 콘텐츠 저장 완료: {BOOK_CONTENT_PATH}")


# ============================================================
# 5. 유저별 선호 프로필
# ============================================================

# 평점 중심화 (Centering: 3점 기준)
ratings_df["adjusted_rating"] = (
    ratings_df["rating"] - 3.0
).astype(np.float32)

user_idx = ratings_df["user_idx"].to_numpy(dtype=np.int64)
book_idx = ratings_df["book_idx"].to_numpy(dtype=np.int64)
adjusted = ratings_df["adjusted_rating"].to_numpy(dtype=np.float32)

n_users = len(user_ids)

ratings_adjusted = csr_matrix(
    (adjusted, (user_idx, book_idx)),
    shape=(n_users, n_books),
    dtype=np.float32,
)

rating_strength = csr_matrix(
    (np.abs(adjusted), (user_idx, book_idx)),
    shape=(n_users, n_books),
    dtype=np.float32,
)

# 3점 초과 평점만 선호도에 반영
preference_weights = ratings_adjusted.maximum(0).tocsr()

# ---------- 5-1. 장르 프로필 ----------

user_genre_profiles = preference_weights @ genres
user_genre_profiles = normalize(
    user_genre_profiles, norm="l2", axis=1
).tocsr()

# ---------- 5-2. 작가 프로필 ----------

user_author_profiles = preference_weights @ authors
user_author_profiles = normalize(
    user_author_profiles, norm="l2", axis=1
).tocsr()

# ---------- 5-3. 줄거리 프로필 ----------

# 줄거리 없는 도서는 줄거리 선호 계산에서 제외
valid_plot_ratings = ratings_adjusted[:, valid_plot_mask]

if plot_dim > 0 and valid_plot_mask.any():
    user_plot_profiles = (
        valid_plot_ratings @ plot_embeddings[valid_plot_mask]
    )
    user_plot_profiles = normalize(
        user_plot_profiles, norm="l2", axis=1
    ).astype(np.float32)
else:
    user_plot_profiles = np.zeros(
        (n_users, plot_dim), dtype=np.float32
    )

# ---------- 5-4. 수치형 프로필 ----------

# 평점 3점 초과인 책만 선호 프로필에 반영
# 4점은 가중치 1, 5점은 가중치 2
preference_weights = ratings_adjusted.maximum(0).tocsr()

# 책의 특징값과 같은 0~1 범위에서 사용자 선호값 계산
numeric_profiles = np.zeros((n_users, 2), dtype=np.float32)

for feature_idx, missing_mask in enumerate(
    [pages_missing, year_missing]
):
    observed = ~missing_mask

    # 해당 특징값이 존재하는 책만 프로필 계산에 사용
    observed_values = numeric_features[:, feature_idx].copy()
    observed_values[~observed] = 0.0

    numerator = np.asarray(
        preference_weights @ observed_values
    ).reshape(-1)

    denominator = np.asarray(
        preference_weights @ observed.astype(np.float32)
    ).reshape(-1)

    np.divide(
        numerator,
        denominator,
        out=numeric_profiles[:, feature_idx],
        where=denominator > 0,
    )

    # 선호하는 책이 없거나 해당 특징값을 가진 선호 책이 없으면
    # 프로필을 0.5(정규화 범위의 중간값)로 설정
    numeric_profiles[denominator <= 0, feature_idx] = 0.5

    # 유사도 계산 시에 책의 페이지, 연도랑 유저 선호 페이지, 연도랑 뺀 후에 절댓값 취하는 방법이 있습니다.
    # 작을 수록 좋습니다

# ---------- 5-5. 유저 프로필을 하나의 파일에 저장 ----------

user_profiles = {
    "user_ids": user_ids,
    "genre_names": genre_encoder.classes_,
    "author_names": author_encoder.classes_,
    "numeric_feature_names": np.array(["pages", "publish_year"]),
    "numeric_profiles": numeric_profiles,
    "plot_profiles": user_plot_profiles,
}

add_sparse(
    user_profiles, "user_genre_profiles", user_genre_profiles
)
add_sparse(
    user_profiles, "user_author_profiles", user_author_profiles
)

np.savez_compressed(USER_PROFILES_PATH, **user_profiles)
print(f"유저 프로필 저장 완료: {USER_PROFILES_PATH}")

print("전처리 완료: 최종 결과 파일 2개 생성")