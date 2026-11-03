import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, coo_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from FlagEmbedding import BGEM3FlagModel


# ============================================================
# 1. 기본 설정
# ============================================================

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MOVIE_FEATURE_PATH = OUTPUT_DIR / "movie_features.csv"
MOVIE_RATINGS_PATH = OUTPUT_DIR / "movie_ratings.csv"

# 기존 도서 전처리 결과
BOOK_CONTENT_PATH = OUTPUT_DIR / "book_content.npz"
BOOK_USER_PROFILES_PATH = OUTPUT_DIR / "book_user_profiles.npz"

# 영화 전처리 결과
MOVIE_CONTENT_PATH = OUTPUT_DIR / "movie_content.npz"
MOVIE_USER_PROFILES_PATH = OUTPUT_DIR / "movie_user_profiles.npz"

# 영화 협업 필터링 결과: 분할 일치 여부 검증용
CF_PATH = OUTPUT_DIR / "item_user_data.npz"


# ============================================================
# 2. 공통 함수
# ============================================================

def parse_list(value):
    """
    리스트 또는 리스트 형태 문자열을 Python 리스트로 변환.
    도서의 저자 등 일반 리스트형 데이터에 사용.
    """

    if isinstance(value, (list, tuple, set)):
        return [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]

    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return []

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

    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]

    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]

    return [text]


def parse_genres(value):
    """
    영화 장르를 리스트로 변환.

    다음 형태를 처리:
    - ['Action', 'Adventure']
    - "['Action', 'Adventure']"
    - "Action|Adventure"
    - "Action;Adventure"
    - "Action, Adventure"
    - [{'id': 28, 'name': 'Action'}, ...]
    """

    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, (list, tuple, set)):
        parsed = value
    else:
        text = str(value).strip()

        if not text or text.lower() in {"nan", "none", "null"}:
            return []

        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            if "|" in text:
                parsed = text.split("|")
            elif ";" in text:
                parsed = text.split(";")
            elif "," in text:
                parsed = text.split(",")
            else:
                parsed = [text]

    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]

    result = []

    for item in parsed:
        # TMDB 형식처럼 장르가 딕셔너리인 경우
        if isinstance(item, dict):
            name = item.get("name")
            if name:
                result.append(str(name).strip())
        else:
            name = str(item).strip()
            if name:
                result.append(name)

    # 중복 제거
    return list(dict.fromkeys(result))


def add_sparse(data_dict, prefix, matrix):
    """CSR 행렬을 NPZ 파일에 저장할 수 있도록 분해."""

    matrix = csr_matrix(matrix)

    data_dict[f"{prefix}_data"] = matrix.data
    data_dict[f"{prefix}_indices"] = matrix.indices
    data_dict[f"{prefix}_indptr"] = matrix.indptr
    data_dict[f"{prefix}_shape"] = np.asarray(
        matrix.shape,
        dtype=np.int64,
    )


def load_sparse(data, prefix):
    """NPZ에 분해되어 저장된 CSR 행렬을 복원."""

    return csr_matrix(
        (
            data[f"{prefix}_data"],
            data[f"{prefix}_indices"],
            data[f"{prefix}_indptr"],
        ),
        shape=tuple(data[f"{prefix}_shape"]),
    )


def remap_sparse_columns(matrix, old_names, new_names):
    """
    기존 장르 행렬의 열을 공통 장르 사전의 열 위치로 재배치.

    기존 장르 열:
        [Action, Fantasy, Romance]

    공통 장르 열:
        [Action, Adventure, Fantasy, Mystery, Romance]

    결과적으로 기존에 없던 장르는 0으로 유지된다.
    """

    matrix = matrix.tocoo()

    new_name_to_idx = {
        name: idx
        for idx, name in enumerate(new_names)
    }

    column_mapping = np.asarray(
        [new_name_to_idx[name] for name in old_names],
        dtype=np.int64,
    )

    remapped = coo_matrix(
        (
            matrix.data,
            (
                matrix.row,
                column_mapping[matrix.col],
            ),
        ),
        shape=(matrix.shape[0], len(new_names)),
        dtype=np.float32,
    ).tocsr()

    return remapped


def minmax_feature(values):
    """
    결측값은 중앙값으로 채운 뒤 0~1로 변환.
    기존 도서 전처리 코드와 같은 방식.
    """

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


def parse_year(values):
    """
    연도 데이터를 숫자로 변환.
    숫자로 변환할 수 없는 값은 문자열에서 연도를 추출.
    """

    result = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=np.float32)

    for i, value in enumerate(values):
        if not np.isfinite(result[i]):
            match = re.search(
                r"\b(1000|1\d{3}|20\d{2})\b",
                str(value),
            )

            if match:
                result[i] = float(match.group(1))

    return result


# ============================================================
# 3. 데이터 불러오기
# ============================================================

movies_df = pd.read_csv(MOVIE_FEATURE_PATH)
ratings_df = pd.read_csv(MOVIE_RATINGS_PATH)

required_movie_columns = {
    "tmdbId",
    "title",
    "genres",
    "runtime",
    "overview",
    "release_year",
}

required_rating_columns = {
    "userId",
    "movieId",
    "rating",
    "timestamp",
    "tmdbId",
}

missing_movie_columns = (
    required_movie_columns - set(movies_df.columns)
)

missing_rating_columns = (
    required_rating_columns - set(ratings_df.columns)
)

if missing_movie_columns:
    raise ValueError(
        f"영화 콘텐츠 데이터에 없는 열: {missing_movie_columns}"
    )

if missing_rating_columns:
    raise ValueError(
        f"영화 평점 데이터에 없는 열: {missing_rating_columns}"
    )


# ID 자료형 통일
movies_df["tmdbId"] = pd.to_numeric(
    movies_df["tmdbId"],
    errors="coerce",
)

ratings_df["movieId"] = pd.to_numeric(
    ratings_df["movieId"],
    errors="coerce",
)

ratings_df["tmdbId"] = pd.to_numeric(
    ratings_df["tmdbId"],
    errors="coerce",
)

ratings_df["userId"] = pd.to_numeric(
    ratings_df["userId"],
    errors="coerce",
)

ratings_df["rating"] = pd.to_numeric(
    ratings_df["rating"],
    errors="coerce",
)

ratings_df["timestamp"] = pd.to_numeric(
    ratings_df["timestamp"],
    errors="coerce",
)

if ratings_df[
    ["userId", "movieId", "rating", "timestamp", "tmdbId"]
].isna().any().any():
    raise ValueError(
        "영화 평점 데이터에 결측값이 있습니다. "
        "협업 필터링 코드와 동일한 데이터인지 확인하세요."
    )

ratings_df["userId"] = ratings_df["userId"].astype(np.int64)
ratings_df["movieId"] = ratings_df["movieId"].astype(np.int64)
ratings_df["tmdbId"] = ratings_df["tmdbId"].astype(np.int64)

movies_df = movies_df.dropna(subset=["tmdbId"]).copy()
movies_df["tmdbId"] = movies_df["tmdbId"].astype(np.int64)

# TMDB ID 중복 제거
movies_df = (
    movies_df
    .drop_duplicates("tmdbId")
    .reset_index(drop=True)
)

# ------------------------------------------------------------
# 중요:
# 영화 CF 코드와 동일하게 userId, movieId 순으로 정렬.
# 이 정렬 순서를 바꾸면 Train/Validation/Test 분할이
# 협업 필터링 결과와 달라질 수 있다.
# ------------------------------------------------------------

ratings_df = (
    ratings_df[
        ["userId", "movieId", "rating", "timestamp", "tmdbId"]
    ]
    .sort_values(["userId", "movieId"])
    .reset_index(drop=True)
)

# movieId 하나가 여러 tmdbId에 연결되는지 확인
movie_mapping = ratings_df[
    ["movieId", "tmdbId"]
].drop_duplicates()

if movie_mapping["movieId"].duplicated().any():
    raise ValueError(
        "하나의 movieId에 여러 tmdbId가 연결되어 있습니다."
    )

# 평점에 등장하는 영화만 사용
movie_mapping = (
    movie_mapping
    .sort_values("movieId")
    .reset_index(drop=True)
)

movie_df = movie_mapping.merge(
    movies_df,
    on="tmdbId",
    how="left",
    validate="many_to_one",
)

if movie_df[
    ["title", "genres", "runtime", "overview", "release_year"]
].isna().all(axis=1).any():
    raise ValueError(
        "영화 평점 데이터에 대응하는 콘텐츠 메타데이터가 없습니다. "
        "movieId와 tmdbId 연결을 확인하세요."
    )

if movie_df["title"].isna().any():
    missing_count = int(movie_df["title"].isna().sum())
    raise ValueError(
        f"콘텐츠 메타데이터가 없는 영화가 {missing_count:,}개 있습니다."
    )

movie_ids = movie_df["movieId"].to_numpy(dtype=np.int64)
tmdb_ids = movie_df["tmdbId"].to_numpy(dtype=np.int64)

user_ids = np.sort(
    ratings_df["userId"].unique()
).astype(np.int64)

movie_to_idx = {
    movie_id: idx
    for idx, movie_id in enumerate(movie_ids)
}

user_to_idx = {
    user_id: idx
    for idx, user_id in enumerate(user_ids)
}

print(f"영화 수: {len(movie_ids):,}")
print(f"유저 수: {len(user_ids):,}")
print(f"평점 수: {len(ratings_df):,}")


# ============================================================
# 4. Train / Validation / Test 분할 (8:1:1)
# ============================================================

def split_ratings(ratings):
    """영화 CF 코드와 동일한 방식의 8:1:1 분할."""

    train_df, temp_df = train_test_split(
        ratings,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


train_df, validation_df, test_df = split_ratings(ratings_df)

print("데이터 분할 완료")
print(f"Train: {len(train_df):,}")
print(f"Validation: {len(validation_df):,}")
print(f"Test: {len(test_df):,}")


# ============================================================
# 5. 도서와 영화의 공통 장르 사전 생성
# ============================================================

if not BOOK_CONTENT_PATH.exists():
    raise FileNotFoundError(
        f"기존 도서 콘텐츠 파일이 없습니다: {BOOK_CONTENT_PATH}"
    )

if not BOOK_USER_PROFILES_PATH.exists():
    raise FileNotFoundError(
        f"기존 도서 유저 프로필 파일이 없습니다: "
        f"{BOOK_USER_PROFILES_PATH}"
    )

# 기존 도서 데이터 불러오기
with np.load(
    BOOK_CONTENT_PATH,
    allow_pickle=False,
) as data:
    book_content = {
        key: data[key]
        for key in data.files
    }

with np.load(
    BOOK_USER_PROFILES_PATH,
    allow_pickle=False,
) as data:
    book_profiles = {
        key: data[key]
        for key in data.files
    }

# 기존 도서 장르 사전
book_genre_names = book_content["genre_names"].astype(str)

# 영화 장르 파싱
movie_genre_lists = (
    movie_df["genres"]
    .apply(parse_genres)
    .tolist()
)

movie_genre_names = sorted({
    genre
    for genres in movie_genre_lists
    for genre in genres
})

# 도서 + 영화 장르의 합집합
shared_genre_names = np.asarray(
    sorted(
        set(book_genre_names.tolist())
        | set(movie_genre_names)
    ),
    dtype=str,
)

print(f"도서 장르 수: {len(book_genre_names):,}")
print(f"영화 장르 수: {len(movie_genre_names):,}")
print(f"공통 장르 수: {len(shared_genre_names):,}")

# ------------------------------------------------------------
# 기존 도서 장르 행렬을 공통 장르 차원으로 확장
# ------------------------------------------------------------

book_genres_old = load_sparse(
    book_content,
    "genres",
)

book_genres = remap_sparse_columns(
    book_genres_old,
    book_genre_names,
    shared_genre_names,
)

# 기존 도서 유저 장르 프로필도 같은 차원으로 확장
book_user_genres_old = load_sparse(
    book_profiles,
    "user_genre_profiles",
)

book_user_genres = remap_sparse_columns(
    book_user_genres_old,
    book_genre_names,
    shared_genre_names,
)

# ------------------------------------------------------------
# 영화 장르 행렬 생성
# 공통 장르 사전을 사용하므로 도서와 열의 의미가 동일하다.
# ------------------------------------------------------------

genre_to_idx = {
    genre: idx
    for idx, genre in enumerate(shared_genre_names)
}

genre_rows = []
genre_cols = []

for row_idx, genres in enumerate(movie_genre_lists):
    for genre in genres:
        genre_rows.append(row_idx)
        genre_cols.append(genre_to_idx[genre])

movie_genres = coo_matrix(
    (
        np.ones(len(genre_rows), dtype=np.float32),
        (genre_rows, genre_cols),
    ),
    shape=(len(movie_df), len(shared_genre_names)),
    dtype=np.float32,
).tocsr()

# 도서 코드와 동일하게 행별 L2 정규화
movie_genres = normalize(
    movie_genres,
    norm="l2",
    axis=1,
).tocsr()

# 도서 결과의 장르 행렬과 장르 이름 갱신
book_content["genre_names"] = shared_genre_names
book_profiles["genre_names"] = shared_genre_names

book_content.pop("genres_data")
book_content.pop("genres_indices")
book_content.pop("genres_indptr")
book_content.pop("genres_shape")

book_profiles.pop("user_genre_profiles_data")
book_profiles.pop("user_genre_profiles_indices")
book_profiles.pop("user_genre_profiles_indptr")
book_profiles.pop("user_genre_profiles_shape")

add_sparse(
    book_content,
    "genres",
    book_genres,
)

add_sparse(
    book_profiles,
    "user_genre_profiles",
    book_user_genres,
)

print("도서와 영화의 공통 장르 공간 생성 완료")


# ============================================================
# 6. 출판 연도와 개봉 연도의 공통 정규화 범위 계산
# ============================================================

# 도서의 기존 연도 정규화 결과 복원
book_numeric_features = book_content[
    "numeric_features"
].astype(np.float32).copy()

book_year_missing = book_content[
    "year_missing"
].astype(bool)

old_book_year_min = float(book_content["year_min"])
old_book_year_max = float(book_content["year_max"])

book_year_raw = (
    book_numeric_features[:, 1]
    * (old_book_year_max - old_book_year_min)
    + old_book_year_min
)

# 기존 결측 도서 연도는 다시 결측으로 처리
book_year_raw[book_year_missing] = np.nan

# 영화 개봉 연도
movie_year_raw = parse_year(
    movie_df["release_year"]
)

movie_year_missing = ~np.isfinite(movie_year_raw)

# 두 도메인의 유효한 연도를 합쳐 공통 범위 계산
valid_years = np.concatenate([
    book_year_raw[~book_year_missing],
    movie_year_raw[~movie_year_missing],
])

if len(valid_years) == 0:
    raise ValueError(
        "도서와 영화 데이터 모두 유효한 연도가 없습니다."
    )

# 두 도메인을 합친 중앙값으로 결측값 처리
shared_year_median = float(np.median(valid_years))

book_year_raw[book_year_missing] = shared_year_median
movie_year_raw[movie_year_missing] = shared_year_median

shared_year_min = float(
    min(book_year_raw.min(), movie_year_raw.min())
)

shared_year_max = float(
    max(book_year_raw.max(), movie_year_raw.max())
)

if shared_year_max == shared_year_min:
    book_year_scaled = np.full(
        len(book_year_raw),
        0.5,
        dtype=np.float32,
    )

    movie_year_scaled = np.full(
        len(movie_year_raw),
        0.5,
        dtype=np.float32,
    )

else:
    book_year_scaled = (
        (book_year_raw - shared_year_min)
        / (shared_year_max - shared_year_min)
    ).astype(np.float32)

    movie_year_scaled = (
        (movie_year_raw - shared_year_min)
        / (shared_year_max - shared_year_min)
    ).astype(np.float32)

# 도서 콘텐츠의 연도 특성 갱신
book_numeric_features[:, 1] = book_year_scaled

book_content["numeric_features"] = book_numeric_features
book_content["year_min"] = np.asarray(
    shared_year_min,
    dtype=np.float32,
)
book_content["year_max"] = np.asarray(
    shared_year_max,
    dtype=np.float32,
)

# 영화 연도 정규화 결과는 아래 영화 numeric_features에 사용


# ============================================================
# 7. 기존 도서 수치형 프로필 다시 계산
# ============================================================

# 연도 정규화 범위가 변경되었으므로
# 기존 Train 평점으로 도서 수치형 프로필을 다시 계산한다.

book_ids = book_content["book_ids"].astype(np.int64)
book_user_ids = book_profiles["user_ids"].astype(np.int64)

book_to_idx = {
    book_id: idx
    for idx, book_id in enumerate(book_ids)
}

book_user_to_idx = {
    user_id: idx
    for idx, user_id in enumerate(book_user_ids)
}

book_train_user_idx = np.asarray([
    book_user_to_idx[user_id]
    for user_id in book_profiles["train_user_ids"]
], dtype=np.int64)

book_train_item_idx = np.asarray([
    book_to_idx[book_id]
    for book_id in book_profiles["train_book_ids"]
], dtype=np.int64)

book_train_adjusted = (
    book_profiles["train_ratings"].astype(np.float32) - 3.0
)

book_ratings_adjusted = csr_matrix(
    (
        book_train_adjusted,
        (book_train_user_idx, book_train_item_idx),
    ),
    shape=(len(book_user_ids), len(book_ids)),
    dtype=np.float32,
)

book_preference_weights = (
    book_ratings_adjusted.maximum(0).tocsr()
)

book_numeric_profiles = np.zeros(
    (len(book_user_ids), 2),
    dtype=np.float32,
)

for feature_idx, missing_mask in enumerate([
    book_content["pages_missing"].astype(bool),
    book_content["year_missing"].astype(bool),
]):
    observed = ~missing_mask

    observed_values = (
        book_content["numeric_features"][:, feature_idx].copy()
    )

    observed_values[~observed] = 0.0

    numerator = np.asarray(
        book_preference_weights @ observed_values
    ).reshape(-1)

    denominator = np.asarray(
        book_preference_weights @ observed.astype(np.float32)
    ).reshape(-1)

    np.divide(
        numerator,
        denominator,
        out=book_numeric_profiles[:, feature_idx],
        where=denominator > 0,
    )

    book_numeric_profiles[denominator <= 0, feature_idx] = 0.5

book_profiles["numeric_profiles"] = book_numeric_profiles

print("도서 연도 특성과 수치형 프로필 갱신 완료")


# ============================================================
# 8. 영화 콘텐츠 특성
# ============================================================

n_movies = len(movie_df)

# ---------- 8-1. 장르 ----------
# 앞에서 공통 장르 사전으로 생성 및 L2 정규화 완료.


# ---------- 8-2. 상영 시간: Min-Max 정규화 ----------

runtime = pd.to_numeric(
    movie_df["runtime"],
    errors="coerce",
).to_numpy(dtype=np.float32)

runtime_missing = (
    ~np.isfinite(runtime)
    | (runtime <= 0)
)

runtime[runtime_missing] = np.nan

runtime_scaled, runtime_missing, runtime_min, runtime_max = (
    minmax_feature(runtime)
)


# ---------- 8-3. 개봉 연도 ----------
# 도서 출판 연도와 동일한 Min-Max 범위 사용

numeric_features = np.column_stack([
    runtime_scaled,
    movie_year_scaled,
]).astype(np.float32)


# ---------- 8-4. 줄거리: BGE-M3 임베딩 ----------

overviews = (
    movie_df["overview"]
    .fillna("")
    .astype(str)
    .str.strip()
    .tolist()
)

valid_plot_mask = np.asarray(
    [bool(text) for text in overviews],
    dtype=bool,
)

valid_plot_indices = np.flatnonzero(valid_plot_mask)

plot_dim = 0

if len(valid_plot_indices) > 0:

    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=True,
        devices=["cuda:0"],
    )

    valid_overviews = [
        overviews[i]
        for i in valid_plot_indices
    ]

    embedding_batches = []

    for start in range(0, len(valid_overviews), 16):
        batch = valid_overviews[start:start + 16]

        result = model.encode(
            batch,
            batch_size=16,
            max_length=2048,
        )

        embedding_batches.append(
            np.asarray(
                result["dense_vecs"],
                dtype=np.float32,
            )
        )

    valid_embeddings = np.concatenate(
        embedding_batches,
        axis=0,
    )

    valid_embeddings = normalize(
        valid_embeddings,
        norm="l2",
        axis=1,
    ).astype(np.float32)

    plot_dim = valid_embeddings.shape[1]

    plot_embeddings = np.zeros(
        (n_movies, plot_dim),
        dtype=np.float32,
    )

    plot_embeddings[valid_plot_indices] = valid_embeddings

else:
    plot_embeddings = np.zeros(
        (n_movies, 0),
        dtype=np.float32,
    )


# ============================================================
# 9. 영화 콘텐츠 저장
# ============================================================

movie_content = {
    "movie_ids": movie_ids,
    "tmdb_ids": tmdb_ids,

    "titles": (
        movie_df["title"]
        .fillna("")
        .astype(str)
        .to_numpy()
    ),

    "genre_names": shared_genre_names,

    "numeric_feature_names": np.array([
        "runtime",
        "release_year",
    ]),

    "numeric_features": numeric_features,

    "runtime_missing": runtime_missing,
    "year_missing": movie_year_missing,

    "runtime_min": np.asarray(
        runtime_min,
        dtype=np.float32,
    ),

    "runtime_max": np.asarray(
        runtime_max,
        dtype=np.float32,
    ),

    # 도서와 동일한 연도 정규화 범위
    "year_min": np.asarray(
        shared_year_min,
        dtype=np.float32,
    ),

    "year_max": np.asarray(
        shared_year_max,
        dtype=np.float32,
    ),

    "plot_embeddings": plot_embeddings,
    "plot_valid_mask": valid_plot_mask,

    "plot_dim": np.asarray(
        plot_dim,
        dtype=np.int64,
    ),
}

add_sparse(
    movie_content,
    "genres",
    movie_genres,
)

np.savez_compressed(
    MOVIE_CONTENT_PATH,
    **movie_content,
)

print(f"영화 콘텐츠 저장 완료: {MOVIE_CONTENT_PATH}")


# ============================================================
# 10. 영화 유저 선호 프로필
#     Train 평점만 사용
# ============================================================

n_users = len(user_ids)

train_user_idx = train_df["userId"].map(
    user_to_idx
).to_numpy(dtype=np.int64)

train_movie_idx = train_df["movieId"].map(
    movie_to_idx
).to_numpy(dtype=np.int64)

train_adjusted = (
    train_df["rating"].to_numpy(dtype=np.float32) - 3.0
).astype(np.float32)

ratings_adjusted = csr_matrix(
    (
        train_adjusted,
        (train_user_idx, train_movie_idx),
    ),
    shape=(n_users, n_movies),
    dtype=np.float32,
)

# ---------- 10-1. 장르 프로필 ----------

# 도서 코드와 동일하게 3점 초과 평점만 반영
# 4점은 가중치 1, 5점은 가중치 2

preference_weights = (
    ratings_adjusted.maximum(0).tocsr()
)

user_genre_profiles = (
    preference_weights @ movie_genres
)

user_genre_profiles = normalize(
    user_genre_profiles,
    norm="l2",
    axis=1,
).tocsr()


# ---------- 10-2. 수치형 프로필 ----------

numeric_profiles = np.zeros(
    (n_users, 2),
    dtype=np.float32,
)

for feature_idx, missing_mask in enumerate([
    runtime_missing,
    movie_year_missing,
]):
    observed = ~missing_mask

    observed_values = (
        numeric_features[:, feature_idx].copy()
    )

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

    numeric_profiles[denominator <= 0, feature_idx] = 0.5


# ---------- 10-3. 줄거리 프로필 ----------

# 기존 도서 코드와 동일하게 평점 중심화 값을 사용.
# 줄거리가 없는 영화는 프로필 계산에서 제외.

if plot_dim > 0 and valid_plot_mask.any():

    valid_plot_ratings = ratings_adjusted[:, valid_plot_mask]

    user_plot_profiles = np.zeros(
        (n_users, plot_dim),
        dtype=np.float32,
    )

    # 사용자 수가 많으므로 행 단위로 나누어 계산
    # 한 번에 큰 dense 행렬을 만드는 것을 방지
    chunk_size = 10000

    valid_movie_embeddings = plot_embeddings[
        valid_plot_mask
    ]

    for start in range(0, n_users, chunk_size):
        end = min(start + chunk_size, n_users)

        batch_profiles = (
            valid_plot_ratings[start:end]
            @ valid_movie_embeddings
        )

        batch_profiles = normalize(
            batch_profiles,
            norm="l2",
            axis=1,
        )

        user_plot_profiles[start:end] = batch_profiles

else:
    user_plot_profiles = np.zeros(
        (n_users, plot_dim),
        dtype=np.float32,
    )


# ============================================================
# 11. 영화 유저 프로필 + 평점 분할 데이터 저장
# ============================================================

movie_user_profiles = {
    "user_ids": user_ids,

    "genre_names": shared_genre_names,

    "numeric_feature_names": np.array([
        "runtime",
        "release_year",
    ]),

    "numeric_profiles": numeric_profiles,
    "plot_profiles": user_plot_profiles,

    # Train 원본 평점
    "train_user_ids": train_df["userId"].to_numpy(),
    "train_movie_ids": train_df["movieId"].to_numpy(),
    "train_tmdb_ids": train_df["tmdbId"].to_numpy(),
    "train_ratings": train_df["rating"].to_numpy(),
    "train_timestamps": train_df["timestamp"].to_numpy(),

    # Validation 원본 평점
    "validation_user_ids": validation_df["userId"].to_numpy(),
    "validation_movie_ids": validation_df["movieId"].to_numpy(),
    "validation_tmdb_ids": validation_df["tmdbId"].to_numpy(),
    "validation_ratings": validation_df["rating"].to_numpy(),
    "validation_timestamps": validation_df["timestamp"].to_numpy(),

    # Test 원본 평점
    "test_user_ids": test_df["userId"].to_numpy(),
    "test_movie_ids": test_df["movieId"].to_numpy(),
    "test_tmdb_ids": test_df["tmdbId"].to_numpy(),
    "test_ratings": test_df["rating"].to_numpy(),
    "test_timestamps": test_df["timestamp"].to_numpy(),

    "random_state": np.asarray(
        RANDOM_STATE,
        dtype=np.int64,
    ),
}

add_sparse(
    movie_user_profiles,
    "user_genre_profiles",
    user_genre_profiles,
)

np.savez_compressed(
    MOVIE_USER_PROFILES_PATH,
    **movie_user_profiles,
)

print(f"영화 유저 프로필 저장 완료: {MOVIE_USER_PROFILES_PATH}")


# ============================================================
# 12. 갱신된 도서 파일 저장
# ============================================================

np.savez_compressed(
    BOOK_CONTENT_PATH,
    **book_content,
)

np.savez_compressed(
    BOOK_USER_PROFILES_PATH,
    **book_profiles,
)

print("도서 공통 장르 및 연도 특성 갱신 완료")


# ============================================================
# 13. 협업 필터링 결과와 분할 검증
# ============================================================

if CF_PATH.exists():

    with np.load(
        CF_PATH,
        allow_pickle=False,
    ) as cf:

        checks = {}

        # 기본 ID 확인
        if "user_ids" in cf:
            checks["user_ids"] = (
                user_ids,
                cf["user_ids"],
            )

        if "movie_ids" in cf:
            checks["movie_ids"] = (
                movie_ids,
                cf["movie_ids"],
            )

        if "tmdb_ids" in cf:
            checks["tmdb_ids"] = (
                tmdb_ids,
                cf["tmdb_ids"],
            )

        # Train / Validation / Test 확인
        for split_name, split_df in [
            ("train", train_df),
            ("validation", validation_df),
            ("test", test_df),
        ]:

            candidates = {
                f"{split_name}_user_ids": (
                    split_df["userId"].to_numpy()
                ),
                f"{split_name}_movie_ids": (
                    split_df["movieId"].to_numpy()
                ),
                f"{split_name}_ratings": (
                    split_df["rating"].to_numpy()
                ),
            }

            for key, mine in candidates.items():
                if key in cf:
                    checks[key] = (mine, cf[key])

        all_ok = True

        for key, (mine, theirs) in checks.items():
            same = np.array_equal(mine, theirs)
            all_ok &= same

            print(
                f"{'OK  ' if same else 'DIFF'} {key}"
            )

        print(
            "협업 필터링 분할 일치"
            if all_ok
            else "분할 불일치: 위 DIFF 항목 확인 필요"
        )

else:
    print(
        f"분할 검증 생략: {CF_PATH} 파일이 없습니다."
    )


print("전처리 완료")
print("영화 결과 파일 2개 생성")
print("기존 도서 결과 파일은 공통 장르 및 연도 기준으로 갱신")