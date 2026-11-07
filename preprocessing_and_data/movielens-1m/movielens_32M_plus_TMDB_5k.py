"""
TMDB 영화 5000개와 movielens-32M의 데이터를 영화 기준으로 합치는 코드.
각각에 유용한 feature가 있어서 두 데이터를 합칩니다.
"""
import pandas as pd
import json
from pathlib import Path


# ============================================================
# 1. 경로
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

ML_PATH = ROOT / "preprocessing_and_data" / "movielens-1m"
TMDB_PATH = ROOT / "preprocessing_and_data" / "movielens-1m"
OUTPUT_PATH = ROOT / "data" / "processed"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

LINKS_FILE = ML_PATH / "links.csv"
RATINGS_FILE = ML_PATH / "ratings.csv"
MOVIES_FILE = TMDB_PATH / "tmdb_5000_movies.csv"

FEATURES_OUTPUT = OUTPUT_PATH / "movie_features.csv"
RATINGS_OUTPUT = OUTPUT_PATH / "movie_ratings.csv"


# ============================================================
# 2. JSON 파싱
# ============================================================

def parse_genres(value):
    if pd.isna(value):
        return []

    try:
        data = json.loads(value)

        return [
            item["name"]
            for item in data
            if isinstance(item, dict) and item.get("name")
        ]

    except (json.JSONDecodeError, TypeError):
        return []


# ============================================================
# 3. TMDB 영화 정보 로드
# ============================================================

print("Loading TMDB movies...")

movies = pd.read_csv(MOVIES_FILE)

movies["id"] = pd.to_numeric(
    movies["id"], errors="coerce"
)

movies = movies.dropna(subset=["id"]).copy()
movies["id"] = movies["id"].astype("int64")

movies = movies.drop_duplicates(subset=["id"])

# 장르
movies["genres"] = movies["genres"].apply(parse_genres)

# 상영 시간
movies["runtime"] = pd.to_numeric(
    movies["runtime"], errors="coerce"
)

# 개봉연도
movies["release_date"] = pd.to_datetime(
    movies["release_date"], errors="coerce"
)

movies["release_year"] = movies["release_date"].dt.year

# 줄거리
movies["overview"] = movies["overview"].fillna("")

features = movies[[
    "id",
    "title",
    "genres",
    "runtime",
    "overview",
    "release_year"
]].copy()

features = features.rename(columns={
    "id": "tmdbId"
})

print(f"TMDB movies: {len(features):,}")


# ============================================================
# 4. MovieLens ID 연결
# ============================================================

print("\nLoading MovieLens links...")

links = pd.read_csv(LINKS_FILE)

links["movieId"] = pd.to_numeric(
    links["movieId"], errors="coerce"
)

links["tmdbId"] = pd.to_numeric(
    links["tmdbId"], errors="coerce"
)

links = links.dropna(
    subset=["movieId", "tmdbId"]
).copy()

links["movieId"] = links["movieId"].astype("int64")
links["tmdbId"] = links["tmdbId"].astype("int64")

total_ml_movies = links["movieId"].nunique()

valid_tmdb_ids = set(features["tmdbId"])

mapping = links[
    links["tmdbId"].isin(valid_tmdb_ids)
][["movieId", "tmdbId"]].drop_duplicates(
    subset=["movieId"]
)

print(f"MovieLens movies with TMDB match: {len(mapping):,}")


# ============================================================
# 5. 평점 데이터 연결
# ============================================================

print("\nMerging ratings...")

if RATINGS_OUTPUT.exists():
    RATINGS_OUTPUT.unlink()

total_ratings = 0
matched_ratings = 0

matched_movie_ids = set()
matched_users = set()

CHUNK_SIZE = 500_000

for idx, chunk in enumerate(
    pd.read_csv(
        RATINGS_FILE,
        chunksize=CHUNK_SIZE
    )
):

    total_ratings += len(chunk)

    merged = chunk.merge(
        mapping,
        on="movieId",
        how="inner",
        validate="many_to_one"
    )

    matched_ratings += len(merged)

    matched_movie_ids.update(
        merged["movieId"].unique()
    )

    matched_users.update(
        merged["userId"].unique()
    )

    merged.to_csv(
        RATINGS_OUTPUT,
        mode="w" if idx == 0 else "a",
        header=(idx == 0),
        index=False,
        encoding="utf-8"
    )

    print(
        f"Chunk {idx + 1} | "
        f"Matched ratings: {matched_ratings:,}"
    )


# ============================================================
# 6. 실제 평점이 존재하는 영화만 특징 데이터에 보존
# ============================================================

rated_mapping = mapping[
    mapping["movieId"].isin(matched_movie_ids)
]

rated_tmdb_ids = set(rated_mapping["tmdbId"])

features = features[
    features["tmdbId"].isin(rated_tmdb_ids)
].copy()

# 리스트를 JSON 문자열로 저장
features["genres"] = features["genres"].apply(
    lambda x: json.dumps(x, ensure_ascii=False)
)

features.to_csv(
    FEATURES_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 7. 결과 통계
# ============================================================

rating_coverage = (
    matched_ratings / total_ratings * 100
    if total_ratings else 0
)

movie_coverage = (
    len(matched_movie_ids) / total_ml_movies * 100
    if total_ml_movies else 0
)

print("\n" + "=" * 50)
print("FINAL RESULT")
print("=" * 50)

print(f"Total ratings: {total_ratings:,}")
print(f"Matched ratings: {matched_ratings:,}")
print(f"Rating coverage: {rating_coverage:.2f}%")

print(f"Matched movies: {len(matched_movie_ids):,}")
print(f"Movie coverage: {movie_coverage:.2f}%")

print(f"Matched users: {len(matched_users):,}")

print("\nSaved files:")
print(FEATURES_OUTPUT)
print(RATINGS_OUTPUT)
