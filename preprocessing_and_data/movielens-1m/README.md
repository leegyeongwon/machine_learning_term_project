## 데이터 소스

- 영화 및 유저 평점 데이터: MovieLens: https://grouplens.org/datasets/movielens/1m/

# 영화 추천 데이터 전처리

영화 추천 실험에 쓰는 입력 데이터를 만드는 두 개의 전처리 스크립트입니다. 두 번째 스크립트는 도서 전처리 결과와 **같은 특성 공간**을 쓰도록 도서 파일도 함께 갱신합니다.

| 스크립트 | 용도 | 결과 파일 |
|---|---|---|
| `preprocess_item_based.py` | 협업 필터링용 평점 행렬 | `item_user_data.npz` |
| `preprocess_content_based.py` | 콘텐츠 기반 추천용 영화 특성 + 유저 프로필, 도서 파일 갱신 | `movie_content.npz`, `movie_user_profiles.npz` (+ 갱신된 `book_content.npz`, `book_user_profiles.npz`) |

모든 결과 파일은 `data/processed/`에 저장됩니다.

## 공통: 데이터와 분할

- 입력: `movie_ratings.csv` (userId, movieId, rating, timestamp, tmdbId), `movie_features.csv` (tmdbId, title, genres, runtime, overview, release_year)
- 평점을 **userId, movieId 순으로 정렬**한 뒤 **Train / Validation / Test = 8:1:1** 로 분할 (`random_state=42`)
- 두 스크립트가 같은 정렬·분할을 쓰므로 정렬 순서를 바꾸면 분할이 달라짐
- movieId와 tmdbId는 1:1로 연결되어야 하며, 아니면 오류 발생

---

## 1. `preprocess_item_based.py` (협업 필터링)

### 가공 방식

1. 행 인덱스만 8:1:1로 분할하고, Train 평점으로 유저 × 영화 희소행렬(CSR) 생성
2. **유저별 Z-score 정규화** (Train 기준 평균/표준편차, 모집단 분산, 표준편차 0이면 1로 대체)
3. 평점이 있는 위치를 1로 표시한 **평가 여부 마스크** 행렬 생성
4. 정규화 결과가 정확히 0인 원소는 평점 행렬에서 제거 (관측 여부는 마스크로 판단)

### 저장: `item_user_data.npz`

| 키 | 내용 |
|---|---|
| `rating_data / indices / indptr / shape` | Train 정규화 평점 CSR |
| `mask_data / indices / indptr / shape` | Train 평가 여부 마스크 CSR |
| `user_ids`, `movie_ids`, `tmdb_ids` | 전체 유저·영화 ID, 영화별 TMDB ID |
| `user_means`, `user_stds` | Train 기준 유저별 평균·표준편차 |
| `train_* / validation_* / test_*` (`user_ids`, `movie_ids`, `tmdb_ids`, `ratings`) | 분할된 원본 평점 |
| `random_state` | 분할 시드 |

---

## 2. `preprocess_content_based.py` (콘텐츠 기반 + 도메인 정렬)

### 가공 방식

**도서와 영화의 공통 공간 만들기**

- 장르: 도서 장르와 영화 장르의 **합집합**을 공통 장르 사전으로 만들고, 도서 장르 행렬·도서 유저 장르 프로필의 열을 이 사전에 맞춰 재배치
- 연도: 도서 출판 연도와 영화 개봉 연도를 합쳐 **공통 중앙값(결측 대체)과 공통 Min-Max 범위**로 정규화
- 연도 범위가 바뀌었으므로 **도서 수치형 프로필을 Train 평점으로 다시 계산**

**영화 특성**

- 장르: 공통 장르 사전 기준 멀티핫 + 행별 L2 정규화 (딕셔너리·구분자 문자열 등 여러 형식 처리)
- 상영 시간: 결측·0 이하는 중앙값으로 채우고 Min-Max 정규화
- 개봉 연도: 도서와 같은 공통 범위로 정규화
- 줄거리: BGE-M3 dense 임베딩 + L2 정규화 (줄거리가 없는 영화는 0 벡터, GPU 권장)

**영화 유저 선호 프로필** (Train 평점만 사용)

- 평점을 3점 기준으로 중심화, 3점 초과(4점=가중치 1, 5점=가중치 2)만 선호로 반영
- 장르 프로필: 선호 가중치 × 영화 장르 합산 후 L2 정규화
- 수치형 프로필: 선호 영화의 가중 평균 (선호 영화가 없으면 0.5)
- 줄거리 프로필: 중심화 평점 × 줄거리 임베딩 합산 후 L2 정규화 (유저 1만 명 단위로 나눠 계산)

### 저장

**`movie_content.npz`** (영화 콘텐츠)

| 키 | 내용 |
|---|---|
| `movie_ids`, `tmdb_ids`, `titles` | 영화 ID, TMDB ID, 제목 |
| `genres_*` | 장르 CSR (`data / indices / indptr / shape`) |
| `genre_names` | 공통 장르 사전 |
| `numeric_features`, `numeric_feature_names` | 상영 시간·개봉 연도 (0~1) |
| `runtime_missing`, `year_missing` | 결측 여부 |
| `runtime_min/max`, `year_min/max` | Min-Max 범위 (연도는 도서와 공통) |
| `plot_embeddings`, `plot_valid_mask`, `plot_dim` | 줄거리 임베딩, 유효 여부, 차원 |

**`movie_user_profiles.npz`** (유저 프로필 + 분할)

| 키 | 내용 |
|---|---|
| `user_ids`, `genre_names` | 전체 유저 ID, 공통 장르 사전 |
| `user_genre_profiles_*` | 장르 프로필 CSR |
| `plot_profiles` | 줄거리 프로필 |
| `numeric_profiles` | 상영 시간·개봉 연도 선호값 |
| `train_* / validation_* / test_*` | 분할된 원본 평점 (`user_ids`, `movie_ids`, `tmdb_ids`, `ratings`, `timestamps`) |
| `random_state` | 분할 시드 |

**갱신되는 도서 파일** (`book_content.npz`, `book_user_profiles.npz`)

- 장르 행렬과 도서 유저 장르 프로필이 공통 장르 차원으로 확장됨
- 도서 연도 특성과 `year_min/max`가 공통 범위로 바뀌고, 도서 수치형 프로필이 재계산됨
- 같은 경로의 기존 파일을 **덮어씀**

### 실행 순서와 검증

1. 도서 `preprocess_content_based.py` → `book_content.npz`, `book_user_profiles.npz` 생성
2. 영화 `preprocess_item_based.py` → `item_user_data.npz` 생성
3. 영화 `preprocess_content_based.py` → 영화 결과 생성 + 도서 파일 갱신

스크립트 마지막에 `item_user_data.npz`가 있으면 유저/영화 ID와 Train·Validation·Test 평점이 협업 필터링 쪽과 동일한지 자동 비교합니다. 도서 두 파일이 없으면 오류가 납니다.
