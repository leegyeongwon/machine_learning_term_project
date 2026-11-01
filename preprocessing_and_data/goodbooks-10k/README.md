## 도서 데이터 소스
원본 저장소: https://github.com/zygmuntz/goodbooks-10k
설명 텍스트가 추가된 확장판: https://github.com/malcolmosh/goodbooks-10k-extended

# preprocess_item_based.py 데이터 전처리

### 1. 데이터 로드
- `goodbooks-10k-extended`의 도서 정보 및 `goodbooks-10k`의 평점 데이터 로드
- 도서 정보에서 장르 등 콘텐츠 정보 활용을 위한 데이터 준비

### 2. 평점 데이터 분할
- 전체 평점 데이터를 **Train : Validation : Test = 8 : 1 : 1** 비율로 분할
- `random_state=42`를 사용하여 재현성 확보

### 3. 유저-도서 인덱스 생성
- 전체 데이터에 등장하는 유저 및 도서 ID 추출
- 각 ID를 희소행렬의 행과 열 인덱스로 변환
- Train 평점만 사용하여 행렬 구성

### 4. 평점 행렬 생성 및 정규화
- 유저를 행, 도서를 열로 하는 CSR 희소행렬 생성
- Train 평점을 기준으로 유저별 Z-score 정규화 수행
- 유저별 평균과 표준편차 별도 계산 및 저장

### 5. 평가 여부 마스크 생성
- Train에서 실제로 평가한 도서 항목을 `1`로 표시
- 미평가 항목은 `0`으로 처리
- 평점 행렬과 동일한 구조의 CSR 희소행렬 생성

### 6. 데이터 저장 및 복원
- 정규화 평점 행렬과 평가 여부 마스크를 CSR 구성 요소별로 저장
- Train / Validation / Test 원본 평점 데이터 저장
- 저장된 NPZ 파일을 다시 로드하여 CSR 행렬 복원 및 데이터 확인

### 7. 최종 결과
- `book_item_user_data.npz` 파일 생성
- Item-based Collaborative Filtering의 도서 간 유사도 계산 및 추천 성능 평가에 활용

### 8. 저장되는 데이터 (`book_item_user_data.npz`)

| 데이터 | 설명 |
|---|---|
| `rating_data` | 정규화 평점 CSR 데이터 |
| `rating_indices` | 평점 CSR 열 인덱스 |
| `rating_indptr` | 평점 CSR 행 포인터 |
| `rating_shape` | 평점 행렬 크기 |
| `mask_data` | 평가 여부 CSR 데이터 |
| `mask_indices` | 평가 여부 CSR 열 인덱스 |
| `mask_indptr` | 평가 여부 CSR 행 포인터 |
| `mask_shape` | 평가 여부 행렬 크기 |
| `user_ids` | 전체 유저 ID |
| `book_ids` | 전체 도서 ID |
| `user_means` | 유저별 Train 평점 평균 |
| `user_stds` | 유저별 Train 평점 표준편차 |
| `train_user_ids` | Train 유저 ID |
| `train_book_ids` | Train 도서 ID |
| `train_ratings` | Train 원본 평점 |
| `validation_user_ids` | Validation 유저 ID |
| `validation_book_ids` | Validation 도서 ID |
| `validation_ratings` | Validation 원본 평점 |
| `test_user_ids` | Test 유저 ID |
| `test_book_ids` | Test 도서 ID |
| `test_ratings` | Test 원본 평점 |
| `random_state` | 데이터 분할 시 사용한 난수 시드 |

# preprocess_content_based.py 데이터 전처리

### 1. 데이터 로드 및 분할
- `goodbooks-10k-extended`의 도서 정보와 `goodbooks-10k`의 평점 데이터 로드
- 전체 평점 데이터를 **Train : Validation : Test = 8 : 1 : 1** 비율로 분할
- `random_state=42`를 사용하여 재현성 확보
- 협업 필터링과 동일한 데이터 분할 및 ID 정렬 기준 유지

### 2. 도서 콘텐츠 특성 추출
- **장르:** 멀티 핫 인코딩 후 행별 L2 정규화
- **작가:** 멀티 핫 인코딩 후 행별 L2 정규화
- **페이지 수 / 출판 연도:** Min-Max 정규화 (0~1)
- **줄거리:** BGE-M3 모델을 사용하여 임베딩 생성 후 L2 정규화
- 결측 데이터는 별도 마스크로 관리하거나 중앙값 및 기본값으로 처리

### 3. 유저 선호 프로필 생성
- 데이터 누수를 방지하기 위해 Train 평점만 사용
- 평점 중심화: `rating - 3.0`
- 3점 초과 평점만 선호도 계산에 반영
  - 4점: 가중치 1
  - 5점: 가중치 2
- 장르, 작가, 줄거리별 유저 선호 벡터 생성 후 L2 정규화
- 페이지 수와 출판 연도는 선호 평점의 가중평균으로 프로필 생성

### 4. 데이터 저장
- 도서 콘텐츠 정보와 유저 프로필을 각각 NPZ 파일로 압축 저장
- 장르 및 작가 프로필은 CSR 희소행렬 구성 요소별로 저장
- Train / Validation / Test 원본 평점 데이터 함께 저장
- 협업 필터링 데이터와 유저 ID, 도서 ID 및 분할 결과가 일치하는지 검증

### 5. 최종 결과
- `book_content.npz`: 도서별 콘텐츠 특성 저장
- `book_user_profiles.npz`: 유저별 선호 프로필 및 평점 분할 데이터 저장
- 두 파일을 통해 콘텐츠 기반 추천 및 유사도 계산에 활용

### 6. 저장되는 데이터

**① book_content.npz**

| 데이터 | 설명 |
|---|---|
| `book_ids` | 전체 도서 ID |
| `titles` | 도서 제목 |
| `genre_names` | 전체 장르 종류 |
| `genres_data` | 장르 CSR 데이터 |
| `genres_indices` | 장르 CSR 열 인덱스 |
| `genres_indptr` | 장르 CSR 행 포인터 |
| `genres_shape` | 장르 행렬 크기 |
| `author_names` | 전체 작가 이름 |
| `authors_data` | 작가 CSR 데이터 |
| `authors_indices` | 작가 CSR 열 인덱스 |
| `authors_indptr` | 작가 CSR 행 포인터 |
| `authors_shape` | 작가 행렬 크기 |
| `numeric_feature_names` | 수치형 특성 이름 (pages, publish_year) |
| `numeric_features` | 정규화된 페이지 수 및 출판 연도 |
| `pages_missing` | 페이지 수 결측 여부 |
| `year_missing` | 출판 연도 결측 여부 |
| `pages_min`, `pages_max` | 페이지 수 정규화 기준 최솟값 및 최댓값 |
| `year_min`, `year_max` | 출판 연도 정규화 기준 최솟값 및 최댓값 |
| `plot_embeddings` | BGE-M3 줄거리 임베딩 |
| `plot_valid_mask` | 줄거리 유효 여부 |
| `plot_dim` | 줄거리 임베딩 차원 |

**② book_user_profiles.npz**

| 데이터 | 설명 |
|---|---|
| `user_ids` | 전체 유저 ID |
| `genre_names` | 장르 종류 |
| `user_genre_profiles_data` | 유저 장르 프로필 CSR 데이터 |
| `user_genre_profiles_indices` | 장르 CSR 열 인덱스 |
| `user_genre_profiles_indptr` | 장르 CSR 행 포인터 |
| `user_genre_profiles_shape` | 장르 프로필 행렬 크기 |
| `author_names` | 작가 종류 |
| `user_author_profiles_data` | 유저 작가 프로필 CSR 데이터 |
| `user_author_profiles_indices` | 작가 CSR 열 인덱스 |
| `user_author_profiles_indptr` | 작가 CSR 행 포인터 |
| `user_author_profiles_shape` | 작가 프로필 행렬 크기 |
| `numeric_feature_names` | 수치형 특성 이름 |
| `numeric_profiles` | 유저별 페이지 수 및 출판 연도 선호 프로필 |
| `plot_profiles` | 유저별 줄거리 선호 임베딩 |
| `train_user_ids` | Train 유저 ID |
| `train_book_ids` | Train 도서 ID |
| `train_ratings` | Train 원본 평점 |
| `validation_user_ids` | Validation 유저 ID |
| `validation_book_ids` | Validation 도서 ID |
| `validation_ratings` | Validation 원본 평점 |
| `test_user_ids` | Test 유저 ID |
| `test_book_ids` | Test 도서 ID |
| `test_ratings` | Test 원본 평점 |
| `random_state` | 데이터 분할 시 사용한 난수 시드 |