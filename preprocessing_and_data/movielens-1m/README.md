## 데이터 소스

- 영화 및 유저 평점 데이터: MovieLens: https://grouplens.org/datasets/movielens/1m/

 # preprocess_item_based.py 데이터 전처리

### 1. 평점 데이터 분할
- 전체 평점 데이터를 **Train : Validation : Test = 8 : 1 : 1** 비율로 분할
- `random_state=42`를 사용하여 재현성 확보
- 분할 후 타임스탬프 제거

### 2. 유저-영화 평점 행렬 생성
- Train 데이터만 사용하여 유저-영화 평점 행렬 생성
- 전체 유저 및 영화 ID를 기준으로 행렬 구성
- 실제 평점이 존재하는 항목과 미평가 항목을 구분하는 마스크 생성

### 3. 평점 정규화
- Train 데이터를 기준으로 유저별 Z-score 정규화 수행
- 미평가 항목은 계산을 위해 0으로 변환
- 유저별 평균과 표준편차 별도 저장

### 4. 희소행렬 변환 및 저장
- 정규화 평점 행렬과 평가 여부 마스크를 CSR 희소행렬로 변환
- CSR 구성 요소(`data`, `indices`, `indptr`, `shape`)를 개별 저장
- 유저 및 영화 ID, 유저별 통계량, Train / Validation / Test 원본 평점도 함께 저장

### 5. 최종 결과
- `item_user_data.npz` 파일 생성
- 저장된 파일을 다시 로드하여 CSR 행렬 복원 가능
- Item-based Collaborative Filtering에서 영화 간 유사도 계산 및 성능 평가에 활용

### 6. 저장되는 데이터 (`item_user_data.npz`)

| 데이터 | 설명 |
|---|---|
| `rating_data`, `rating_indices`, `rating_indptr`, `rating_shape` | Train 정규화 평점 CSR 행렬 |
| `mask_data`, `mask_indices`, `mask_indptr`, `mask_shape` | Train 평가 여부 CSR 행렬 |
| `user_ids` | 전체 유저 ID |
| `movie_ids` | 전체 영화 ID |
| `user_means` | 유저별 Train 평점 평균 |
| `user_stds` | 유저별 Train 평점 표준편차 |
| `train_user_ids` | Train 유저 ID |
| `train_movie_ids` | Train 영화 ID |
| `train_ratings` | Train 원본 평점 |
| `validation_user_ids` | Validation 유저 ID |
| `validation_movie_ids` | Validation 영화 ID |
| `validation_ratings` | Validation 원본 평점 |
| `test_user_ids` | Test 유저 ID |
| `test_movie_ids` | Test 영화 ID |
| `test_ratings` | Test 원본 평점 |
| `random_state` | 데이터 분할 시 사용한 난수 시드 |

# preprocess_content_based.py 데이터 전처리

### 1. 영화 데이터 전처리
- 영화 데이터에서 장르 정보 추출
- `MultiLabelBinarizer`를 사용하여 장르 멀티 핫 인코딩
- 영화별 장르 벡터에 L2 정규화 적용

### 2. 평점 데이터 분할
- 전체 평점 데이터를 **Train : Validation : Test = 8 : 1 : 1** 비율로 분할
- `random_state=42`를 사용하여 재현성 확보
- Item-based 데이터와 동일한 분할 설정 및 정렬 기준 사용

### 3. 유저 프로필 생성
- Train 평점만 사용하여 유저별 선호도 프로필 생성
- 평점 중심화: `rating - 3.0`
- 영화의 장르 벡터에 중심화된 평점을 적용하여 유저별 장르 선호도 합산
- 유저 프로필에 L2 정규화 적용

### 4. 데이터 저장 및 복원
- 압축된 NPZ 형식으로 영화 정보와 유저 프로필 분리 저장
- 저장된 데이터를 다시 로드하여 DataFrame 복원 및 데이터 확인

### 5. 최종 결과
- `movies_multihot_data.npz`: 영화별 정규화 장르 벡터 저장
- `user_profiles_data.npz`: 유저별 정규화 선호도 프로필 및 Train / Validation / Test 원본 평점 저장

### 6. 저장되는 데이터

**① movies_multihot_data.npz**

| 데이터 | 설명 |
|---|---|
| `multi_hot_norm` | 영화별 L2 정규화 장르 벡터 |
| `classes` | 전체 장르 종류 |
| `movie_ids` | 영화 ID |
| `titles` | 영화 제목 |

**② user_profiles_data.npz**

| 데이터 | 설명 |
|---|---|
| `user_ids` | 유저 프로필이 생성된 유저 ID |
| `user_profiles_norm` | 유저별 L2 정규화 선호도 프로필 |
| `genre_classes` | 유저 프로필의 장르 종류 및 순서 |
| `train_user_ids` | Train 유저 ID |
| `train_movie_ids` | Train 영화 ID |
| `train_ratings` | Train 원본 평점 |
| `validation_user_ids` | Validation 유저 ID |
| `validation_movie_ids` | Validation 영화 ID |
| `validation_ratings` | Validation 원본 평점 |
| `test_user_ids` | Test 유저 ID |
| `test_movie_ids` | Test 영화 ID |
| `test_ratings` | Test 원본 평점 |
| `random_state` | 데이터 분할 시 사용한 난수 시드 |