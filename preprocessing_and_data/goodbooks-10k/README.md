## 도서 데이터 소스
원본 저장소: https://github.com/zygmuntz/goodbooks-10k
설명 텍스트가 추가된 확장판: https://github.com/malcolmosh/goodbooks-10k-extended

# 도서 추천 데이터 전처리

도서 추천 실험에 쓰는 입력 데이터를 만드는 두 개의 전처리 스크립트입니다.

| 스크립트 | 용도 | 결과 파일 |
|---|---|---|
| `preprocess_item_based.py` | 협업 필터링용 평점 행렬 | `book_item_user_data.npz` |
| `preprocess_content_based.py` | 콘텐츠 기반 추천용 도서 특성 + 유저 프로필 | `book_content.npz`, `book_user_profiles.npz` |

모든 결과 파일은 `data/processed/`에 저장됩니다.

## 공통: 데이터와 분할

- 평점: goodbooks-10k `ratings.csv` (user_id, book_id, rating)
- 평점을 **Train / Validation / Test = 8:1:1** 로 분할 (`random_state=42`, `train_test_split`)
- 두 스크립트의 분할이 완전히 같아야 하므로, 평점 원본은 필터링·정렬 없이 그대로 사용
- 유저/도서 인덱스는 전체 평점에 등장하는 ID를 오름차순 정렬해 부여

---

## 1. `preprocess_item_based.py` (협업 필터링)

### 가공 방식

1. Train 평점으로 유저 × 도서 희소행렬(CSR) 생성
2. **유저별 Z-score 정규화** (Train 기준 평균/표준편차, 모집단 분산, 표준편차 0이면 1로 대체)
3. 실제 평점이 있는 위치를 1로 표시한 **평가 여부 마스크** 행렬 생성

### 저장: `book_item_user_data.npz`

| 키 | 내용 |
|---|---|
| `rating_data / indices / indptr / shape` | Train 정규화 평점 CSR |
| `mask_data / indices / indptr / shape` | Train 평가 여부 마스크 CSR |
| `user_ids`, `book_ids` | 전체 유저·도서 ID |
| `user_means`, `user_stds` | Train 기준 유저별 평균·표준편차 |
| `train_* / validation_* / test_*` (`user_ids`, `book_ids`, `ratings`) | 분할된 원본 평점 |
| `random_state` | 분할 시드 |

---

## 2. `preprocess_content_based.py` (콘텐츠 기반)

### 가공 방식

**도서 특성** (평점을 쓰지 않으므로 전체 도서 대상)

- 장르 / 저자: 멀티핫 인코딩 + 행별 L2 정규화 (※ 저자는 코드에 남아 있으나 사용하지 않기로 함)
- 페이지 수 / 출판 연도: 결측은 중앙값으로 채우고 Min-Max로 0~1 변환 (두 자리 연도 오류 보정 포함)
- 줄거리: BGE-M3 dense 임베딩 + L2 정규화 (줄거리가 없는 책은 0 벡터, GPU 권장)

**유저 선호 프로필** (데이터 누수 방지를 위해 Train 평점만 사용)

- 평점을 3점 기준으로 중심화, 3점 초과(4점=가중치 1, 5점=가중치 2)만 선호로 반영
- 장르/저자 프로필: 선호 가중치 × 도서 특성 합산 후 L2 정규화
- 줄거리 프로필: 중심화 평점 × 줄거리 임베딩 합산 후 L2 정규화
- 페이지/연도 프로필: 선호 도서의 가중 평균 (선호 도서가 없으면 0.5)

**유사도 계산 (이후 단계 예시)**

- 줄거리·장르(·저자): 코사인 유사도 → (A+1)/2 로 0~1 변환
- 페이지·연도: |유저 선호값 − 도서값| → (1−A) 로 변환
- 최종 점수 = 각 요소 × 가중치(w1~w5) 합. 가중치는 Validation으로 정하고 Test로 평가

### 저장

**`book_content.npz`** (도서 콘텐츠)

| 키 | 내용 |
|---|---|
| `book_ids`, `titles` | 도서 ID, 제목 |
| `genres_*`, `authors_*` | 장르·저자 CSR (`data / indices / indptr / shape`) |
| `genre_names`, `author_names` | 멀티핫 열 이름 |
| `numeric_features`, `numeric_feature_names` | 페이지·연도 (0~1) |
| `pages_missing`, `year_missing` | 결측 여부 |
| `pages_min/max`, `year_min/max` | Min-Max 원래 범위 |
| `plot_embeddings`, `plot_valid_mask`, `plot_dim` | 줄거리 임베딩, 유효 여부, 차원 |

**`book_user_profiles.npz`** (유저 프로필 + 분할)

| 키 | 내용 |
|---|---|
| `user_ids` | 전체 유저 ID |
| `user_genre_profiles_*`, `user_author_profiles_*` | 장르·저자 프로필 CSR |
| `plot_profiles` | 줄거리 프로필 |
| `numeric_profiles` | 페이지·연도 선호값 |
| `train_* / validation_* / test_*` | 분할된 원본 평점 |
| `random_state` | 분할 시드 |

### 분할 일치 검증

스크립트 마지막에 `book_item_user_data.npz`가 있으면 유저/도서 ID와 Train·Validation·Test 평점이 협업 필터링 쪽과 동일한지 자동 비교합니다. 실행 순서는 `preprocess_item_based.py` → `preprocess_content_based.py`를 권장합니다.
