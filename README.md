# machine_learning_term_project

영화 도메인 추천 시스템을 만드는 팀 프로젝트입니다.
학습은 MovieLens 1M, 관련 도메인 테스트는 Goodbooks-10k를 씁니다.
아이템 기반 협업 필터링, 콘텐츠 기반 필터링, 하이브리드를 각각 구현한 뒤 성능을 비교합니다.

## 역할 분담

1. 전처리, 하이브리드 (+시간 된다면 머신러닝)
2. 콘텐츠 기반 필터링
3. 아이템 기반 협업 필터링

## 폴더 구조

아이템 기반은 `origin/feature/item-based`에 이미 있으므로 이 브랜치에는 같은 경로 파일을 두지 않습니다. 나중에 합치면 `src/`, `scripts/`, `tests/`, `results/` 아래로 모입니다.

```
src/
  content_based.py      # 2번 뼈대
  data_loader.py        # 1번 뼈대
  hybrid.py             # 1번 뼈대
  item_based.py         # 3번 (다른 브랜치)
scripts/
  run_content_based.py  # 2번 뼈대
  run_hybrid.py         # 1번 뼈대
  run_item_based.py     # 3번 (다른 브랜치)
tests/
  test_content_based.py # 2번 뼈대
  test_hybrid.py        # 1번 뼈대
  test_item_based.py    # 3번 (다른 브랜치)
results/
  content_based/
  hybrid/
  item_based/           # 3번 (다른 브랜치)
data/raw/ml-1m/         # 학습용 MovieLens (git 제외)
data/raw/goodbooks-10k/ # 테스트용 Goodbooks (git 제외)
data/processed/
notebooks/              # 1번 ML 실험
docs/
```

공개 API는 모델마다 같습니다.

- `fit(train_ratings)`
- `predict(user_id, item_id) -> float`
- `recommend(user_id, top_n=10) -> DataFrame` (`item_id`, `predicted_rating`, `method`, `n_contributors`)



## 데이터

원본 데이터는 git에 올리지 않습니다.
로더는 `src/data_loader.py`의 `load_movielens`, `load_goodbooks`입니다.
Goodbooks의 `book_id`는 `item_id`로 바꿔서 모델 API를 같게 유지합니다.

### 학습: MovieLens 1M

[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)을 받아 아래 경로에 두세요.

```
data/raw/ml-1m/ratings.dat
data/raw/ml-1m/movies.dat
data/raw/ml-1m/users.dat
```

### 테스트: Goodbooks-10k

사용자별 도서 평점이 있어 MovieLens와 같은 아이템 기반 CF를 적용할 수 있습니다.
콘텐츠 기반은 설명 텍스트가 있는 확장판을 쓸 수 있습니다.

- 원본: https://github.com/zygmuntz/goodbooks-10k
- 확장판(설명 텍스트): https://github.com/malcolmosh/goodbooks-10k-extended

```
data/raw/goodbooks-10k/ratings.csv
data/raw/goodbooks-10k/books.csv
```

공통 컬럼명은 `user_id`, `item_id`, `rating`입니다.
예측 점수는 학습 평점 범위(MovieLens·Goodbooks 모두 보통 1~5)로 맞춥니다.

## 환경

```
pip install -r requirements.txt
```

