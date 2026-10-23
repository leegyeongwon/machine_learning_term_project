# machine_learning_term_project

MovieLens 1M으로 추천 시스템을 만드는 팀 프로젝트입니다.
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
data/raw/ml-1m/         # 원본 (git 제외)
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
[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)을 받아 압축을 풀고 아래 경로에 두세요.

```
data/raw/ml-1m/ratings.dat
data/raw/ml-1m/movies.dat
data/raw/ml-1m/users.dat
```

공통 컬럼명은 `user_id`, `item_id`, `rating`입니다.
예측 점수는 학습 평점 범위(MovieLens는 1~5)로 맞춥니다.

## 환경

```
pip install -r requirements.txt
```
