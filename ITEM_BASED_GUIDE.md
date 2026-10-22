# Item-based Collaborative Filtering 구현·발표 가이드

## 실행 환경과 명령어

프로젝트 루트의 PowerShell에서 실행합니다. 실제 검증 환경은 Windows / Python 3.14.7이며 주요 라이브러리 버전은 `requirements-item-based.txt`에 고정했습니다.

```powershell
# 최초 환경 구성 (이번 작업에서 .venv 및 패키지 설치 완료)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-item-based.txt

# 소규모 실제 데이터 검증
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 전체 MovieLens 1M 학습·평가 및 사용자 1의 Top-10
.\.venv\Scripts\python.exe scripts/run_item_based.py

# 다른 사용자와 K값 설정 (추천용 K도 비교 목록에 포함)
.\.venv\Scripts\python.exe scripts/run_item_based.py --user-id 100 --k-values 20 30 50 --recommend-k 30 --top-n 10
```

가상환경 활성화 없이 위 명령을 사용할 수 있습니다. VS Code의 Python 인터프리터를 `.venv\Scripts\python.exe`로 지정하면 같은 환경으로 실행됩니다. 실행 스크립트의 기본 데이터/결과 경로는 스크립트 위치로부터 계산하므로 현재 셸 디렉터리에 의존하지 않습니다. 그래프 캐시는 Git에서 제외된 `.venv/matplotlib-cache/`에 둡니다.

## 파일별 역할

| 파일 | 역할 |
|---|---|
| `src/item_based.py` | `ItemBasedCF`, 공통 입력 검증, 사용자별 분할. 영화 메타데이터와 독립 |
| `scripts/run_item_based.py` | MovieLens 어댑터, CLI, 전체 평가, 제목 결합, CSV·JSON·그래프 저장 |
| `tests/test_item_based.py` | 실제 데이터 일부로 코사인·가중합·누수·예외·추천·지표 검증 |
| `requirements-item-based.txt` | 직접 사용하는 라이브러리 버전 |
| `results/item_based/metrics.csv` | K별 오차·CF 커버리지·대체 건수·시간·메모리 |
| `results/item_based/metrics.png` | K별 MAE/RMSE 그래프 |
| `results/item_based/predictions_k*.csv` | 테스트 전체의 실제/예상 평점, 예측 방식, 기여 이웃 수 |
| `results/item_based/recommendations_user_1.csv` | 사용자 1 Top-10, 영화 ID·제목·예상 평점 |
| `results/item_based/test_split.csv` | 사용한 테스트 행. seed와 입력 해시로 재현 가능 |
| `results/item_based/run_summary.json` | 데이터 검수, 분할 설정, 입력 SHA-256, 환경, 총 실행시간·최대 작업집합 |

기존 `src/item_based.py`는 주석 한 줄의 자리표시자여서 실제 구현으로 확장했습니다. 기존 README, .gitignore, 원본 데이터는 수정하지 않았습니다. `data/raw/`, `.venv/`, Python 캐시는 기존 .gitignore로 제외됩니다. Git 브랜치 전환, 커밋, 푸시, 병합은 하지 않습니다.

## 실제 파일 형식과 데이터 확인

- `data/raw/ml-1m/ratings.dat`: `UserID::MovieID::Rating::Timestamp`
- `data/raw/ml-1m/movies.dat`: `MovieID::Title::Genres`
- `data/raw/ml-1m/users.dat`: `UserID::Gender::Age::Occupation::Zip-code`

`pandas.read_csv(sep='::', engine='python', encoding='latin-1')`로 다중문자 구분자와 제목의 특수문자를 처리합니다. users.dat는 존재와 해시를 확인하며, 알고리즘에는 사용자 인구통계 정보가 필요하지 않아 읽어 학습하지 않습니다.

실제 데이터: 사용자 6,040명, 평점이 있는 영화 3,706개, 영화 카탈로그 3,883개, 평점 1,000,209개입니다. ratings와 movies의 각 컬럼 결측은 0개이며, 중복 사용자-영화와 메타데이터 ID, 평점 범위 및 영화 ID 참조도 검증합니다. 카탈로그에만 있는 영화 177개는 CF 벡터가 없어 추천 후보에서 제외됩니다.

| 평점 | 건수 |
|---|---:|
| 1 | 56,174 |
| 2 | 107,557 |
| 3 | 261,197 |
| 4 | 348,971 |
| 5 | 226,310 |

## 알고리즘 설명

1. 사용자별 평점을 seed=42로 무작위 분리합니다. 테스트 수는 사용자마다 `floor(평점 수 × 0.2)`이며 최소 학습 1개를 보장합니다. 평점 하나인 사용자는 학습에만 둡니다. 실제 학습 802,553개, 테스트 197,656개, 사용자별 최소 학습 16개입니다. 이 분할은 미래 시점 예측을 측정하는 시간순 평가가 아닙니다.
2. 학습 데이터의 ID를 행렬 인덱스로 매핑하여 SciPy `csr_matrix` 사용자×아이템 행렬을 생성합니다. 전치한 아이템×사용자 CSR 행렬로 영화를 비교합니다. 저장하지 않은 0은 미평가이며 사용자 평균과 가중평균에 실제 0점으로 포함하지 않습니다.
3. `NearestNeighbors(metric='cosine', algorithm='brute')`를 아이템 행렬에 학습합니다. 희소 입력을 사용하고, 128개 아이템씩 조회하여 전체 영화×영화 유사도 행렬을 보관하지 않습니다. 최종 저장량은 평점 희소 행렬과 아이템 수×K 이웃 목록입니다. sklearn 내부에서 배치×전체 아이템 거리 임시 배열은 생길 수 있습니다.
4. 영화 i, j에 대해 `similarity(i,j) = dot(r_i,r_j) / (norm(r_i) × norm(r_j))`입니다. sklearn의 cosine distance는 `1 - similarity`이므로 `1 - distance`로 유사도를 얻고 자기 자신을 제거합니다. 미평가 차원은 0으로 두는 원시 평점 코사인입니다. 사용자 평균 중심화나 adjusted cosine은 사용하지 않습니다.
5. 대상 영화의 전체 Top-K 이웃 중 사용자 u가 **학습에서 평가한** 영화만 골라 `예상 평점 = Σ(sim(i,j) × rating(u,j)) / Σ(sim(i,j))`로 예측합니다. K는 사용자가 본 영화 중 K개가 아니라, 대상 영화의 전체 이웃 K개입니다. 따라서 실제 기여 이웃은 K보다 적을 수 있습니다. 유효 유사도 임계값은 1e-12입니다.
6. 양수 유사도 이웃이 없으면 학습 사용자 평균, 사용자도 미지이면 학습 영화 평균, 둘 다 미지이면 학습 전체 평균을 반환합니다. `predict_with_details`가 `cf`, `user_mean`, `item_mean`, `global_mean`과 기여 이웃 수를 제공합니다. 테스트 행을 버리지 않고 대체 예측까지 전부 평가합니다.
7. 학습에서 평가하지 않은 학습 카탈로그의 모든 아이템을 예측한 뒤 점수 내림차순, 동점은 ID 오름차순으로 추천합니다. 제목은 실행 스크립트에서 movies와 결합합니다. 테스트 영화는 학습에서 미평가이므로 추천 후보가 될 수 있습니다. 대체 점수도 추천에 들어갈 수 있어 CSV의 method를 함께 확인해야 합니다.

예상 평점은 학습 평점 범위 안에 있습니다. 평균 중심화·인기도 보정·최소 기여 이웃 제한을 넣지 않은 강의 방식이므로 단 한 개의 5점 이웃만으로도 5점 예측이 나올 수 있습니다. 예상 평점 5점이 강한 추천 신뢰도를 뜻하지는 않습니다.

## 평가 해석과 한계

실제 실행 결과(2026-10-22, 테스트 197,656건 전부 평가):

| K | MAE | RMSE | 순수 CF 비율 | 대체 예측 건수 | 학습 초 | 평가 초 |
|---|---:|---:|---:|---:|---:|---:|
| 20 | 0.731815 | 0.958860 | 95.87% | 8,162 | 1.37 | 42.61 |
| 30 | 0.727212 | 0.947272 | 97.42% | 5,103 | 1.31 | 42.50 |
| 50 | 0.725633 | 0.938802 | 98.59% | 2,790 | 1.42 | 43.06 |

각 K에서 학습에 없는 영화에 대한 테스트 평점은 36건입니다. 이들 역시 평균 대체로 평가에 포함했습니다. 총 실행시간은 138.22초, 프로세스 최대 작업집합은 543.30 MiB, 사용자×아이템 및 전치 CSR 버퍼 합은 18.41 MiB였습니다. 소규모 실제 데이터 검증 테스트는 6개 모두 성공했고 로그는 `results/item_based/test_output.txt`에 저장했습니다. 전체 설치 버전은 `environment-lock.txt`에 있습니다.

K=30의 실제 테스트 예시:

| 사용자 | 영화 ID | 실제 평점 | 예상 평점 | 기여 이웃 |
|---|---:|---:|---:|---:|
| 1 | 3408 | 4 | 4.482761 | 4 |
| 1 | 2355 | 5 | 4.161511 | 7 |
| 1 | 595 | 5 | 4.388659 | 13 |
| 1 | 2321 | 3 | 4.407599 | 5 |

사용자 1의 실제 Top-10(K=30):

| 영화 ID | 제목 | 예상 평점 | 기여 이웃 |
|---|---|---:|---:|
| 26 | Othello (1995) | 5.0 | 1 |
| 55 | Georgia (1995) | 5.0 | 1 |
| 94 | Beautiful Girls (1996) | 5.0 | 1 |
| 147 | Basketball Diaries, The (1995) | 5.0 | 1 |
| 161 | Crimson Tide (1995) | 5.0 | 2 |
| 230 | Dolores Claiborne (1994) | 5.0 | 1 |
| 238 | Far From Home: The Adventures of Yellow Dog (1995) | 5.0 | 1 |
| 282 | Nell (1994) | 5.0 | 1 |
| 305 | Ready to Wear (Pret-A-Porter) (1994) | 5.0 | 1 |
| 313 | Swan Princess, The (1994) | 5.0 | 2 |

MAE는 절대오차 평균, RMSE는 제곱오차 평균의 제곱근입니다. 둘 다 작을수록 좋고, RMSE는 큰 오차를 더 강하게 반영합니다. `mae`, `rmse`는 대체 예측을 포함한 전체 테스트 결과, `cf_mae`, `cf_rmse`는 순수 가중평균 예측만의 결과입니다. `n_cf / n_test`가 CF 커버리지입니다. K가 커지면 이웃의 정확도와 커버리지 사이의 변화가 발생할 수 있습니다.

K=20·30·50 비교는 동일 테스트 분할에서의 진단 결과입니다. 이 표에서 가장 좋은 K를 고른 뒤 그 점수를 독립적인 최종 성능이라고 주장하면 선택 편향이 생깁니다. 정식 튜닝은 train/validation/test 3분할 또는 교차검증을 추가해야 합니다. 기본 추천 K=30은 실행 전에 지정했습니다. 오차 지표만 평가하며 Precision@K/NDCG 같은 추천 순위 품질이나 미래 시점 일반화는 아직 검증하지 않았습니다.

시간은 `time.perf_counter`, 메모리는 psutil로 측정합니다. `sparse_matrix_mib`는 두 CSR의 데이터·인덱스 버퍼 합이며 모델 전체 메모리는 아닙니다. `process_rss_mib`는 K별 평가 직후 프로세스 작업집합, `peak_working_set_mib`는 Windows에서의 프로세스 최대 작업집합입니다. 입력 파싱·라이브러리·임시 배열을 포함하므로 순수 모델 크기와 구별합니다.

## Hybrid·Goodbooks 연동

```python
from src.item_based import ItemBasedCF

# train_ratings: user_id, item_id, rating 컬럼의 pandas DataFrame
model = ItemBasedCF(n_neighbors=30).fit(train_ratings)
cf_score = model.predict(user_id=1, item_id=10)
recommendations = model.recommend(user_id=1, top_n=10)
score, method, contributors = model.predict_with_details(1, 10)

# Content 모델의 출력도 동일 평점 범위로 맞춘 후 결합
# hybrid_score = alpha * cf_score + (1 - alpha) * content_score
```

Hybrid 팀원은 같은 사용자/아이템 ID와 같은 train/validation/test 분할을 사용해야 합니다. alpha는 validation에서 결정하고 test는 최종 평가 때만 사용합니다. CF 이웃이 없는 경우 method와 contributors를 이용해 Content 점수의 비중을 높일 수 있습니다. Content 모델이 새 아이템을 처리하더라도 현재 CF recommend는 학습 아이템만 후보로 만들므로 Hybrid에서는 별도 공통 후보 집합을 두고 후보별 predict를 호출하세요.

Goodbooks-10k는 외부 로더에서 `book_id`를 `item_id`로 바꾸고 공통 세 컬럼을 제공하면 됩니다. 도서 제목은 외부 메타데이터에서 결합하며 영화 전용 로더를 사용하지 않습니다. 0점/결측 처리와 중복 평점은 해당 데이터셋 의미에 따라 어댑터에서 정리해야 합니다.

## 발표용 핵심 문장

“학습 평점을 희소 사용자-영화 행렬로 만들고 전치하여 영화를 사용자 평점 벡터로 표현했습니다. 영화 간 코사인 유사도로 가까운 K개 영화를 찾고, 그중 사용자가 평가한 영화의 평점을 유사도 가중평균하여 미평가 영화의 점수를 예측했습니다. 데이터 누수를 막기 위해 테스트 평점은 유사도와 평균 계산에서 제외했으며, 예측 근거가 없으면 학습 평균으로 대체하고 그 건수까지 공개했습니다.”

## 참고 출처

외부 구현 코드를 복사하지 않고 요구한 알고리즘을 직접 구현했습니다. API 동작 및 데이터 출처는 다음 공식 자료를 확인했습니다.

- scikit-learn NearestNeighbors: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
- GroupLens MovieLens 1M: https://grouplens.org/datasets/movielens/1m/
- 데이터 형식·이용 조건: 로컬 `data/raw/ml-1m/README`
- 데이터셋 논문: F. Maxwell Harper, Joseph A. Konstan (2015), *The MovieLens Datasets: History and Context*, DOI: https://doi.org/10.1145/2827872

발표·보고서에서 GroupLens와 데이터셋 논문을 인용하고 원본 데이터를 저장소에 재배포하지 마세요.
