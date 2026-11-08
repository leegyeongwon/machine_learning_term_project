# 팀 공통 8:1:1 → Item-based → Hybrid Validation 전달

## 실행

프로젝트 루트의 PowerShell에서 실행. 기존 `.venv`의 pandas, NumPy, SciPy, scikit-learn, matplotlib, psutil을 사용하며 추가 설치는 없었다.

```powershell
$env:PYTHONUTF8 = '1'
# NPZ가 없으면 팀 전처리부터 실행
.\.venv\Scripts\python.exe -B scripts/run_item_based_common_split.py

# 재실행은 새 결과 폴더 사용
# 기존 NPZ는 분할·통계·마스크 검증 후 재사용
.\.venv\Scripts\python.exe -B scripts/run_item_based_common_split.py --output-dir results/item_based/common_split_811/rerun_01

# 현재 기본 결과 전체 검증(공통 연동 7개 + 기존 모델 6개)
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

기존 결과는 덮어쓰지 않으므로 재실행 시 새 `--output-dir`이 필요하다. 스크립트 절대경로를 사용하면 다른 작업 디렉터리에서도 실행할 수 있다. 전처리 subprocess의 cwd는 `data/raw/ml-1m/`이다.

실행 브랜치는 `feature/item-based`로 제한한다. Git commit/push/merge는 수행하지 않는다. 기존 NPZ가 현재 전처리 코드·원본 데이터의 분할과 다르면 검증을 중단한다. 원본이나 전처리 코드 변경 시 기존 NPZ 보존 후 재생성이 필요하다.

## 처리 흐름과 코드 역할

1. `preprocessing_and_data/movielens-1m/preprocess_item_based.py`: 팀원의 GitHub preprocessing 브랜치 원문을 새 파일로 추가했다. 원문을 수정하지 않았고 Git blob SHA는 `ae227240c65d17ba47a932f6f7b397293062214f`이다. 출처는 `scripts/common_split_811_source.json`에 기록했다.
2. 팀 스크립트를 그대로 실행해 `data/processed/item_user_data.npz`를 생성한다. 원점수 분할 배열, Train 기준 Z-score/마스크 CSR, 사용자 통계, seed가 들어 있다.
3. `src/common_item_data.py`: `*_movie_ids`를 `item_id`로 매핑하고 원점수 DataFrame을 반환한다. 원본 triple, 중복, ID·평점, Train 통계·Z-score·마스크를 검증한다. 원문의 split_ratings 함수만 AST로 추출해 분할을 대조하므로 최상위 I/O는 재실행하지 않는다.
4. `scripts/run_item_based_common_split.py`: 기존 ItemBasedCF와 기존 evaluate 함수를 재사용한다. train 원점수만 fit하고 validation만 evaluate에 전달한다. 기존 8:2 split 함수를 호출하지 않는다. 기존 평가 함수의 n_test 필드명은 n_validation으로 바꿔 저장한다.
5. K=20/30/50의 Validation MAE로 선택하고, 동률이면 RMSE, 다시 동률이면 작은 K를 선택한다. 선택된 예측 CSV를 Hybrid 전달 파일로 만든다. Train+Validation 재학습이나 Test 평가, Hybrid 가중치 학습은 하지 않는다.
6. `tests/test_common_item_data.py`: 실제 공통 파일 로딩, 분할/마스크, 작은 실제 Train의 fit, 전체 저장 지표 재계산, MAE 우선 및 동률 정책, CSV 일대일 대응과 누락/중복 거부를 검증한다.

`src/item_based.py`, `scripts/run_item_based.py`, 기존 테스트 및 이전 Baseline/임시 실험은 수정하지 않았다.

## 분할과 원점수 모델

새 팀 공통 방식은 전체 평점을 `(user_id, movie_id)`로 정렬한 뒤 sklearn train_test_split을 random_state=42로 두 번 적용하는 전체 평점 무작위 분할이다. 사용자별 할당을 보장하는 예전 임시 분할과 다르다. 이번 실제 데이터에서는 모든 사용자에게 Train이 최소 10개 남았지만 이 방식 자체가 일반적으로 최소 Train 수를 보장하는 것은 아니다.

NPZ의 원점수 키는 `train_user_ids/train_movie_ids/train_ratings`, `validation_user_ids/validation_movie_ids/validation_ratings`, `test_user_ids/test_movie_ids/test_ratings`이다. 원점수는 1~5점이며 기존 모델에는 Train의 원점수만 넣는다. Z-score는 누수 검증용이지 현재 CF 입력이 아니다.

Train 사용자 평균/표준편차(ddof=0, 0/NaN은 유효 표준편차 1)와 mask/rating CSR를 독립적으로 다시 계산해 NPZ와 비교했다. 관측 Z=0도 존재하므로 mask.nnz와 rating.nnz가 다른 것은 정상이다. 전체 카탈로그 ID로 행렬 차원을 확장한 것은 ID/형상 정렬이며 Validation/Test 평점값을 학습한 것은 아니다.

## Hybrid 담당자에게 전달

기본 파일은 `results/item_based/common_split_811/hybrid_ready_validation_predictions.csv`이며 100,021행이다.

```text
user_id,item_id,rating,predicted_rating,method,n_contributors
```

`rating`은 실제 Validation 평점이고 `predicted_rating`은 원래 1~5점 척도이다. `method=cf`는 이웃 가중평균, `user_mean`은 Train 사용자 평균 대체이다. `n_contributors`는 실제 기여 이웃 수이며 fallback은 0이다. 파일은 Validation과 일대일 대응하고 중복/누락 없이 유한한 1~5점인지 확인했다.

```python
import pandas as pd

cf = pd.read_csv('results/item_based/common_split_811/hybrid_ready_validation_predictions.csv')
# content: 공통 Train으로 학습한 Content 모델의 Validation 예측
joined = cf.merge(content, on=['user_id', 'item_id'], how='outer',
                  validate='one_to_one', indicator=True, suffixes=('_cf', '_content'))
assert joined['_merge'].eq('both').all()
# 양쪽 Validation 정답 일치 여부 확인
assert joined['rating_cf'].eq(joined['rating_content']).all()
```

Content용 user_profiles_data.npz가 없어 실제 분할 일치는 미검증 상태이다. Content 파일 확보 후 어댑터의 content_paths 또는 위 일대일 조인으로 ID·정답 일치 여부를 확인한다.

`hybrid_contract.json`에는 선택 K, Train-only 여부, Train/Validation triple 및 NPZ·예측 CSV의 SHA-256, 검증 결과가 있다. Hybrid 통합 조건은 동일 분할과 원점수 척도이다. 결합 가중치는 Validation에서 선택하고 Test는 최종 평가에 사용한다. Validation 선택 점수와 최종 Test 성능은 구분한다.

## 현재 실행 결과

자세한 수치·검증·실행 로그는 `results/item_based/common_split_811/REPORT.md`, `data_validation_report.json`, `validation_metrics.csv`, `test_output.txt`에 있다. 이전 사용자별 8:1:1과는 학습/평가 행이 달라 직접적인 개선/악화 비교를 하지 않는다.

## 데이터와 결과의 Git 제외

신규 `data/processed/.gitignore`는 이번 NPZ만 제외한다. 결과 폴더 내부 `.gitignore`는 생성 결과/로그/캐시를 제외한다. 원본은 기존 `data/raw/` 제외 규칙을 유지한다. 코드·출처 메타데이터·가이드는 새 미추적 파일로 검토 가능하며 아직 commit하지 않았다.

출처: https://github.com/leegyeongwon/machine_learning_term_project/blob/preprocessing/preprocessing_and_data/movielens-1m/preprocess_item_based.py
