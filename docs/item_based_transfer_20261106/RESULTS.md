# Strict 전이 최종 결과 (2026-11-06)

## 결과와 판단

기존 구현을 수정하지 않은 genre-only CF metric distillation의 전체 실행을 완료함. 영화 Train 17,401,383행·Validation 2,175,173행을 사용했고, 선택은 rank=8, Ridge alpha=100, anchor K=10임.

| 모델 | 영화 Validation MAE | 영화 Validation RMSE | Goodbooks Test MAE | Goodbooks Test RMSE |
|---|---:|---:|---:|---:|
| source_global_mean | 0.836722 | 1.055111 | 0.905561 | 1.049362 |
| raw_genre_cosine | 0.886953 | 1.093479 | 1.036270 | 1.216714 |
| cf_metric_distillation | 0.886334 | 1.089380 | 1.005357 | 1.177881 |

협업 증류는 단순 장르 KNN보다 도서 MAE -0.030913, RMSE -0.038832로 낮음. 그러나 영화 Train global mean보다 도서 MAE +0.099796, RMSE +0.128520로 높음. 영화 Validation에서도 global mean보다 나빴음. 결과를 개선하려고 설정을 바꾸거나 재선택하지 않았음.

Teacher permutation의 영화 Validation MAE/RMSE는 0.911207/1.103712, 실제 teacher는 0.886334/1.089380. 예측의 평균 절대 변화 0.181371로 협업 teacher 대응이 예측에 실제 영향을 미침. 하지만 이것이 개인화 또는 좋은 도서 성능을 뜻하지는 않음. rank-8 정규화 협업 표현의 표본 off-diagonal cosine 재구성 MSE도 약 0.221440으로 표현 손실이 존재함.

영화 Train에서 231개 아이템·904,256개 평점을 제외한 고정 설정의 cold audit: 증류 MAE/RMSE 0.891977/1.095705, genre 0.883713/1.095634, global mean 0.854889/1.078341. 이 진단에서도 증류 우수성은 관측되지 않음. 도서와 무관한 source-only 진단이며 튜닝에는 사용하지 않음.

후속 채점용 분석에서 실제 Book Test 평균은 3.920607, 증류 예측 평균은 3.322098, global mean 예측은 3.570683임. 증류의 평균 부호 오차는 -0.598510로 도서에서 낮게 예측하는 경향을 관측함. 원인의 인과 검증은 하지 않았음. 이 통계는 예측·설정 동결 및 최종 평가 후 설명용으로만 계산했고 파라미터를 변경하지 않았음.

## 엄격한 평가 분리

도서 콘텐츠는 source 모델과 설정 해시 고정 이후 로딩. 예측 배열·CSV·책별 점수의 SHA-256 봉인이 먼저 완료됐고 그 다음 Book Test 정답을 읽었음. 총 597,648개 Test 평점만 채점. Book Train/Validation 배열, 영화 Test, 사용자 프로필, plot/year/numeric 배열은 실제 실행에서 한 번도 디코딩하지 않았음. 전체 파일 무결성을 위한 byte checksum은 별도임.

도서 사용자 ID는 길이 정렬 외에 사용하지 않음. 동일한 책은 모든 사용자에게 동일 점수. 영화·도서 숫자 ID가 겹쳐도 source lookup을 하지 않음. 장르 연결이 없는 2,129권(해당 Test 131,115행)은 source 전체 평균으로 fallback함.

## 검증과 보존

- 전용 단위 테스트 12개 및 source-only 실제 Smoke Test 통과.
- 전체 실행 내 독립 검증 통과: 동결 해시, 선택 설정, 정답 후개방, 사용자 ID 불변성, save/load, CSV·NPZ 일치, MAE/RMSE 재계산.
- 실행 시작 전 존재하던 관련 파일 399개 내용·크기·mtime 변경 0개.
- 1단계 초기 원본 270개 별도 재검증도 변경 0개. Git/GitHub 사용 0회.
- 모델/데이터/그래프까지 104.00초. 원본 재검증과 독립 검증을 포함한 실행 완료 시각은 시작 후 112.13초. 최고 working set 2112.80 MiB.

## 생성 파일과 실행

새 코드: `src/item_based_transfer.py`, `src/item_based_transfer_data.py`, `scripts/run_item_based_strict_transfer.py`, `scripts/verify_item_based_strict_transfer.py`, `tests/test_item_based_strict_transfer.py`. 새 문서: `docs/item_based_transfer_20261106/`.

전체 결과: `results/item_based/strict_transfer_20261106_0007_full/`. 모델은 `source_model_frozen.npz`, 설정은 `selected_config.json`, 예측은 `book_test_predictions_frozen.csv/.npz`, 책별 점수는 `book_item_scores_frozen.csv`. 실행 로그·seal·array_access·독립 검증·전체 보고서는 같은 폴더에 있음.

```powershell
.venv\Scripts\python.exe -B scripts/run_item_based_strict_transfer.py --smoke
.venv\Scripts\python.exe -B scripts/run_item_based_strict_transfer.py
.venv\Scripts\python.exe -B scripts/verify_item_based_strict_transfer.py --run-dir results/item_based/strict_transfer_20261106_0007_full
```

복구 지점: `local_recovery/item_based_20261105_234311_KST/`, 96개 파일의 SHA-256 원본/사본 일치. RESTORE.md 참조. 기존 코드를 수정하지 않았으므로 기존 모델/전처리/NPZ에 되돌릴 변경은 없음.

영화 Validation이 cross-domain 성능을 보장하지 않으며 upstream 데이터 선택 과정은 NPZ에서 되돌릴 수 없음. 논문 발상과 구현의 구분, 정규화 수식, API/Hybrid 비호환성은 README.md와 전체 REPORT.md에 기록함.
