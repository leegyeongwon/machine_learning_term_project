# Hybrid 담당자 연결 지침 — 기존 코드는 수정하지 않음

이 문서는 연결 시 담당자가 검토·수정해야 할 지점만 설명합니다. 이번 작업에서는 Hybrid/Content/shared runner를 수정하거나 Hybrid를 새로 학습·평가하지 않았습니다.

## 현재 연결이 그대로는 실패하는 지점

| 위치 | 현재 동작 | 담당자에게 필요한 변경 |
|---|---|---|
| `src/hybrid.py`, `HybridCF.fit` | `item_model.fit(train_ratings)` 호출 | 새 Item은 `source_content`가 필요하므로 ValueError. frozen 모델을 받는 별도 결합 경로가 필요하며 기존 fit으로 재학습하면 안 됨 |
| `HybridCF.fit`의 `item_model.item_ids_` 접근 | 두 모델 영화 ID의 합집합을 추천 후보로 사용 | 새 모델은 source `anchor_ids_`와 target `bound_ids_`를 분리함. frozen book catalog만 추천 후보로 사용. 영화 ID와 책 ID를 합치지 않음 |
| `HybridCF.fit/recommend`의 `seen_` | Train의 사용자별 아이템을 필터링 | 영화 이력을 책의 seen으로 쓰지 않음. Book Train/Test 평점으로 seen을 만들면 strict 위반. 현재 strict 경로에는 사용자별 seen이 없음 |
| `scripts/run_full_recsys_pipeline.py:80` | `scores, methods, contributors = model.predict_many(...)` | 새 Item의 반환은 ndarray 하나 `(N,)`임. tuple unpack을 제거하는 전용 분기/어댑터가 필요 |
| shared runner의 `book_content=content.transfer(..., books['train'])` | 도서 지원 평점으로 사용자 프로필 생성 | strict 금지. 이 경로 재사용 불가. Content 담당자의 별도 support-free 추론 설계 필요 |
| shared runner의 `ItemBasedCF(...).fit(books['train'])` | 도서 평점으로 책 간 유사도 재학습 | strict 금지. frozen ItemBasedTransfer를 로딩하고 콘텐츠 bind만 수행 |
| 기존 보고서 생성기 | 내부 Git 상태 조회 | Git 금지 세션에서는 호출하지 않음. 전용 보고서 경로 사용 |

`HybridCF.predict` 자체의 가중합은 개별 모델의 float 예측과 연결할 수 있지만, 기존 `fit`과 추천 상태 초기화까지 호환된다는 뜻은 아닙니다. 기존 source에 묵시적인 호환 수정을 넣지 말고 별도 frozen wrapper/진입점을 설계해야 합니다.

## 올바른 frozen Item 사용 흐름

다음은 연결 설명용 예시이며 여기서 실행한 코드가 아닙니다.

```python
from src.item_based_transfer import ItemBasedTransfer
from src.item_based_transfer_data import StrictArrayReader

reader = StrictArrayReader(processed_dir)
# 영화 Validation으로 확정한 모델/설정만 전달
reader.allow_target_features([source_model_path, selected_config_path])
book_catalog = reader.catalog("book")
item = ItemBasedTransfer.load(source_model_path)
item.bind_target_content(book_catalog, namespace="book")
users = reader.read("book_item_user_data.npz", "test_user_ids")
books = reader.read("book_item_user_data.npz", "test_book_ids")
item_predictions = item.predict_many(users, books)  # ndarray; tuple이 아님
```

`bind_target_content`는 frozen 영화 장르 사전·Ridge·협업 anchor를 변환·조회하며 fit하지 않습니다. 사용자 ID 값은 완전히 무시합니다. 모든 사용자에게 같은 책의 점수는 같습니다. 같은 정수 ID의 영화 사용자/영화 아이템을 조회하지 않습니다. 미등록 책 fallback도 영화 Train global mean이며 영화 ID lookup이 아닙니다.

## CSV 정렬과 정답 분리

1. `book_test_predictions_frozen.csv`는 `user_id, book_id, distilled, genre, global_mean`을 갖고 정답 rating은 포함하지 않습니다. 새 Item 점수 열은 `distilled`입니다.
2. 담당자 결합 포맷에서 `book_id -> item_id`, `distilled -> predicted_rating`로 이름만 변환할 수 있습니다. `domain="book"`을 명시하거나 파일·객체 수준으로 domain을 격리합니다. 이름을 바꾼 숫자 ID를 영화 모델 ID로 해석하면 안 됩니다.
3. 두 모델의 **동일 Book Test 쌍**을 `(domain, user_id, item_id)`로 일대일 정렬하고 중복·누락·범위·유한성을 확인합니다. 파일 행 순서가 같다고 가정하지 않습니다. 원본에 중복 평점 쌍이 있으면 조용히 제거하지 말고 안정적인 row ID 계약 또는 오류로 처리합니다.
4. 책별 점수 테이블 `book_item_scores_frozen.csv`는 책당 한 행입니다. 이를 사용자·책 Test ID에 붙일 때는 `book_id` 기준 `many_to_one` 검증을 사용합니다. 사용자별 평점 예측끼리 결합할 때의 `one_to_one` 검증과 구분합니다.
5. Item·Content 예측 및 최종 Hybrid 예측을 저장하고 설정·모델·CSV/배열 해시를 먼저 봉인합니다. 그 후에만 Book Test rating을 열어 정답 테이블로 별도 join하고 MAE/RMSE를 계산합니다. `align_predictions`처럼 rating을 필수로 받는 기존 helper를 예측 전 단계에 그대로 호출하면 안 됩니다.
6. source Movie Validation 파일 `movie_validation_predictions.csv`에는 `(user_id,item_id,rating,distilled,genre,permuted,global_mean)`이 있습니다. 가중치 학습 시 Item은 `distilled`를 사용하고 source Content의 동일 Validation 쌍과 정답을 검증합니다. 서로 다른 split 또는 기존 ItemBasedCF 예측으로 혼용하지 않습니다.

## 가중치·Content·추천 프로토콜

- 새 Item은 이전 ItemBasedCF와 다른 모델입니다. 과거 Hybrid alpha=0.85나 0.90을 자동 재사용하지 않습니다. 결합 가중치는 동일 영화 Validation에서만 새로 확정하고 도서 결과와 무관하게 고정합니다.
- Book Test로 alpha를 학습하거나 결과를 보고 재튜닝하면 안 됩니다. Book Train/Validation로 alpha를 진단·보정하는 것도 이번 strict 프로토콜에는 포함되지 않습니다.
- Content 쪽이 도서 support 평점으로 평균·프로필·변환을 계산하면 Item 성분만 strict일 뿐 **전체 Hybrid는 strict 조건을 충족하지 않습니다**. Content도 도서 평점을 읽지 않는 frozen 경로가 필요합니다.
- 도서 사용자와 영화 사용자가 비연결이므로 source의 개인화 능력과 target의 비개인화 능력 차이도 문서화해야 합니다. 영화 Validation이 실제 target 추론 조건과 맞는지도 확인해야 합니다.
- `recommend()`에서 Book Train/Test 평점으로 사용자별 seen 필터를 만들면 안 됩니다. 영화 사용자의 seen을 같은 숫자 도서 사용자에게 가져오는 것도 금지입니다. 현재 입력만으로 개인별 추천/미평가 목록은 식별할 수 없습니다.

## 세 방법의 차이와 실제 결과

| 방법 | 학습·추론 구조 | 도서 사용자 개인화 |
|---|---|---|
| 기존 ItemBasedCF | 원점수 사용자×영화 cosine 이웃과 해당 사용자의 영화 평점으로 가중평균 | 비연결 도서 사용자·책 카탈로그에 그대로 적용 불가. Book Train 재학습은 별도 적응 실험 |
| raw genre KNN | 장르 cosine으로 영화 anchor를 선택하고 영화 Train 평균 평점을 집계 | 없음. 협업 latent/metric을 학습하지 않는 비개인화 대조군 |
| 새 ItemBasedTransfer | 영화 raw item-user cosine의 저차원 협업 표현을 장르에서 Ridge 증류하고, 실제 협업 anchor 공간 이웃으로 영화 Train 평균 평점을 집계 | 없음. CF metric distillation이며 기존 CF의 무수정 직접 전이는 아님 |

선택 설정은 rank=8, Ridge alpha=100, K=10입니다. 기존 Goodbooks Test 597,648행에서:

| 모델 | MAE | RMSE |
|---|---:|---:|
| 영화 Train global mean | **0.905561** | **1.049362** |
| raw genre KNN | 1.036270 | 1.216714 |
| CF metric distillation | 1.005357 | 1.177881 |

새 모델은 장르 KNN보다 좋았지만 **global baseline보다 성능이 낮았습니다**. 영화 Validation에서도 global baseline보다 나빴습니다. Teacher permutation으로 예측이 변하는 것은 협업 신호가 작동한다는 증거이며 성능 우수성/개인화/도메인 일반화의 증거는 아닙니다. 이 결과로 모델·설정·가중치를 다시 선택하지 않았습니다.

상세 결과는 `RESULTS.md`, 전체 결과 폴더 `results/item_based/strict_transfer_20261106_0007_full/` 참조. 기존 코드 수정·새 Hybrid 구현·추가 도서 평가가 필요하면 별도 작업으로 진행해야 합니다.
