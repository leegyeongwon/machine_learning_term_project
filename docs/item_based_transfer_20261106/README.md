# 협업 정보를 활용한 Item-based 교차 도메인 전이 모델

## 구현 목적과 범위

본 모델은 영화 평점으로 학습한 협업 정보를 도서 평점 예측에 활용하는 모델이다. 영화 데이터만으로 학습과 설정 선택을 마친 뒤 도서의 장르를 입력하여 평점을 예측한다. 도서 평점은 최종 MAE와 RMSE를 계산할 때 정답으로만 사용한다.

기존 Item-based CF는 사용자들의 평점 패턴으로 아이템 간 유사도를 구하고, 예측 대상 사용자가 이웃 아이템에 준 평점을 가중평균한다. 그러나 영화와 도서는 사용자와 아이템이 서로 연결되어 있지 않으므로 영화에서 구한 이웃 관계를 책에 그대로 적용할 수 없다. 새 모델은 영화의 협업 표현을 장르로 추정하는 방식을 사용한다. 협업 표현이란 여러 사용자의 평점 패턴을 요약한 아이템별 벡터이다.

새 모델은 해당 사용자의 이웃 평점 대신 이웃 영화들의 평균 평점을 이용한다. 따라서 같은 책에는 모든 사용자에게 같은 점수를 반환한다. 전통적인 개인화 Item-based CF와 구분하여 협업 정보를 활용한 비개인화 콘텐츠 전이 모델로 설명한다.

기존 구현은 보존하고 다음 전용 파일을 사용한다.

| 파일 | 역할 |
|---|---|
| `src/item_based_transfer.py` | 모델 학습과 예측을 담당한다. |
| `src/item_based_transfer_data.py` | 데이터 접근 범위와 원본 보존 검사를 담당한다. |
| `scripts/run_item_based_strict_transfer.py` | 학습, 설정 선택, 최종 평가를 수행한다. |
| `scripts/verify_item_based_strict_transfer.py` | 저장된 모델과 예측 결과를 재검증한다. |
| `tests/test_item_based_strict_transfer.py` | 전용 단위 테스트를 수행한다. |

기존 실행기와 보고서 생성기에는 Git 명령을 호출하는 부분이 있으므로 이 실험에서는 사용하지 않는다. Content-based, Hybrid, 전처리 코드와 공유 NPZ 원본도 수정하지 않는다.

## 학습 및 평가 조건

| 구분 | 사용 데이터와 조건 |
|---|---|
| Train | 영화 평점 17,401,383행 전체로 협업 표현과 Ridge를 학습한다. 장르 사전도 영화 Train에 등장한 아이템만으로 구성한다. |
| Validation | 영화 평점 2,175,173행 전체로 협업 표현의 차원 수(rank), Ridge 규제 계수(alpha), 이웃 수(K)를 선택한다. |
| Test | 기존 도서 Test 평점 597,648행을 최종 정답으로만 사용한다. |
| 사용하지 않는 데이터 | 영화 Test, 도서 Train·Validation 평점, 미리 계산된 사용자 프로필을 사용하지 않는다. |

입력 특징은 장르의 포함 여부를 나타내는 이진 벡터이다. 영화 Train 장르 사전에 맞춰 열을 정렬하고, 아이템별 벡터 길이를 1로 맞추는 L2 정규화를 적용한다. IDF, 영화와 도서를 함께 정규화한 연도, 생성 조건이 충분히 확인되지 않은 줄거리 임베딩은 사용하지 않는다.

영화 학습부터 도서 평가까지의 순서는 다음과 같다.

1. 영화 Train으로 장르 사전, 협업 표현, Ridge를 학습한다.
2. 영화 Validation으로 rank, alpha, K를 선택한다.
3. 영화 모델과 설정을 저장하고 SHA-256 해시를 기록하여 이후 변경 여부를 확인한다.
4. 도서 콘텐츠를 읽고 고정된 영화 장르 사전과 모델을 적용한다. 도서 데이터로 변환 기준을 다시 학습하지 않는다.
5. 도서 Test의 사용자·아이템 ID에 맞춰 예측 배열, CSV, 책별 점수를 먼저 저장하고 해시를 기록한다.
6. 저장된 파일이 변경되지 않았는지 확인한 뒤 도서 Test 정답을 읽어 MAE와 RMSE를 계산한다.

도서 평가 결과를 보고 모델이나 설정을 다시 선택하지 않는다. 사용자 ID는 평가 행을 맞추는 데 사용하며 사용자별 선호를 계산하는 데 사용하지 않는다.

## 모델의 학습과 예측 방식

### 영화 평점에서 협업 표현을 만드는 과정

영화 Train의 사용자×아이템 평점 행렬을 `R_movie_train`이라 할 때 이를 전치하고 영화별로 정규화한 행렬을 다음과 같이 정의한다.

```text
A = row_normalize(R_movie_train.T)
```

`A`의 행은 영화, 열은 사용자에 해당한다. `A A^T`는 기존 원점수 기반 Item-based CF와 같은 정의의 코사인 유사도이다. 즉, 장르가 아니라 사용자들의 공동 평가와 평점 패턴을 비교한다.

이 행렬에 최대 rank=32의 SVD를 한 번 학습하고 앞부분의 8/16/32개 차원을 각각 후보로 사용한다. 차원 수가 r일 때 협업 표현은 다음과 같다.

```text
Q_r = row_normalize(A V_r)
```

`V_r`는 SVD에서 얻은 변환 행렬이다. `Q_r`는 사용자 평점 패턴을 낮은 차원으로 요약한 영화별 표현이며 원래 협업 유사도를 그대로 보존하는 것은 아니다.

### 장르로 협업 표현을 추정하는 과정

영화 장르 행렬을 `X`라 할 때 Ridge는 `X`로부터 협업 표현 `Q_r`를 예측하도록 학습한다.

```text
B,c = argmin ||X B+c-Q_r||^2 + alpha ||B||^2
```

`B`는 회귀 계수, `c`는 절편이다. Ridge의 학습 정답은 평점 자체가 아니라 영화 평점 행렬에서 얻은 협업 벡터이다. 장르를 직접 비교하는 콘텐츠 이웃 모델과 구분되는 부분이다.

### 도서 평점을 계산하는 과정

도서 b의 장르 벡터를 `x_b`라 할 때 영화에서 학습한 회귀 계수를 그대로 적용한다.

```text
z_b = normalize(x_b B+c)
```

추정한 도서 표현 `z_b`와 실제 영화 협업 표현 `Q_r[j]`의 코사인 유사도를 계산한다. 양수인 유사도 중 상위 K개 영화만 이웃으로 선택하고 그 영화들의 Train 평균 평점을 가중평균한다.

```text
score(b) = sum_j w_bj mean_train_movie(j) / sum_j w_bj
```

`w_bj`는 선택한 영화와의 유사도이다. 영화 장르 사전에 연결되는 장르가 없거나 가중치 합이 0이면 영화 Train 전체 평균을 대체 예측값으로 사용한다. 도서 평점으로 사용자 프로필이나 책 간 유사도를 만들지는 않는다.

## 비교 모델과 협업 정보 확인 방법

장르 KNN 기준선인 `raw_genre_cosine`은 `w_bj = cosine(x_b, x_j)`로 이웃 영화를 선택한다. 협업 표현은 학습하지 않지만 최종 점수에는 동일하게 영화 Train 평균 평점을 사용한다.

협업 정보의 역할을 확인하기 위해 영화 장르와 협업 표현의 학습 대응만 무작위로 섞는 실험(teacher permutation)을 수행한다. 이때 실제 이웃 영화의 협업 표현과 평균 평점은 유지한다. 이 실험은 영화 Validation에서만 수행하며 올바른 학습 대응이 예측에 영향을 주는지 확인하는 데 사용한다.

## 모델의 한계

도서 사용자의 선호 정보와 영화 사용자와의 연결 정보가 없으므로 개인화 예측은 할 수 없다. 같은 장르 조합을 가진 아이템도 선형 Ridge에서 구별하지 못한다. 또한 장르만으로 영화의 협업 구조를 충분히 복원하지 못할 수 있다. 따라서 영화 Validation 결과가 좋더라도 영화에서 도서로의 전이 성능까지 보장하지는 않는다.

협업 정보의 역할은 학습 대응을 섞은 결과, 협업 유사도 복원 오차, 장르 KNN과의 차이를 함께 살펴 판단한다. 협업 정보가 예측에 사용된다는 사실과 정확도가 높다는 주장은 구분한다.

영화와 도서를 합쳐 만든 기존 장르 사전에서도 영화 Train에 없는 장르 열은 학습에서 제외한다. 공동 정규화된 연도와 생성 조건이 불명확한 임베딩도 읽지 않는다. 다만 기존 NPZ만으로는 앞선 TMDB 매칭, 데이터 선정, 도서 메타데이터 수집 과정을 되돌리거나 모두 확인할 수 없다. 도서 Test가 과거의 다른 실험에서도 사용된 평가셋이라는 한계도 남아 있다.

## 모델 사용 방법과 Hybrid 연동

다음은 모델 사용 예시이다. 예시의 설정값은 최종 실험에서 선택된 값과 별개이며 실제 선택 결과는 `RESULTS.md`와 저장된 설정 파일에 기록한다.

```python
model = ItemBasedTransfer(rank=16, ridge_alpha=10, n_neighbors=30)
model.fit(movie_train, source_content=movie_catalog)
model.bind_target_content(book_catalog, namespace="book")
model.predict(user_id, book_id)
model.predict_many(user_ids, book_ids)
model.recommend(user_id, top_n=10)
model.save(new_path)
restored = ItemBasedTransfer.load(new_path)
```

`bind_target_content`는 영화에서 학습한 변환을 도서에 적용하고 예측값을 저장한다. 이 과정에서 모델을 다시 학습하지 않는다. 새 모델은 기존 Hybrid의 `fit(train_ratings)` 호출 방식과 바로 호환되지 않으므로 별도의 연결 처리가 필요하다. 이 구현에서는 기존 Hybrid 코드를 수정하지 않는다.

## 실행 방법과 저장 파일

```powershell
.venv\Scripts\python.exe -B scripts/run_item_based_strict_transfer.py --smoke
.venv\Scripts\python.exe -B scripts/run_item_based_strict_transfer.py
.venv\Scripts\python.exe -B scripts/verify_item_based_strict_transfer.py --run-dir <새결과폴더>
```

기본 출력 경로는 `results/item_based/strict_transfer_<KST 새시각>/`이다. 이미 존재하는 출력 경로는 사용하지 않아 이전 결과를 덮어쓰지 않는다.

저장 항목은 실행 로그, 사전에 정한 설정 후보, 영화 모델과 선택 설정, 영화 Validation 결과, 학습 대응을 섞은 비교 결과, 영화 아이템 일부를 제외한 진단 결과, 도서 예측 파일과 해시, 데이터 접근 기록, 최종 지표, 원본 보존 검사 결과 및 보고서이다. 복구 경로와 방법은 `RESTORE.md`에 정리한다.

## 참고 논문

- [van den Oord et al., Deep content-based music recommendation, 2013](https://proceedings.neurips.cc/paper/2013/file/b3ba8f1bee1238a2f37603d90b58898d-Paper.pdf)의 협업 잠재요인을 콘텐츠에서 추정하는 아이디어를 참고하였다. 해당 논문의 오디오 CNN을 재현한 것은 아니다.
- [Zhou et al., Contrastive Collaborative Filtering for Cold-Start Item Recommendation, 2023](https://arxiv.org/abs/2302.02151)에서 다룬 신규 아이템의 콘텐츠 표현과 협업 표현 사이의 불일치 문제를 참고하였다. 본 Ridge 구현은 CCFCRec이나 대조학습 모델을 재현한 것은 아니다.
