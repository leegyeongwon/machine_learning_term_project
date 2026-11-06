# Item-based strict genre-only 전이

## 범위

기존 코드를 수정하지 않는 별도 연구 경로입니다. `src/item_based_transfer.py`, `src/item_based_transfer_data.py`, `scripts/run_item_based_strict_transfer.py`, `scripts/verify_item_based_strict_transfer.py`, `tests/test_item_based_strict_transfer.py`를 사용합니다. 기존 실행기·보고서 생성기에는 내부 Git 호출이 있으므로 사용하지 않습니다. Content-based/Hybrid/전처리/공유 NPZ 원본은 수정하지 않습니다.

## 프로토콜

- 영화 Train 17,401,383행과 Validation 2,175,173행 전체 사용. 영화 Test 배열 사용 금지.
- 영화 Train 아이템만으로 장르 사전·협업 표현·Ridge 학생 학습. Validation으로 rank/alpha/K만 선택.
- 기본 입력은 binary 장르의 source vocabulary 정렬 및 행 L2 정규화. IDF·공동 연도·출처 불확실한 plot embedding 제외.
- 도서 콘텐츠는 source 모델·설정의 저장 및 SHA-256 동결 후에만 로딩.
- 도서 Train/Validation 평점과 모든 사용자 프로필 사용 금지. 기존 Test ID만으로 예측 배열·CSV·책별 점수를 먼저 저장·봉인.
- prediction seal 검증 후에만 기존 Book Test 정답 597,648행을 읽어 최종 MAE/RMSE 계산.
- 도서 결과를 보고 재선택하지 않음. 모델과 예측은 사용자 ID 값을 무시함.

## 수식

`A = row_normalize(R_movie_train.T)`. `A A^T`는 기존 원점수 item cosine 유사도와 같은 정의입니다. SVD의 최대 rank=32를 한 번 fit하고 prefix rank 8/16/32를 후보로 사용합니다.

`Q_r = row_normalize(A V_r)`, `B,c = argmin ||X B+c-Q_r||^2 + alpha ||B||^2`.

도서 `z_b = normalize(x_b B+c)`. 실제 협업 표현 `Q_r[j]`와의 cosine에서 양수 Top-K만 사용합니다. `score(b)=sum_j w_bj mean_train_movie(j)/sum_j w_bj`. 연결 장르가 없거나 가중치 합이 0이면 영화 Train global mean fallback입니다.

raw genre cosine baseline은 `w_bj = cosine(x_b, x_j)`이므로 협업 표현을 학습하지 않습니다. Teacher permutation은 source 콘텐츠와 협업 표현 target의 대응만 섞고 실제 anchor 표현·평점은 유지합니다. 이는 영화 Validation에서만 수행합니다.

## 한계

서로 연결되지 않은 실제 도서 사용자의 선호 정보가 없으므로 개인화는 식별할 수 없습니다. 같은 책은 모든 사용자에게 같은 점수입니다. 이것은 영화 CF를 teacher로 쓰는 비개인화 metric distillation이며 기존 Item-based CF를 그대로 책에 적용한 것이 아닙니다.

영화 Validation이 좋아도 cross-domain 성능은 보장되지 않습니다. 선형 genre-only 학생은 같은 장르 조합을 구별하지 못하고 협업 구조를 충분히 복구하지 못할 수 있습니다. CF 기여는 permutation·표현 재구성 진단과 단순 콘텐츠 baseline 차이를 함께 봐야 합니다. 성능 우수성은 사전에 주장하지 않습니다.

공동 genre union의 도서 전용 열은 영화 Train vocabulary에서 배제합니다. 공동 연도 및 encoder provenance 미확인 임베딩은 읽지 않습니다. 다만 upstream의 TMDB 매칭·데이터 선정·도서 메타데이터 수집 과정 자체를 NPZ만으로 되돌릴 수는 없습니다. 과거의 다른 프로토콜에서 도서 Test가 노출된 반복 벤치마크라는 점도 보고합니다.

## API

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

Target bind는 frozen transformer를 적용하고 예측을 cache할 뿐 fit하지 않습니다. 원래 Hybrid의 `fit(train_ratings)` 경로와 자동 호환되지 않습니다. 변경이 필요하면 별도 승인·어댑터가 필요하며 기존 Hybrid 코드를 수정하지 않습니다.

## 실행

```powershell
.venv\Scripts\python.exe -B scripts/run_item_based_strict_transfer.py --smoke
.venv\Scripts\python.exe -B scripts/run_item_based_strict_transfer.py
.venv\Scripts\python.exe -B scripts/verify_item_based_strict_transfer.py --run-dir <새결과폴더>
```

기본 출력은 `results/item_based/strict_transfer_<KST 새시각>/`. 이미 존재하는 출력 경로는 거부합니다. 실행 로그, predeclared grid, source 모델·설정, source Validation 후보·permutation·item-holdout 진단, target prediction seal, 접근 기록, 최종 지표, 입력 보존 결과와 보고서를 저장합니다.

복구 경로와 방법은 `RESTORE.md` 참조.

## 논문

- [van den Oord et al., Deep content-based music recommendation, 2013](https://proceedings.neurips.cc/paper/2013/file/b3ba8f1bee1238a2f37603d90b58898d-Paper.pdf): 협업 latent factor를 콘텐츠에서 추정하는 발상 참고. 오디오 CNN 재현은 아님.
- [Zhou et al., Contrastive Collaborative Filtering for Cold-Start Item Recommendation, 2023](https://arxiv.org/abs/2302.02151): cold item의 콘텐츠-협업 표현 불일치 문제 참고. 본 Ridge 구현은 CCFCRec/contrastive 모델 재현이 아님.
