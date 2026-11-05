# 추천시스템 실행 및 평가

## 실행

프로젝트 루트에서 기존 가상환경 사용.

```powershell
.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.venv\Scripts\python.exe -B scripts/run_full_recsys_pipeline.py --dataset 32m --validate-only
.venv\Scripts\python.exe -B scripts/run_full_recsys_pipeline.py --dataset 32m --smoke
.venv\Scripts\python.exe -B scripts/run_full_recsys_pipeline.py --dataset 32m
.venv\Scripts\python.exe -B scripts/verify_recsys_run.py --run-dir <전체실행결과폴더>
```

기본 출력은 실행 시각별 새 폴더. `--output-dir`이 이미 존재하면 중단함. 1M은 `--dataset 1m`으로 별도 실행.

## 데이터 경로

| 역할 | 파일 |
|---|---|
| 기존 1M | `data/processed/item_user_data.npz` |
| 새 영화 CF | `data/processed/movie_item_user_data.npz` |
| 새 영화 콘텐츠 | `data/processed/movie_content.npz` |
| 새 영화 공통 분할·저장 프로필 | `data/processed/movie_user_profiles.npz` |
| 도서 CF | `data/processed/book_item_user_data.npz` |
| 도서 콘텐츠 | `data/processed/book_content.npz` |
| 도서 공통 분할·저장 프로필 | `data/processed/book_user_profiles.npz` |

NPZ가 있으면 원본 CSV는 필수가 아님. 전처리를 다시 실행하거나 원본을 변경하지 않음. 원본 CSV가 없으면 실제 원본 32M 총수와 외부 `links.csv` 매핑의 정확성은 독립 검증하지 못함. NPZ 내부 일치 검증과 구분해야 함.

문자열 object 배열인 제목·작가는 로딩하지 않음. 필요한 수치·유니코드 배열만 `allow_pickle=False`로 읽음. 입력 전체 SHA-256과 배열 헤더를 저장하고 실행 후 동일성을 재확인함.

## 검증

두 모델 NPZ의 모든 분할 쌍·평점·순서 비교. 영화 CF의 float32와 콘텐츠 프로필의 float64는 반점 단위 평점 값이 정확히 같은지 검사함. 사용자·아이템 쌍의 분할 내·분할 간 중복, 8:1:1 비율, 범위, 결측, cold-start 확인.

영화는 NPZ 전체 쌍을 원문의 사용자·영화 순으로 정렬해 seed 42 분할을 재현함. 도서는 원본 행 순서가 없어서 seed의 독립 재현은 확인하지 못하며, 두 모델의 분할 일치와 저장 seed를 확인함.

Train만으로 평균·모집단 표준편차·마스크·Z-score를 재계산. Z-score 0과 미평가 0은 마스크로 구분함. 저장 장르·수치 프로필은 전체 사용자에서 재계산하며, 1,024차원 대형 줄거리 프로필은 64개 사용자 표본을 스트리밍으로 검사함. 수치 프로필의 float32 누적·연도 재정규화 반올림을 위해 절대·상대 허용오차 1e-5와 실제 최대 오차를 기록함.

검증한 저장 프로필을 예측에 넣지는 않음. 모델에 필요한 잔차 프로필은 공통 Train 또는 Book Train support에서 다시 계산함.

## 모델과 공통 입력

Item-based는 기존 원점수 cosine KNN 유지. 최대 K=50의 이웃 순서를 한 번 계산하고 앞 K개를 사용해 20, 30, 50을 비교함. CSR 행렬과 배치 예측을 사용함.

Content-based는 기존 `유저 평균 + Ridge 잔차 보정`과 콘텐츠·취향 상호작용 구조를 재사용함. 평점별 콘텐츠 벡터 전체를 펼치는 대신 `희소 잔차 평점 행렬 × 아이템 콘텐츠`로 같은 사용자 프로필을 계산함. 세 Ridge 규제 후보는 동일한 충분통계 `XᵀX`, `Xᵀy`를 재사용함.

영화에서 실제 관측된 20개 장르를 기준으로 대소문자·공백·하이픈을 정규화함. `Science Fiction`과 `science-fiction`은 같은 열로 연결되지만 `Drama`와 `fiction`, `Comedy`와 `humor-and-comedy`는 임의로 연결하지 않음. 의미의 완전한 일치를 보장하지 않음.

공통 입력 순서: 장르 TF-IDF 20개, 영화 기준 연도 z-score 1개, 장르 취향×장르 20개, 줄거리 SVD 32개, 줄거리 취향 cosine 1개. 총 74차원. 영화와 도서에 동일한 이름·순서 사용.

기존 BGE-M3 1,024차원 임베딩은 재사용함. 전처리 원문은 양쪽 모두 `BAAI/bge-m3`를 지정하나, NPZ 자체에 모델 해시가 없어 생성 출처의 암호학적 증명까지는 제공하지 못함. 영화 Train에 등장한 아이템만으로 seed 42의 SVD32를 맞추고 도서는 같은 변환을 고정 적용함. 이는 메모리·연산량 절감을 위한 명시적인 콘텐츠 어댑터 변경이며 기존 원시 1,024차원 Ridge와 동일 실험이라고 주장하지 않음.

평점·작가·페이지·상영시간은 공통 콘텐츠 입력에서 제외함. 연도는 NPZ min/max로 원래 값에 가깝게 복원한 후 영화 기준으로 변환함. 도서 미래 연도는 원본 수정·행 삭제 없이 유지하고 이상값 목록을 보고함. 연도 결측은 영화 기준 평균으로 대체함.

## 선택과 평가

영화 Validation에서 K와 Ridge alpha(1, 10, 100)를 MAE, RMSE, 작은 후보 순으로 선택함. Weighted Hybrid는 `a × Item + (1-a) × Content`, a=0~1을 0.01 간격으로 탐색함. 선택 설정을 저장한 뒤 Test 평가. Test 결과에 따른 재튜닝·Train+Validation 재학습 없음.

도서는 별도 카탈로그·별도 사용자 프로필 객체를 사용하므로 숫자 ID가 겹쳐도 영화 사용자 이력으로 조회하지 않음. Book Train만 support로 사용하며 Book Test는 ID 쌍만 모델에 전달함.

- Content: 영화 Ridge·IDF·연도 변환·SVD·입력 순서를 고정하고 도서 support로 사용자 평균과 취향만 계산.
- Item: 영화에서 선택한 K와 알고리즘을 고정하고 도서 support로 책 유사도를 새로 계산.
- Hybrid: 영화에서 선택한 가중치를 고정.

이 평가는 support를 사용한 교차 도메인 평가이며 Item은 명시적인 도서 적응임. 영화 아이템 유사도 행렬의 책 카탈로그 직접 전이는 정의되지 않으므로 세 모델 모두의 엄격한 zero-shot 평가로 해석할 수 없음. Book Validation은 진단만 수행하며 설정을 변경하지 않음.

MovieLens는 0.5~5, Goodbooks는 1~5의 실제 점수로 평가함. 도서 예측은 알려진 1~5 범위로 제한함. 평점 분포·지원 밀도·척도 하한이 다르므로 오차 차이를 순수한 전이 효과로 해석할 수 없음.

## 결과물

`input_inventory.json`, 영화·도서 데이터 감사 JSON, Validation 예측·후보 지표·Hybrid grid, 설정 JSON, 콘텐츠 계수·SVD 변환 NPZ, 세 모델 Test 예측·지표·cold-start/fallback 진단, 비교 CSV/그래프, 단계별 시간·메모리, 입력 보존 확인을 저장함.

`verify_recsys_run.py`는 최종 CSV를 NPZ Test 쌍·정답과 대조하고, 지표·고정 가중치 결합·Validation 선택 및 입력 SHA-256을 독립 재확인함.
