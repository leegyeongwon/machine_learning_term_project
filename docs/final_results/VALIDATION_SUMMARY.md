# 실험 검증 요약

2026-11-07에 기존 Item-based CF, Content-based, Hybrid의 전체 실험을 다시 실행하고 결과를 검증하였다.

- 단위 테스트 27개가 모두 통과하였다.
- 데이터 분할 일치 여부와 입력 데이터 검증을 통과하였다.
- 소규모 전체 실행인 Smoke Test를 통과하였다.
- 과거 MAE·RMSE 결과 12개와 원본 정밀도에서 완전히 일치하였다.
- Test 예측 CSV 6개가 이전 파일과 동일하였다.
- 별도 독립 검증에서 지표, 선택 설정, Hybrid 가중합과 예측 쌍의 일치를 확인하였다.
- 기존 입력 데이터는 수정하지 않았다.

이 문서를 다듬는 과정에서는 학습, 평가, 튜닝을 다시 실행하지 않았다. 도서 결과는 Book Train을 활용한 Support-adapted 평가이다.

## 검증 근거

원본 실행 결과 폴더는 `../../results/hybrid/reproduction_20261107_000603_KST/`이다. 주요 파일의 경로는 이 폴더를 기준으로 한다.

- `unit_test_summary.json`: 단위 테스트 결과이다.
- `final_verification_summary.json`: 단계별 실행과 데이터 보존을 포함한 종합 확인 결과이다.
- `historical_reproduction_summary.json`: 과거 지표 및 예측 파일과의 비교 결과이다.
- `full/independent_verification.json`: 독립 검증 결과이다.
