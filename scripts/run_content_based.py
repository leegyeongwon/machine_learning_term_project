"""MovieLens 1M 로딩, 학습/평가, 추천 및 결과 저장용 CLI (콘텐츠 기반)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def evaluate(model, test):
    """전체 테스트를 평가하고 행별 근거를 남긴다.

    입력:
        model: fit이 끝난 ContentBasedCF.
        test: user_id, item_id, rating 컬럼의 DataFrame.
    출력:
        (metrics: dict, predictions: DataFrame)
        predictions 컬럼: user_id, item_id, rating, predicted_rating, method, n_contributors
    하는 일:
        테스트 행을 버리지 않고 MAE, RMSE를 계산한다.
    """
    raise NotImplementedError


def main():
    """콘텐츠 기반 모델을 학습·평가하고 results/content_based/에 저장한다.

    기본 경로:
        --data-dir data/raw/ml-1m
        --output-dir results/content_based
    ItemBasedCF 실행 스크립트와 같은 분할(seed=42, test_size=0.2)을 쓴다.
    """
    raise NotImplementedError


if __name__ == '__main__':
    main()
