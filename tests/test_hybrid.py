"""하이브리드 추천 검증 틀. 실제 구현 후 ItemBasedTests와 같은 방식으로 채운다."""

import unittest

from src.hybrid import HybridCF


class HybridTests(unittest.TestCase):
    """같은 분할, 점수 범위, 추천 제외, 지표를 검증한다."""

    @unittest.skip('뼈대: 미구현')
    def test_recommendations(self):
        """이미 본 아이템 제외 및 정렬, 반환 길이를 검증한다."""
        raise NotImplementedError

    @unittest.skip('뼈대: 미구현')
    def test_predict_range(self):
        """predict가 학습 평점 범위(MovieLens면 1~5)의 float을 반환하는지 검증한다."""
        raise NotImplementedError

    @unittest.skip('뼈대: 미구현')
    def test_evaluation_metrics(self):
        """holdout의 행별 오차로 MAE/RMSE 및 전체 평가 수를 대조한다."""
        raise NotImplementedError


if __name__ == '__main__':
    unittest.main()
