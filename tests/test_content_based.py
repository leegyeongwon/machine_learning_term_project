"""작은 가상 데이터로 콘텐츠 기반 필터링을 검증한다."""

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.content_based import ContentBasedCF


def _write_features(directory):
    path = Path(directory) / "movies_multihot_data.npz"
    # 아이템 1: Action만, 아이템 2: Comedy만, 아이템 3: Action+Comedy
    raw = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    np.savez_compressed(
        path,
        multi_hot_norm=raw / norms,
        classes=np.array(["Action", "Comedy"], dtype=object),
        movie_ids=np.array([1, 2, 3]),
        titles=np.array(["A", "C", "AC"], dtype=object),
    )
    return path


class ContentBasedTests(unittest.TestCase):
    """누수, 추천 제외, 점수 범위를 검증한다."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.features = _write_features(self.tmpdir.name)
        self.train = pd.DataFrame({
            "user_id": [1, 1],
            "item_id": [1, 2],
            "rating": [5.0, 1.0],
        })
        self.model = ContentBasedCF(item_features_path=self.features).fit(self.train)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_predict_range(self):
        """predict가 학습 평점 범위의 float을 반환하는지 검증한다."""
        score = self.model.predict(1, 3)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 1.0)
        self.assertLessEqual(score, 5.0)

    def test_recommendations(self):
        """이미 본 아이템 제외 및 정렬, 반환 길이를 검증한다."""
        rec = self.model.recommend(1, top_n=10)
        self.assertFalse(set(rec.item_id) & {1, 2})
        self.assertEqual(list(rec.item_id), [3])
        self.assertTrue(rec.predicted_rating.is_monotonic_decreasing)

    def test_unseen_user_fallback(self):
        """학습에 없는 사용자는 평균 대체 예측을 쓴다."""
        score, method, contributors = self.model.predict_with_details(-1, 1)
        self.assertEqual(method, "item_mean")
        self.assertEqual(contributors, 0)
        self.assertTrue(1.0 <= score <= 5.0)

    def test_known_user_uses_content(self):
        """학습에 있는 사용자는 Ridge 콘텐츠 예측을 쓴다."""
        score, method, contributors = self.model.predict_with_details(1, 3)
        self.assertEqual(method, "content")
        self.assertGreaterEqual(contributors, 1)
        self.assertTrue(1.0 <= score <= 5.0)

    def test_prediction_uses_user_mean(self):
        """후한 유저의 예측이 전역 3점에 붙지 않는지 검증한다."""
        train = pd.DataFrame({
            "user_id": [7, 7],
            "item_id": [1, 2],
            "rating": [5.0, 4.0],
        })
        model = ContentBasedCF(item_features_path=self.features).fit(train)
        score = model.predict(7, 3)
        self.assertGreater(score, 3.2)


if __name__ == "__main__":
    unittest.main()
