"""임의 평점을 생성하지 않고 실제 MovieLens 일부 및 수학적 불변식을 검증한다."""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix_csr

from src.item_based import ItemBasedCF, split_by_user
from scripts.run_item_based import load_movielens, evaluate

ROOT = Path(__file__).resolve().parents[1]


class ItemBasedTests(unittest.TestCase):
    """실제 데이터의 앞 12명으로 작은 검증을 먼저 수행한다."""

    @classmethod
    def setUpClass(cls):
        """원본의 실제 평점만 선택한다."""
        ratings, cls.movies = load_movielens(ROOT / 'data/raw/ml-1m')
        cls.ratings = ratings[ratings.user_id <= 12].copy()
        cls.train, cls.test = split_by_user(cls.ratings)
        cls.model = ItemBasedCF(30).fit(cls.train)

    def test_split_no_leakage(self):
        """분할 겹침, 평점 유실, 사용자 학습 누락을 검사한다."""
        keys = ['user_id', 'item_id']
        self.assertTrue(self.train.merge(self.test, on=keys).empty)
        self.assertEqual(len(self.train) + len(self.test), len(self.ratings))
        self.assertEqual(set(self.ratings.user_id), set(self.train.user_id))
        a, b = split_by_user(self.ratings)
        pd.testing.assert_frame_equal(a, self.train)
        pd.testing.assert_frame_equal(b, self.test)
        for row in self.test.itertuples():
            u = self.model.user_ids_.get_loc(row.user_id)
            i = self.model.item_ids_.get_indexer([row.item_id])[0]
            if i >= 0:
                self.assertEqual(self.model.user_item_[u, i], 0)

    def test_cosine_and_weighted_average(self):
        """별도 NumPy 내적과 직접 가중합으로 실제 모델의 수학을 검증한다."""
        model = self.model
        self.assertTrue(isspmatrix_csr(model.user_item_))
        found = False
        for i in range(min(100, len(model.item_ids_))):
            self.assertNotIn(i, model.neighbor_indices_[i])
            a = model.item_user_[i].toarray().ravel()
            for j, sim in zip(model.neighbor_indices_[i], model.neighbor_similarities_[i]):
                b = model.item_user_[j].toarray().ravel()
                self.assertAlmostEqual(sim, np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            for u, user in enumerate(model.user_ids_):
                values = model.user_item_[u, model.neighbor_indices_[i]].toarray().ravel()
                weights = model.neighbor_similarities_[i]
                valid = (values > 0) & (weights > 1e-12)
                if valid.any():
                    expected = sum(values[valid] * weights[valid]) / sum(weights[valid])
                    self.assertAlmostEqual(model.predict(user, model.item_ids_[i]), expected)
                    found = True
        self.assertTrue(found)

    def test_recommendations(self):
        """이미 본 영화 제외 및 정렬, 반환 길이를 검증한다."""
        rec = self.model.recommend(1)
        seen = set(self.train.loc[self.train.user_id == 1, 'item_id'])
        self.assertFalse(set(rec.item_id) & seen)
        self.assertEqual(len(rec), 10)
        self.assertTrue(rec.predicted_rating.is_monotonic_decreasing)
        self.assertTrue(rec.item_id.isin(self.movies.item_id).all())

    def test_fallback_and_single_item(self):
        """미지 사용자/아이템, 단일 아이템, 0 유사도의 대체 예측 검증."""
        model = self.model
        self.assertEqual(model.predict_with_details(-1, -1)[1], 'global_mean')
        self.assertEqual(model.predict_with_details(1, -1)[1], 'user_mean')
        self.assertEqual(model.predict_with_details(-1, model.item_ids_[0])[1], 'item_mean')
        one = self.train.iloc[:1]
        tiny = ItemBasedCF(50).fit(one)
        self.assertEqual(tiny.neighbor_indices_.shape, (1, 0))
        self.assertTrue(tiny.recommend(one.iloc[0].user_id).empty)
        a, b = split_by_user(one)
        self.assertEqual(len(a), 1)
        self.assertTrue(b.empty)
        # 서로 다른 사용자/아이템의 실제 두 평점: 코사인 내적이 0.
        first = self.train.iloc[0]
        second = self.train[(self.train.user_id != first.user_id) &
                            (self.train.item_id != first.item_id)].iloc[:1]
        zero = ItemBasedCF().fit(pd.concat([self.train.iloc[:1], second]))
        self.assertEqual(zero.predict_with_details(first.user_id, second.iloc[0].item_id)[1], 'user_mean')

    def test_invalid_input(self):
        """잘못된 데이터와 미학습 호출을 조용히 수용하지 않는다."""
        with self.assertRaises(ValueError):
            ItemBasedCF(0)
        with self.assertRaises(ValueError):
            ItemBasedCF().predict(1, 1)
        for frame in [self.train.iloc[:0], pd.concat([self.train.iloc[:1]] * 2),
                      self.train.iloc[:1].assign(rating=0),
                      self.train.iloc[:1].assign(rating=np.nan)]:
            with self.assertRaises(ValueError):
                ItemBasedCF().fit(frame)

    def test_evaluation_metrics(self):
        """실제 holdout의 행별 오차로 MAE/RMSE 및 전체 평가 수를 대조한다."""
        metrics, predictions = evaluate(self.model, self.test)
        errors = predictions.predicted_rating.to_numpy() - self.test.rating.to_numpy()
        self.assertAlmostEqual(metrics['mae'], np.abs(errors).mean())
        self.assertAlmostEqual(metrics['rmse'], np.sqrt(np.mean(errors ** 2)))
        self.assertEqual(metrics['n_evaluated'], len(self.test))
        self.assertEqual(metrics['n_cf'] + metrics['n_fallback'], len(self.test))
        self.assertTrue(np.isfinite(predictions.predicted_rating).all())


if __name__ == '__main__':
    unittest.main()
