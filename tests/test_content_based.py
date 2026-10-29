"""작은 가상 데이터로 콘텐츠 기반 필터링을 검증한다."""

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

from src.content_based import ContentBasedCF, split_by_user_three


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


def _write_book_features(directory):
    """테스트용 book_content.npz. 팀원 파일과 같은 CSR 키를 쓴다."""
    path = Path(directory) / "book_content.npz"
    genres = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    authors = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    plot = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    numeric = np.array([[0.2, 0.1], [0.8, 0.9], [0.5, 0.5]], dtype=np.float32)
    np.savez_compressed(
        path,
        book_ids=np.array([1, 2, 3]),
        titles=np.array(["A", "C", "AC"], dtype=object),
        genre_names=np.array(["Fantasy", "Romance"], dtype=object),
        numeric_features=numeric,
        plot_embeddings=plot,
        genres_data=genres.data,
        genres_indices=genres.indices,
        genres_indptr=genres.indptr,
        genres_shape=np.array(genres.shape),
        authors_data=authors.data,
        authors_indices=authors.indices,
        authors_indptr=authors.indptr,
        authors_shape=np.array(authors.shape),
    )
    return path


class BookContentTests(unittest.TestCase):
    """book_content.npz 로더와 train-only Ridge를 검증한다."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.features = _write_book_features(self.tmpdir.name)
        self.train = pd.DataFrame({
            "user_id": [1, 1],
            "item_id": [1, 2],
            "rating": [5.0, 1.0],
        })
        self.model = ContentBasedCF(
            item_features_path=self.features,
            movies_path=None,
            users_path=None,
        ).fit(self.train)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_book_npz_fit_and_predict(self):
        """책 특성으로 fit한 뒤 미평가 책에 콘텐츠 예측을 한다."""
        self.assertIsNone(self.model.users_path)
        self.assertFalse(self.model.has_demo_)
        self.assertIsNotNone(self.model.item_plot_)
        self.assertIsNotNone(self.model.author_csr_)
        score, method, _ = self.model.predict_with_details(1, 3)
        self.assertEqual(method, "content")
        self.assertTrue(1.0 <= score <= 5.0)

    def test_book_year_comes_from_numeric(self):
        """출판연도는 제목이 아니라 numeric_features에서 온다."""
        np.testing.assert_allclose(self.model.year_norm_, [0.1, 0.9, 0.5])


class SplitTests(unittest.TestCase):
    """사용자별 train/val/test 분할을 검증한다."""

    def test_three_way_keeps_one_train_and_holds_out_val_test(self):
        """유저마다 학습 최소 1개, val·test는 겹치지 않는다."""
        ratings = pd.DataFrame({
            "user_id": [1] * 10 + [2] * 5 + [3],
            "item_id": list(range(16)),
            "rating": [4.0] * 16,
        })
        train, val, test = split_by_user_three(ratings, val_size=0.2, test_size=0.2, random_state=42)
        self.assertEqual(len(train) + len(val) + len(test), 16)
        self.assertEqual(set(train.index) & set(val.index), set())
        self.assertEqual(set(train.index) & set(test.index), set())
        self.assertEqual(set(val.index) & set(test.index), set())
        for user_id in (1, 2, 3):
            n_train = int((train.user_id == user_id).sum())
            self.assertGreaterEqual(n_train, 1)
        self.assertEqual(int((test.user_id == 3).sum()), 0)


if __name__ == "__main__":
    unittest.main()
