"""공통 8:1:1 Content 분할 및 Hybrid Validation 산출물 테스트."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.common_item_data import assert_identical_splits, load_common_splits
from src.content_based import ContentBasedCF
from scripts.run_content_based_common_split import (
    choose_best_alpha,
    validate_hybrid_predictions,
)

ROOT = Path(__file__).resolve().parents[1]
CONTENT_NPZ = ROOT / "data" / "processed" / "user_profiles_data.npz"
ITEM_NPZ = ROOT / "data" / "processed" / "item_user_data.npz"
OUT = ROOT / "results" / "content_based" / "common_split_811"


def _write_split_npz(path, train, validation, test):
    np.savez_compressed(
        path,
        train_user_ids=train.user_id.to_numpy(),
        train_movie_ids=train.item_id.to_numpy(),
        train_ratings=train.rating.to_numpy(),
        validation_user_ids=validation.user_id.to_numpy(),
        validation_movie_ids=validation.item_id.to_numpy(),
        validation_ratings=validation.rating.to_numpy(),
        test_user_ids=test.user_id.to_numpy(),
        test_movie_ids=test.item_id.to_numpy(),
        test_ratings=test.rating.to_numpy(),
        random_state=np.asarray(42),
    )


class CommonContentSplitUnitTests(unittest.TestCase):
    """작은 가상 데이터로 분할 비교와 선택 규칙을 검증한다."""

    def test_assert_identical_splits(self):
        """같은 triple은 통과하고 다른 triple은 실패한다."""
        train = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 20], "rating": [5.0, 4.0]})
        val = pd.DataFrame({"user_id": [1], "item_id": [20], "rating": [3.0]})
        test = pd.DataFrame({"user_id": [2], "item_id": [10], "rating": [2.0]})
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.npz"
            right = Path(tmp) / "right.npz"
            other = Path(tmp) / "other.npz"
            _write_split_npz(left, train, val, test)
            _write_split_npz(right, train.iloc[::-1], val, test)
            _write_split_npz(
                other,
                train,
                pd.DataFrame({"user_id": [1], "item_id": [99], "rating": [3.0]}),
                test,
            )
            parts = assert_identical_splits(left, right)
            self.assertEqual(len(parts["train"]), 2)
            with self.assertRaises(AssertionError):
                assert_identical_splits(left, other)

    def test_choose_best_alpha_prefers_mae_then_rmse_then_smaller(self):
        """MAE가 같으면 RMSE, 그것도 같으면 작은 alpha를 고른다."""
        metrics = pd.DataFrame({
            "ridge_alpha": [1.0, 10.0, 100.0],
            "mae": [0.80, 0.70, 0.70],
            "rmse": [1.10, 0.95, 0.95],
        })
        self.assertEqual(choose_best_alpha(metrics), 10.0)
        metrics.loc[metrics.ridge_alpha == 10.0, "rmse"] = 1.05
        self.assertEqual(choose_best_alpha(metrics), 100.0)

    def test_validate_hybrid_predictions_accepts_content_methods(self):
        """content/fallback method와 1:1 Validation 대응을 검사한다."""
        validation = pd.DataFrame({
            "user_id": [1, 2],
            "item_id": [10, 20],
            "rating": [4.0, 5.0],
        })
        predictions = validation.copy()
        predictions["predicted_rating"] = [3.5, 4.5]
        predictions["method"] = ["content", "user_mean"]
        predictions["n_contributors"] = [2, 0]
        checks = validate_hybrid_predictions(predictions, validation)
        self.assertEqual(checks["rows"], 2)
        bad = predictions.copy()
        bad.loc[1, "n_contributors"] = 3
        with self.assertRaises(ValueError):
            validate_hybrid_predictions(bad, validation)


@unittest.skipUnless(CONTENT_NPZ.exists() and ITEM_NPZ.exists(), "로컬 공통 NPZ가 필요합니다.")
class RealContentSplitTests(unittest.TestCase):
    """실제 MovieLens NPZ의 공통 분할 일치를 검증한다."""

    def test_content_and_item_npz_splits_match(self):
        """content/item NPZ의 train/validation/test triple이 같다."""
        parts = assert_identical_splits(CONTENT_NPZ, ITEM_NPZ)
        self.assertEqual(len(parts["train"]), 800167)
        self.assertEqual(len(parts["validation"]), 100021)
        self.assertEqual(len(parts["test"]), 100021)
        self.assertEqual(sum(len(frame) for frame in parts.values()), 1000209)


@unittest.skipUnless((OUT / "hybrid_ready_validation_predictions.csv").exists(), "공통 분할 실행 결과가 필요합니다.")
class ContentHybridOutputTests(unittest.TestCase):
    """저장된 Hybrid Validation CSV를 재검증한다."""

    def test_hybrid_ready_file_matches_validation(self):
        """저장 CSV가 Validation과 일대일이고 Test를 쓰지 않았는지 확인한다."""
        validation = load_common_splits(CONTENT_NPZ)["validation"]
        saved = pd.read_csv(OUT / "hybrid_ready_validation_predictions.csv")
        self.assertEqual(validate_hybrid_predictions(saved, validation)["rows"], 100021)
        contract = (OUT / "hybrid_contract.json").read_text(encoding="utf-8")
        self.assertIn('"test_evaluated": false', contract)
        self.assertIn('"train_plus_validation_refit": false', contract)


class TrainOnlyFitTests(unittest.TestCase):
    """작은 데이터에서 Validation이 학습에 안 들어가는지 확인한다."""

    def test_fit_does_not_see_validation_items(self):
        """train만 fit하면 validation 쌍은 seen에 없다."""
        with tempfile.TemporaryDirectory() as tmp:
            features = Path(tmp) / "movies_multihot_data.npz"
            raw = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            np.savez_compressed(
                features,
                multi_hot_norm=raw / norms,
                classes=np.array(["Action", "Comedy"], dtype=object),
                movie_ids=np.array([1, 2, 3]),
                titles=np.array(["A", "C", "AC"], dtype=object),
            )
            train = pd.DataFrame({"user_id": [1, 1], "item_id": [1, 2], "rating": [5.0, 1.0]})
            validation = pd.DataFrame({"user_id": [1], "item_id": [3], "rating": [4.0]})
            model = ContentBasedCF(item_features_path=features).fit(train)
            self.assertEqual(model.seen_[1], {1, 2})
            self.assertNotIn(3, model.seen_[1])
            score = model.predict(1, 3)
            self.assertTrue(1.0 <= score <= 5.0)
            self.assertEqual(len(validation), 1)


if __name__ == "__main__":
    unittest.main()
