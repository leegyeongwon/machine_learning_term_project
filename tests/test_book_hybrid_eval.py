import unittest

import numpy as np
import pandas as pd

from scripts.run_book_hybrid_eval import aligned_positions, combine_predictions
from scripts.run_full_recsys_pipeline import scored
from src.item_based import ItemBasedCF


class BookHybridTests(unittest.TestCase):
    def test_id_alignment_preserves_requested_order(self):
        np.testing.assert_array_equal(aligned_positions([7, 2, 9], [9, 7]), [2, 0])
        with self.assertRaises(ValueError):
            aligned_positions([7, 2], [8])

    def test_fixed_weight_and_actual_ratings(self):
        item = pd.DataFrame({'user_id': [1, 2], 'item_id': [10, 20],
                             'rating': [1., 5.], 'predicted_rating': [2., 4.]})
        content = item.assign(predicted_rating=[4., 2.])
        result = combine_predictions(item, content, .98)
        np.testing.assert_allclose(result.predicted_rating, [2.04, 3.96])
        pd.testing.assert_frame_equal(result[['user_id', 'item_id', 'rating']], item[['user_id', 'item_id', 'rating']])
        for broken in [content.iloc[::-1], content.assign(rating=3.),
                       content.assign(predicted_rating=np.nan), content.assign(predicted_rating=.5)]:
            with self.assertRaises(ValueError):
                combine_predictions(item, broken, .98)
        with self.assertRaises(ValueError):
            combine_predictions(pd.concat([item, item]), pd.concat([content, content]), .98)

    def test_existing_item_prediction_and_fallback(self):
        train = pd.DataFrame([(u, i, float((u+i)%5+1)) for u in range(4) for i in range(5)],
                             columns=['user_id', 'item_id', 'rating'])
        model = ItemBasedCF(3).fit(train)
        pairs = pd.DataFrame({'user_id': [0, 99], 'item_id': [1, 99], 'rating': [3., 4.]})
        result = scored(model, pairs)
        np.testing.assert_allclose(result.predicted_rating, [model.predict(0, 1), model.predict(99, 99)])
        self.assertEqual(result.method.iloc[1], 'global_mean')
        self.assertTrue(result.predicted_rating.between(1, 5).all())


if __name__ == '__main__':
    unittest.main()
