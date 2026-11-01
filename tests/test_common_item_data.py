"""공통 NPZ 및 Validation 산출물 통합 테스트.

scripts/run_item_based_common_split.py 실행 결과가 필요하다.
"""

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.common_item_data import load_common_splits, validate_common_data
from src.item_based import ItemBasedCF
from scripts.run_item_based_common_split import (choose_best_k, validate_hybrid_predictions,
                                                 triple_fingerprint, sha256)

ROOT=Path(__file__).resolve().parents[1]
NPZ=ROOT/'data/processed/item_user_data.npz'
OUT=ROOT/'results/item_based/common_split_811'


class CommonSplitTests(unittest.TestCase):
    """공통 분할 로딩, 누수 검사, 모델 연동 및 Hybrid 산출물 검증."""

    @classmethod
    def setUpClass(cls):
        """공통 분할 및 기존 실행 산출물 로딩."""
        cls.parts,cls.report=validate_common_data(
            NPZ,ROOT/'data/raw/ml-1m/ratings.dat',
            ROOT/'preprocessing_and_data/movielens-1m/preprocess_item_based.py')
        cls.metrics=pd.read_csv(OUT/'validation_metrics.csv')
        cls.predictions={k:pd.read_csv(OUT/f'validation_predictions_k{k}.csv') for k in [20,30,50]}

    def test_npz_loading_and_actual_source_split(self):
        """NPZ 로딩 및 원문 분할 결과 일치 여부 검증."""
        loaded=load_common_splits(NPZ)
        for name in self.parts:
            pd.testing.assert_frame_equal(loaded[name],self.parts[name])
        self.assertTrue(self.report['all_checks_passed'])
        self.assertTrue(self.report['source_split_exact_order_match'])
        self.assertEqual(self.report['random_state'],42)
        self.assertEqual(self.report['sum_split_count'],1000209)
        self.assertTrue(all(value==0 for value in self.report['overlap_counts'].values()))

    def test_train_only_statistics_and_mask(self):
        """Train 통계·마스크와 holdout 미관측 상태 검증."""
        checks=self.report['matrix_checks']
        self.assertTrue(checks['train_only_means_stds_verified'])
        self.assertTrue(checks['train_only_mask_verified'])
        self.assertTrue(checks['holdout_mask_and_matrix_zero'])
        self.assertEqual(checks['mask_nnz'],len(self.parts['train']))
        self.assertLessEqual(checks['zscore_max_absolute_difference'],1e-6)

    def test_existing_model_fit_on_actual_small_train(self):
        """실제 Train 일부로 기존 모델의 원점수 입력 호환성 검증."""
        users=self.parts['train'].user_id.unique()[:5]
        small=self.parts['train'][self.parts['train'].user_id.isin(users)]
        validation=self.parts['validation'][self.parts['validation'].user_id.isin(users)].head(20)
        model=ItemBasedCF(20).fit(small)
        self.assertEqual(model.user_item_.nnz,len(small))
        for row in validation.itertuples():
            value=model.predict(row.user_id,row.item_id)
            self.assertTrue(np.isfinite(value))
            self.assertTrue(1<=value<=5)

    def test_all_k_validation_metrics(self):
        """K별 저장 예측으로 MAE·RMSE·CF 비율 재계산."""
        validation=self.parts['validation']
        self.assertEqual(set(self.metrics.k),{20,30,50})
        for row in self.metrics.itertuples():
            p=self.predictions[row.k]
            pd.testing.assert_frame_equal(p[['user_id','item_id','rating']],validation)
            self.assertEqual(validate_hybrid_predictions(p,validation)['rows'],len(validation))
            error=p.predicted_rating-p.rating
            self.assertAlmostEqual(float(error.abs().mean()),row.mae,places=12)
            self.assertAlmostEqual(float(np.sqrt(np.mean(error**2))),row.rmse,places=12)
            self.assertEqual(row.n_evaluated,len(validation))
            self.assertEqual(row.n_cf+row.n_fallback,len(validation))
            self.assertAlmostEqual(float(p.method.eq('cf').mean()),row.cf_coverage)

    def test_best_k_and_hybrid_csv(self):
        """선택 K와 Train-only Hybrid CSV의 행·해시 검증."""
        best=choose_best_k(self.metrics)
        self.assertEqual(best,int(self.metrics.sort_values(['mae','rmse','k']).iloc[0].k))
        path=OUT/'hybrid_ready_validation_predictions.csv'
        saved=pd.read_csv(path)
        pd.testing.assert_frame_equal(saved,self.predictions[best])
        self.assertTrue(validate_hybrid_predictions(saved,self.parts['validation'])['finite_and_in_1_to_5'])
        contract=json.loads((OUT/'hybrid_contract.json').read_text(encoding='utf-8'))
        self.assertEqual(contract['best_k'],best)
        self.assertEqual(contract['csv_sha256'],sha256(path))
        self.assertEqual(contract['train_triple_sha256'],triple_fingerprint(self.parts['train']))
        self.assertEqual(contract['validation_triple_sha256'],triple_fingerprint(self.parts['validation']))
        self.assertFalse(contract['train_plus_validation_refit'])
        self.assertFalse(contract['test_evaluated'])

    def test_selection_prioritizes_mae_and_resolves_ties(self):
        """MAE 우선 선택 및 RMSE·작은 K 동률 처리 검증."""
        changed=self.metrics.copy()
        best=choose_best_k(changed)
        changed.loc[changed.k.eq(best),'rmse']=changed.rmse.max()+1
        self.assertEqual(choose_best_k(changed),best)
        changed['mae']=changed.mae.min()
        changed['rmse']=changed.rmse.max()
        self.assertEqual(choose_best_k(changed),20)
        changed.loc[changed.k.eq(30),'rmse']=changed.rmse.min()-0.1
        self.assertEqual(choose_best_k(changed),30)

    def test_hybrid_rejects_duplicate_or_missing_predictions(self):
        """중복·누락 예측의 검증 실패 확인."""
        val=self.parts['validation'].head(5)
        good=self.predictions[20].head(5)
        validate_hybrid_predictions(good,val)
        for bad in [good.iloc[:-1],pd.concat([good.iloc[:4],good.iloc[:1]])]:
            with self.assertRaises(ValueError):
                validate_hybrid_predictions(bad,val)


if __name__=='__main__':
    unittest.main()
