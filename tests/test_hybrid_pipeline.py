"""가상 데이터 단위 검증. 실제 통합 실행과 분리."""
import tempfile
from io import BytesIO
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.hybrid import HybridCF, align_predictions, tune_weight, metrics
from src.transfer_data import SharedContentCF, load_splits, validate_splits, load_content
from src.item_based import ItemBasedCF
from scripts.run_full_recsys_pipeline import predict
from src.transfer_data import canonical_genre


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.train=pd.DataFrame([(u,i,float((u+i)%5+1)) for u in range(4) for i in range(5)],columns=['user_id','item_id','rating'])
        self.content=(np.arange(5),np.array(['Drama','Fantasy']),np.array([[1,0],[0,1],[1,1],[1,0],[0,1]],float),np.arange(2000,2005,dtype=float))

    def test_weight_and_range(self):
        a,t=tune_weight([1,5],[1,5],[5,1])
        self.assertEqual(a,1.)
        self.assertEqual(len(t),101)
        with self.assertRaises(ValueError):HybridCF(np.nan)
        with self.assertRaises(ValueError):metrics([1],[np.nan])

    def test_alignment(self):
        f=self.train.copy();f['predicted_rating']=3.
        self.assertEqual(len(align_predictions(f,f.sample(frac=1,random_state=42))),len(f))
        for broken in [f.iloc[:-1],pd.concat([f,f.iloc[:1]]),f.assign(rating=1)]:
            with self.assertRaises(ValueError):align_predictions(f,broken)

    def test_item_batch_matches_original_and_cold_start(self):
        m=ItemBasedCF(3).fit(self.train)
        pairs=pd.DataFrame({'user_id':[0,1,99,99],'item_id':[1,99,1,99]})
        p,methods=predict(m,pairs)
        np.testing.assert_allclose(p,[m.predict(u,i) for u,i in pairs.itertuples(index=False,name=None)])
        self.assertEqual(methods[-1],'global_mean')

    def test_hybrid_public_api(self):
        m=HybridCF(.3,ItemBasedCF(3),SharedContentCF(self.content)).fit(self.train)
        self.assertTrue(np.isfinite(m.predict(99,99)))
        np.testing.assert_allclose(m.predict_many([0,99],[1,99]),[m.predict(0,1),m.predict(99,99)])
        self.assertTrue(m.recommend(0).empty)

    def test_transfer_freezes_coefficients_and_aligns(self):
        source=SharedContentCF(self.content).fit(self.train)
        ids,names,g,years=self.content
        target=(ids+100,names[::-1],g[:,::-1],years)
        support=self.train.assign(item_id=self.train.item_id+100)
        book=source.transfer(target,support)
        np.testing.assert_array_equal(source.ridge_.coef_,book.ridge_.coef_)
        np.testing.assert_allclose(source.item_tfidf_,book.item_tfidf_)
        self.assertTrue(set(source.item_ids_).isdisjoint(book.item_ids_))
        np.testing.assert_allclose(source.predict_many([0],[1])[0],book.predict_many([0],[101])[0])
        self.assertNotIn(999,book.user_means_)

    def test_support_is_only_profile_input(self):
        model=SharedContentCF(self.content).fit(self.train)
        support=self.train[self.train.user_id<3]
        a=model.transfer(self.content,support)
        b=model.transfer(self.content,support.copy())
        np.testing.assert_array_equal(a.user_profiles_,b.user_profiles_)
        self.assertNotIn(3,a.user_index_)
        self.assertEqual(a.predict_with_details(99,99)[1],'global_mean')

    def test_npz_split_loading_overlap_and_reproducibility(self):
        parts={s:f.reset_index(drop=True) for s,f in zip(['train','validation','test'],[self.train[:16],self.train[16:18],self.train[18:]])}
        validate_splits(parts)
        path=BytesIO()
        arrays={s+'_'+key:f[col].to_numpy() for s,f in parts.items() for key,col in [('user_ids','user_id'),('movie_ids','item_id'),('ratings','rating')]}
        np.savez(path,**arrays)
        path.seek(0)
        loaded=load_splits(path)
        for s in parts:pd.testing.assert_frame_equal(parts[s],loaded[s])
        parts['test']=parts['validation']
        with self.assertRaises(ValueError):validate_splits(parts)

    def test_saved_metric_recalculation(self):
        actual=np.array([1.,3.,5.]);pred=np.array([2.,3.,4.])
        expected=metrics(actual,pred)
        self.assertAlmostEqual(expected['mae'],2/3)
        self.assertAlmostEqual(expected['rmse'],np.sqrt(2/3))

    def test_no_answers_in_prediction_interface(self):
        with self.assertRaises(ValueError):predict(ItemBasedCF(3).fit(self.train),self.train)

    def test_content_npz_schema(self):
        from scipy.sparse import csr_matrix
        ids,names,g,year=self.content
        sparse=csr_matrix(g)
        buffer=BytesIO()
        np.savez(buffer,movie_ids=ids,genre_names=names,genres_data=sparse.data,
                 genres_indices=sparse.indices,genres_indptr=sparse.indptr,genres_shape=sparse.shape,
                 numeric_feature_names=np.array(['runtime','release_year']),
                 numeric_features=np.column_stack([np.ones(5),(year-2000)/4]),year_min=2000.,year_max=2004.)
        buffer.seek(0)
        loaded=load_content(buffer,'movie')
        for a,b in zip(self.content,loaded):np.testing.assert_array_equal(a,b)

    def test_genre_case_hyphen_alignment(self):
        self.assertEqual(canonical_genre('Science Fiction'),canonical_genre('science-fiction'))
        self.assertNotEqual(canonical_genre('Drama'),canonical_genre('fiction'))
        source=SharedContentCF(self.content).fit(self.train)
        ids,names,g,year=self.content
        target=(ids,np.array(['drama','fantasy']),g,year)
        book=source.transfer(target,self.train)
        np.testing.assert_allclose(source.item_tfidf_,book.item_tfidf_)

    def test_cached_ridge_matches_direct_objective(self):
        m=SharedContentCF(self.content).fit(self.train)
        users=np.array([m.user_index_[u] for u in self.train.user_id])
        items=np.array([m.item_index_[i] for i in self.train.item_id])
        x=m._features_from_indices(users,items).astype(float)
        y=self.train.rating.to_numpy()-m.user_mean_arr_[users]
        from sklearn.linear_model import Ridge
        for alpha in [1.,10.,100.]:
            expected=Ridge(alpha=alpha).fit(x,y)
            m.set_alpha(alpha)
            np.testing.assert_allclose(m.ridge_.predict(x),expected.predict(x),atol=1e-10)

    def test_fixed_plot_projection_and_id_collision(self):
        rng=np.random.default_rng(42)
        source=SharedContentCF(self.content+(rng.normal(size=(5,8)).astype(np.float32),),plot_components=3).fit(self.train)
        # 숫자 ID가 같아도 도서 support 프로필로 완전히 교체
        support=self.train.assign(rating=6-self.train.rating)
        book=source.transfer(source.content,support)
        np.testing.assert_array_equal(source.plot_projection_.components_,book.plot_projection_.components_)
        np.testing.assert_array_equal(source.ridge_.coef_,book.ridge_.coef_)
        self.assertFalse(np.allclose(source.user_profiles_,book.user_profiles_))
        self.assertEqual(source.feature_names_,book.feature_names_)
        self.assertTrue(np.isfinite(book.predict_many([0,999],[1,999])[0]).all())

    def test_svd_fit_excludes_holdout_only_items(self):
        rng=np.random.default_rng(7)
        plot=rng.normal(size=(5,8)).astype(np.float32)
        train=self.train[self.train.item_id.lt(4)]
        first=SharedContentCF(self.content+(plot,),plot_components=3).fit(train)
        changed=plot.copy();changed[4]*=-100
        second=SharedContentCF(self.content+(changed,),plot_components=3).fit(train)
        np.testing.assert_array_equal(first.plot_projection_.components_,second.plot_projection_.components_)
        np.testing.assert_allclose(first.ridge_.coef_,second.ridge_.coef_,atol=1e-10)


if __name__=='__main__':unittest.main()
