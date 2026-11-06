"""새 Item 전용 경로의 접근 통제·협업 기여·재현성 검증."""
import os
from pathlib import Path
import unittest
import uuid
import numpy as np
import pandas as pd
from src.item_based_transfer import ItemBasedTransfer
from src.item_based_transfer_data import GenreCatalog,StrictArrayReader,verify_originals,sha256

ROOT=Path(__file__).resolve().parents[1]


class StrictTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=os.environ.get('STRICT_TRANSFER_TEST_DIR')
        cls.folder=Path(base) if base else ROOT/'results/item_based'/('strict_transfer_tests_'+uuid.uuid4().hex)
        cls.folder.mkdir(parents=True,exist_ok=True)

    def setUp(self):
        self.folder=self.folder/uuid.uuid4().hex;self.folder.mkdir()
        self.train=pd.DataFrame([(u,i,.5+.5*((u*3+i*i+u*i)%10)) for u in range(8) for i in range(6)],columns=['user_id','item_id','rating'])
        binary=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,1,0,0],[0,1,1,0],[1,0,1,0],[0,0,0,1]],float)
        self.catalog=GenreCatalog(np.arange(7),['Fantasy','Crime','Science Fiction','target-only'],binary)
        self.model=ItemBasedTransfer(3,1.,3,4).fit(self.train,self.catalog)
        self.book=GenreCatalog(np.array([0,1,100]),['fantasy','crime','science-fiction','target-only'],binary[[1,2,6]])

    def test_source_vocabulary_train_only(self):
        self.assertNotIn('targetonly',self.model.genre_names_)
        np.testing.assert_array_equal(self.model.anchor_ids_,np.arange(6))
        before=self.model.coef_.copy();vocab=list(self.model.genre_names_)
        self.model.bind_target_content(self.book)
        self.assertEqual(vocab,self.model.genre_names_);np.testing.assert_array_equal(before,self.model.coef_)

    def test_users_ignored_and_namespaces_separated(self):
        self.model.bind_target_content(self.book)
        a=self.model.predict_many([0,7,-1],[0,0,0])
        np.testing.assert_array_equal(a,np.full(3,a[0]))
        # 같은 콘텐츠를 다른 책 ID로 주어도 원래 영화 ID lookup은 하지 않음
        other=GenreCatalog([999,888,777],self.book.names,self.book.binary)
        p=self.model.bound_scores_.copy();self.model.bind_target_content(other)
        np.testing.assert_array_equal(p,self.model.bound_scores_)
        with self.assertRaises(ValueError):self.model.bind_target_content(self.book,namespace='movie')

    def test_zero_genre_and_unknown_fallback(self):
        self.model.bind_target_content(self.book)
        self.assertEqual(self.model.predict(0,100),self.model.global_mean_)
        self.assertEqual(self.model.predict(0,99999),self.model.global_mean_)
        self.assertTrue(self.model.recommend(0,2).predicted_rating.between(1,5).all())

    def test_cf_teacher_permutation_changes_predictions(self):
        normal=self.model.score_catalog(self.catalog,namespace='movie')[0][3]
        shuffled=self.model.candidate(3,1.,3,314159).score_catalog(self.catalog,namespace='movie')[0][3]
        self.assertGreater(float(np.max(np.abs(normal-shuffled))),1e-8)
        self.assertGreater(float(np.std(self.model.teacher_raw_)),0)

    def test_save_load_exact(self):
        path=self.folder/'model.npz';self.model.save(path)
        loaded=ItemBasedTransfer.load(path).bind_target_content(self.book)
        self.model.bind_target_content(self.book)
        np.testing.assert_array_equal(loaded.predict_many([0,1,2],[0,1,100]),self.model.predict_many([99,88,77],[0,1,100]))
        with self.assertRaises(FileExistsError):self.model.save(path)

    def test_reproducibility_and_rank_configuration(self):
        second=ItemBasedTransfer(3,1.,3,4).fit(self.train.sample(frac=1,random_state=4),self.catalog)
        np.testing.assert_allclose(self.model.coef_,second.coef_,rtol=0,atol=1e-10)
        with self.assertRaises(ValueError):self.model.candidate(99,1,3)
        with self.assertRaises(ValueError):self.model.predict(1,1)

    def fake_book_file(self,truth):
        np.savez(self.folder/'book_item_user_data.npz',test_user_ids=[0,1],test_book_ids=[0,1],test_ratings=truth,
                 train_ratings=np.array([{'poison':'do not unpickle'}],dtype=object),validation_ratings=[-999])

    def test_forbidden_arrays_blocked_before_npz_read(self):
        reader=StrictArrayReader(self.folder)
        for file,key in [('book_item_user_data.npz','train_ratings'),('book_item_user_data.npz','validation_ratings'),
                         ('book_item_user_data.npz','test_ratings'),('movie_item_user_data.npz','test_ratings'),
                         ('book_user_profiles.npz','plot_profiles'),('movie_content.npz','plot_embeddings')]:
            with self.assertRaises(PermissionError):reader.read(file,key)
        self.assertTrue(all(not e['allowed'] for e in reader.events))

    def test_truth_after_seal_and_poison_not_materialized(self):
        self.fake_book_file([1,5]);frozen=self.folder/'frozen.txt';frozen.write_text('frozen')
        pred=self.folder/'pred.npy';np.save(pred,[3.,3.])
        reader=StrictArrayReader(self.folder);reader.allow_target_features([frozen])
        np.testing.assert_array_equal(reader.read('book_item_user_data.npz','test_book_ids'),[0,1])
        with self.assertRaises(PermissionError):reader.read('book_item_user_data.npz','test_ratings')
        reader.seal_predictions([pred]);np.testing.assert_array_equal(reader.read('book_item_user_data.npz','test_ratings'),[1,5])
        self.assertFalse(any(e['allowed'] and e['key'].startswith(('train_','validation_')) for e in reader.events))

    def test_target_truth_mutation_does_not_change_predictions(self):
        self.fake_book_file([1,5]);self.model.bind_target_content(self.book)
        original=self.model.predict_many([0,1],[0,1])
        self.fake_book_file([5,1])
        np.testing.assert_array_equal(original,self.model.predict_many([999,-3],[0,1]))

    def test_sealed_file_tampering_blocks_truth(self):
        self.fake_book_file([1,5]);frozen=self.folder/'frozen.txt';frozen.write_text('frozen')
        prediction=self.folder/'pred.txt';prediction.write_text('3,3')
        reader=StrictArrayReader(self.folder);reader.allow_target_features([frozen]);reader.seal_predictions([prediction])
        prediction.write_text('4,4')
        with self.assertRaises(PermissionError):reader.read('book_item_user_data.npz','test_ratings')

    def test_protected_file_hash_detection(self):
        path=self.folder/'protected.txt';path.write_text('original');stat=path.stat()
        record=[{'path':str(path.relative_to(ROOT)),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':sha256(path)}]
        self.assertTrue(verify_originals(ROOT,record)['all_preserved'])
        path.write_text('changed')
        self.assertFalse(verify_originals(ROOT,record)['all_preserved'])

    def test_metric_recalculation(self):
        from scripts.run_item_based_strict_transfer import errors
        result=errors([1.,3.,5.],[2.,3.,4.])
        self.assertAlmostEqual(result['mae'],2/3);self.assertAlmostEqual(result['rmse'],np.sqrt(2/3))


if __name__=='__main__':unittest.main()
