"""봉인 후 정답만 읽어 strict 전이 결과를 독립 재검증."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.item_based_transfer import ItemBasedTransfer
from src.item_based_transfer_data import StrictArrayReader,save_json,verify_originals,sha256


def verify(run,processed):
    run=Path(run)
    seal=json.loads((run/'prediction_seal.json').read_text(encoding='utf-8'))
    for name,digest in seal.items():
        if sha256(name)!=digest:raise ValueError('동결 파일 변경: '+name)
    access=json.loads((run/'array_access.json').read_text(encoding='utf-8'))
    forbidden=[e for e in access if e['allowed'] and (
        (e['file'].startswith('book_') and e['key'].startswith(('train_','validation_')))
        or (e['file'].startswith('movie_') and e['key'].startswith('test_'))
        or any(x in e['key'] for x in ['plot','numeric','year','profile']))]
    if forbidden:raise ValueError('금지 배열 접근')
    truth_events=[e for e in access if e['file']=='book_item_user_data.npz' and e['key']=='test_ratings']
    if len(truth_events)!=1 or truth_events[0]['phase']!='evaluation':raise ValueError('정답 접근 순서 오류')
    phases=json.loads((run/'phase_journal.json').read_text(encoding='utf-8'))
    order={e['phase']:e['index'] for e in phases}
    if not order['source_model_and_configuration_frozen']<order['target_predictions_frozen_before_truth']<order['book_test_truth_first_opened_after_seal']:
        raise ValueError('단계 순서 오류')
    cfg=json.loads((run/'selected_config.json').read_text(encoding='utf-8'))
    grid=pd.read_csv(run/'movie_validation_grid.csv').sort_values(['mae','rmse','rank','alpha','k'])
    best=grid.iloc[0]
    if (int(best['rank']),float(best.alpha),int(best.k))!=(cfg['rank'],cfg['alpha'],cfg['k']):raise ValueError('선택 설정 불일치')
    reader=StrictArrayReader(processed)
    reader.allow_target_features([run/'source_model_frozen.npz',run/'selected_config.json'])
    catalog=reader.catalog('book')
    users=reader.read('book_item_user_data.npz','test_user_ids');items=reader.read('book_item_user_data.npz','test_book_ids')
    model=ItemBasedTransfer.load(run/'source_model_frozen.npz').bind_target_content(catalog)
    with np.load(run/'book_test_predictions_frozen.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(users,z['user_ids']);np.testing.assert_array_equal(items,z['book_ids'])
        predictions={k:z[k].copy() for k in ['distilled','genre','global_mean']}
    np.testing.assert_allclose(model.predict_many(users,items),predictions['distilled'],rtol=0,atol=1e-12)
    np.testing.assert_array_equal(model.predict_many(np.full(len(users),-999),items),predictions['distilled'])
    reader.seal_predictions(list(seal))
    truth=reader.read('book_item_user_data.npz','test_ratings')
    metrics=pd.read_csv(run/'book_test_metrics.csv').set_index('model')
    csv=pd.read_csv(run/'book_test_predictions_frozen.csv')
    np.testing.assert_array_equal(csv[['user_id','book_id']].to_numpy(),np.column_stack([users,items]))
    for label,key in [('cf_metric_distillation','distilled'),('raw_genre_cosine','genre'),('source_global_mean','global_mean')]:
        p=predictions[key]
        if not np.isfinite(p).all() or not np.all((p>=1)&(p<=5)):raise ValueError('예측 범위 오류')
        np.testing.assert_allclose(p,csv[key],rtol=0,atol=1e-12)
        error=p-truth
        np.testing.assert_allclose([np.abs(error).mean(),np.sqrt(np.mean(error**2))],metrics.loc[label,['mae','rmse']].to_numpy(dtype=float),rtol=1e-12,atol=1e-12)
    preservation=verify_originals(ROOT,json.loads((run/'originals_before.json').read_text(encoding='utf-8')))
    if not preservation['all_preserved']:raise ValueError('기존 파일 변경')
    result={'all_checks_passed':True,'prediction_seal_verified':True,'source_selection_verified':True,
            'book_train_validation_and_movie_test_never_decoded':True,'same_book_all_users_same_score':True,
            'save_load_predictions_identical':True,'prediction_csv_and_npz_identical':True,'metrics_recomputed':True,
            'originals_preserved':preservation,'verification_array_access':reader.events}
    save_json(run/'independent_verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--processed-dir',type=Path,default=ROOT/'data/processed')
    args=parser.parse_args();print(json.dumps(verify(args.run_dir,args.processed_dir),ensure_ascii=False,indent=2))
