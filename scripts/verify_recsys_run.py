"""실제 결과 CSV의 정답·가중치·MAE/RMSE 독립 재검증."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.hybrid import metrics
from src.processed_validation import sha256


def verify(run,processed):
    table=pd.read_csv(run/'comparison.csv')
    selection=json.loads((run/'movielens_32m/selection.json').read_text(encoding='utf-8'))
    validation=pd.read_csv(run/'movielens_32m/validation_metrics.csv')
    for model,parameter in [('Item-based','k'),('Content-based','ridge_alpha')]:
        expected=validation[validation.model.eq(model)].sort_values(['mae','rmse','parameter']).iloc[0].parameter
        if expected!=selection[parameter]:raise ValueError('Validation 최적 설정 불일치')
    grid=pd.read_csv(run/'movielens_32m/hybrid_grid.csv')
    expected=grid.sort_values(['mae','rmse','alpha']).iloc[0].alpha
    if expected!=selection['hybrid_alpha']:raise ValueError('Hybrid Validation 최적 가중치 불일치')
    checks=[]
    for domain,folder,stem in [('movie','movielens_32m','movie'),('book_support_adapted','goodbooks','book')]:
        with np.load(processed/(stem+'_item_user_data.npz'),allow_pickle=False) as z:
            truth=pd.DataFrame({'user_id':z['test_user_ids'],'item_id':z['test_'+stem+'_ids'],'rating':z['test_ratings']})
        predictions={}
        for name in ['Item-based','Content-based','Hybrid']:
            path=run/folder/(name.lower()+'_test_predictions.csv')
            frame=pd.read_csv(path)
            np.testing.assert_array_equal(frame[['user_id','item_id','rating']].to_numpy(),truth.to_numpy())
            if frame.duplicated(['user_id','item_id']).any() or not np.isfinite(frame.predicted_rating).all():raise ValueError('중복·비유한 예측')
            low=.5 if stem=='movie' else 1.
            if not frame.predicted_rating.between(low,5).all():raise ValueError('예측 범위 오류')
            actual=metrics(frame.rating,frame.predicted_rating)
            saved=table[table.domain.eq(domain)&table.model.eq(name)].iloc[0]
            np.testing.assert_allclose([actual['mae'],actual['rmse']],[saved.mae,saved.rmse],atol=1e-12)
            checks.append({'domain':domain,'model':name,'rows':len(frame),'sha256':sha256(path),**actual})
            predictions[name]=frame.predicted_rating.to_numpy()
        weight=selection['hybrid_alpha']
        np.testing.assert_allclose(predictions['Hybrid'],weight*predictions['Item-based']+(1-weight)*predictions['Content-based'],atol=1e-12)
    inventory=json.loads((run/'input_inventory.json').read_text(encoding='utf-8'))
    for name,meta in inventory.items():
        if sha256(processed/name)!=meta['sha256']:raise ValueError('원본 NPZ 변경: '+name)
    result={'all_checks_passed':True,'validation_selection_verified':True,'fixed_hybrid_formula_both_domains_verified':True,
            'test_pairs_truth_order_verified':True,'all_input_npz_unchanged':True,'predictions':checks}
    (run/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--processed-dir',type=Path,default=Path('data/processed'))
    args=parser.parse_args()
    print(json.dumps(verify(args.run_dir,args.processed_dir),indent=2))
