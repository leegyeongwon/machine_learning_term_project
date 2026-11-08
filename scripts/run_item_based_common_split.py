"""공통 전처리·NPZ 검증·원점수 CF 평가·Hybrid CSV 생성.

기존 NPZ는 검증 후 재사용하며 결과 덮어쓰기는 허용하지 않는다.
Test는 무결성 검사에만 사용한다.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import psutil
from src.common_item_data import validate_common_data
from src.item_based import ItemBasedCF
from scripts.run_item_based import evaluate


def sha256(path):
    """파일 SHA-256 반환."""
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_json(path, value):
    """결과를 UTF-8 JSON으로 저장."""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def choose_best_k(metrics):
    """Validation MAE, RMSE, 작은 K 순으로 최적 K 선택."""
    return int(metrics.sort_values(['mae','rmse','k']).iloc[0]['k'])


def validate_hybrid_predictions(predictions, validation):
    """Validation 일대일 대응, 중복·누락, 평점 범위 검증."""
    keys = ['user_id','item_id']
    if predictions.duplicated(keys).any() or len(predictions) != len(validation):
        raise ValueError('Hybrid 예측에 중복 또는 누락이 있습니다.')
    paired = validation.merge(predictions, on=keys, how='outer', suffixes=('_actual','_saved'),
                               indicator=True, validate='one_to_one')
    if not paired['_merge'].eq('both').all():
        raise ValueError('Validation ID가 다릅니다.')
    np.testing.assert_array_equal(paired.rating_actual, paired.rating_saved)
    values = predictions.predicted_rating.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 1) & (values <= 5)).all():
        raise ValueError('예측은 유한한 1~5점이어야 합니다.')
    if not predictions.n_contributors.ge(0).all():
        raise ValueError('잘못된 기여 이웃 수입니다.')
    cf = predictions.method.eq('cf')
    if not predictions.loc[cf,'n_contributors'].gt(0).all() or not predictions.loc[~cf,'n_contributors'].eq(0).all():
        raise ValueError('예측 방식과 기여 이웃 수가 불일치합니다.')
    return {'rows':len(predictions), 'one_prediction_per_validation_pair':True,
            'original_ids_and_ratings_match':True, 'finite_and_in_1_to_5':True,
            'method_contributors_consistent':True}


def triple_fingerprint(frame):
    """행 순서에 독립적인 평점 triple SHA-256 반환."""
    text = frame[['user_id','item_id','rating']].sort_values(['user_id','item_id']).to_csv(index=False, lineterminator='\n')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main():
    """공통 Train 학습 및 Validation 기반 K 선택·예측 저장."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/item_based/common_split_811')
    parser.add_argument('--npz', type=Path, default=ROOT/'data/processed/item_user_data.npz')
    args = parser.parse_args()
    out, npz_path = args.output_dir.resolve(), args.npz.resolve()
    source = ROOT/'preprocessing_and_data/movielens-1m/preprocess_item_based.py'
    source_meta = json.loads((ROOT/'scripts/common_split_811_source.json').read_text(encoding='utf-8'))
    source_bytes = source.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(source_bytes)).encode()+b'\0'+source_bytes).hexdigest()
    if blob != source_meta['git_blob_sha']:
        raise ValueError('전처리 원문이 출처 메타데이터와 다릅니다.')
    if (out/'validation_metrics.csv').exists():
        parser.error('기존 결과를 보존합니다. 새 --output-dir을 지정하세요.')
    branch = subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip()
    if branch != 'feature/item-based':
        raise ValueError('feature/item-based에서만 실행하세요.')
    out.mkdir(parents=True, exist_ok=True)
    if not (out/'.gitignore').exists():
        (out/'.gitignore').write_text('*\n', encoding='utf-8')
    started = time.perf_counter()
    generated = False
    if not npz_path.exists():
        if npz_path != ROOT/'data/processed/item_user_data.npz':
            parser.error('팀 전처리의 기본 출력 NPZ가 없습니다.')
        env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
        with (out/'preprocessing_log.txt').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-B', str(source)], cwd=ROOT/'data/raw/ml-1m',
                           stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        generated = True
    content_paths = [path for path in (ROOT/'data/processed').rglob('*.npz')
                     if path.name != 'item_user_data.npz' and
                     ('user_profiles' in path.name or 'content' in path.name or 'movies_multihot' in path.name)]
    parts, checks = validate_common_data(npz_path, ROOT/'data/raw/ml-1m/ratings.dat', source, content_paths)
    save_json(out/'data_validation_report.json',checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    train, validation = parts['train'], parts['validation']
    del parts  # 이후 학습·평가에서 Test 제외
    rows = []
    for k in [20,30,50]:
        tick = time.perf_counter()
        model = ItemBasedCF(n_neighbors=k).fit(train)
        fit_seconds = time.perf_counter()-tick
        if model.user_item_.nnz != len(train) or model.global_mean_ != float(train.rating.mean()):
            raise ValueError('モデル学習データがTrainと一致しません。')
        # 모델 학습 행렬의 Validation 위치가 미평가인지 확인
        u = model.user_ids_.get_indexer(validation.user_id)
        i = model.item_ids_.get_indexer(validation.item_id)
        known = (u>=0)&(i>=0)
        if np.asarray(model.user_item_[u[known],i[known]]).any():
            raise ValueError('Validationが学習に混入しています。')
        tick = time.perf_counter()
        metrics, predictions = evaluate(model, validation)
        evaluation_seconds = time.perf_counter()-tick
        metrics['n_validation'] = metrics.pop('n_test')
        metrics['unknown_validation_items'] = metrics.pop('unknown_test_items')
        metrics.update(k=k, fit_seconds=fit_seconds, evaluation_seconds=evaluation_seconds,
                       fallback_methods=predictions.loc[predictions.method.ne('cf'),'method'].value_counts().to_dict())
        validate_hybrid_predictions(predictions, validation)
        predictions.to_csv(out/f'validation_predictions_k{k}.csv',index=False)
        rows.append(metrics)
        print(json.dumps(metrics,ensure_ascii=False),flush=True)
        del model
    table = pd.DataFrame([{key:value for key,value in row.items() if key!='fallback_methods'} for row in rows])
    table.to_csv(out/'validation_metrics.csv',index=False)
    save_json(out/'validation_metrics.json',rows)
    best_k = choose_best_k(table)
    selected = pd.read_csv(out/f'validation_predictions_k{best_k}.csv')
    selected.to_csv(out/'hybrid_ready_validation_predictions.csv',index=False)
    saved = pd.read_csv(out/'hybrid_ready_validation_predictions.csv')
    hybrid_checks = validate_hybrid_predictions(saved, validation)
    error = saved.predicted_rating-saved.rating
    best = table.set_index('k').loc[best_k]
    np.testing.assert_allclose([error.abs().mean(),np.sqrt(np.mean(error**2))],
                               [best.mae,best.rmse],rtol=1e-12)
    save_json(out/'hybrid_contract.json',{
        'best_k':best_k,'selection':'Validation MAE, then RMSE, then smaller K',
        'model':'existing raw-rating ItemBasedCF', 'fit_data':'common Train only',
        'train_plus_validation_refit':False, 'test_evaluated':False,
        'merge_keys':['user_id','item_id'], 'rating_scale':[1,5],
        'method_values':sorted(saved.method.unique().tolist()),
        'train_triple_sha256':triple_fingerprint(train),
        'validation_triple_sha256':triple_fingerprint(validation),
        'npz_sha256':sha256(npz_path),
        'csv_sha256':sha256(out/'hybrid_ready_validation_predictions.csv'),
        'checks':hybrid_checks,
        'content_split_verified':any(x['status']=='all_split_triples_identical' for x in checks['content_comparison'])})
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(table.k,table.mae,'o-',label='Validation MAE')
    ax.plot(table.k,table.rmse,'o-',label='Validation RMSE')
    ax.set(xlabel='K',ylabel='Rating error',title='Team common global 80/10/10 split')
    ax.set_xticks(table.k);ax.legend();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(out/'validation_metrics.png',dpi=160);plt.close(fig)
    save_json(out/'run_summary.json',{
        'source':source_meta,'source_sha256':sha256(source),
        'raw_ratings_sha256':sha256(ROOT/'data/raw/ml-1m/ratings.dat'),
        'model_sha256':sha256(ROOT/'src/item_based.py'),
        'npz_sha256':sha256(npz_path),'npz_generated_this_run':generated,
        'best_k':best_k,'test_evaluated':False,'total_seconds':time.perf_counter()-started,
        'peak_working_set_mib':getattr(psutil.Process().memory_info(),'peak_wset',0)/2**20})
    print(table.to_string(index=False),flush=True)
    print(f'Best K={best_k}; Hybrid rows={len(saved)}; Test not evaluated.',flush=True)


if __name__=='__main__':
    main()
