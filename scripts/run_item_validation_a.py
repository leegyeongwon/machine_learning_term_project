import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.item_based import ItemBasedCF, validate_ratings
from src.hybrid import metrics
from scripts.run_full_recsys_pipeline import predict
from scripts.split_hybrid_validation import sha256


def main():
    source_path = ROOT / 'data/processed/movie_item_user_data.npz'
    split_dir = ROOT / 'data/processed/hybrid_validation'
    out = ROOT / 'results/item_based/validation_a'
    if out.exists():
        raise FileExistsError('Output directory already exists: ' + str(out))
    info = json.loads((split_dir / 'split_info.json').read_text(encoding='utf-8'))
    source_hash = sha256(source_path)
    if source_hash != info['source_sha256'] or info['seed'] != 42:
        raise ValueError('Source hash or split seed differs')
    with np.load(source_path, allow_pickle=False) as source:
        train = pd.DataFrame({'user_id': source['train_user_ids'],
                              'item_id': source['train_movie_ids'],
                              'rating': source['train_ratings']})
    with np.load(split_dir / 'validation_a.npz', allow_pickle=False) as source:
        val = pd.DataFrame({'user_id': source['validation_user_ids'],
                            'item_id': source['validation_movie_ids'],
                            'rating': source['validation_ratings']})
        val_indices = source['validation_indices']
    with np.load(split_dir / 'validation_split_indices.npz', allow_pickle=False) as indices:
        if not np.array_equal(val_indices, indices['a_indices']):
            raise ValueError('Validation A indices differ')
    if len(train) != info['counts']['train'] or len(val) != info['counts']['validation_a']:
        raise ValueError('Data counts differ')
    validate_ratings(val)
    print(json.dumps({'train_count': len(train), 'validation_a_count': len(val),
                      'stage': 'fit'}), flush=True)

    candidates = [20, 30, 50]
    item = ItemBasedCF(max(candidates)).fit(train)
    indices, similarities = item.neighbor_indices_, item.neighbor_similarities_
    pairs = val[['user_id', 'item_id']]
    rows = []
    for k in candidates:
        item.neighbor_indices_ = indices[:, :k]
        item.neighbor_similarities_ = similarities[:, :k]
        item.n_neighbors = k
        prediction, _ = predict(item, pairs)
        row = {'k': k, **metrics(val.rating, prediction), 'count': len(prediction)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    best = min(rows, key=lambda row: (row['mae'], row['rmse'], row['k']))
    if any(row['count'] != len(val) for row in rows):
        raise ValueError('Evaluation counts differ')
    if sha256(source_path) != source_hash:
        raise ValueError('Source changed during the run')
    selected = {
        'selected_k': best['k'], 'mae': best['mae'], 'rmse': best['rmse'],
        'k_candidates': candidates, 'train_data': 'Existing MovieLens 32M Train',
        'evaluation_data': 'Validation A', 'split_seed': 42,
        'train_count': len(train), 'validation_a_count': len(val),
        'selection': 'Lowest MAE, then RMSE, then K',
        'source_file': source_path.relative_to(ROOT).as_posix(),
        'source_sha256': source_hash,
    }
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / 'k_validation_metrics.csv', index=False)
    (out / 'selected_k.json').write_text(json.dumps(selected, indent=2) + '\n', encoding='utf-8')
    saved = pd.read_csv(out / 'k_validation_metrics.csv', float_precision='round_trip')
    if saved.to_dict('records') != rows:
        raise ValueError('Saved CSV differs from calculated metrics')
    if json.loads((out / 'selected_k.json').read_text(encoding='utf-8')) != selected:
        raise ValueError('Saved JSON differs from calculated selection')
    print(json.dumps({'selected_k': best['k'], 'verification': 'passed'}), flush=True)


if __name__ == '__main__':
    main()
