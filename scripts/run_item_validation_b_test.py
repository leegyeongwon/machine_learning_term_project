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
from scripts.run_full_recsys_pipeline import scored
from scripts.split_hybrid_validation import sha256


def load_ratings(source, split):
    return pd.DataFrame({'user_id': source[split + '_user_ids'],
                         'item_id': source[split + '_movie_ids'],
                         'rating': source[split + '_ratings']})


def save_predictions(item, frame, path):
    validate_ratings(frame)
    result = scored(item, frame)
    values = result.predicted_rating.to_numpy()
    if not np.isfinite(values).all() or not result.predicted_rating.between(*item.rating_range_).all():
        raise ValueError('Invalid prediction values')
    if not result.method.isin(['cf', 'user_mean', 'item_mean', 'global_mean']).all():
        raise ValueError('Unknown prediction method')
    result.to_csv(path, index=False, mode='x')
    saved = pd.read_csv(path, float_precision='round_trip',
                        dtype={'user_id': frame.user_id.dtype, 'item_id': frame.item_id.dtype,
                               'rating': frame.rating.dtype})
    for key in result.columns:
        if not np.array_equal(saved[key].to_numpy(), result[key].to_numpy()):
            raise ValueError('CSV round-trip mismatch: ' + key)
    if saved.duplicated(['user_id', 'item_id']).any():
        raise ValueError('Duplicate prediction pairs')
    print(json.dumps({'file': path.name, 'count': len(saved),
                      **metrics(saved.rating, saved.predicted_rating),
                      'csv_verification': 'passed'}), flush=True)


def main():
    source_path = ROOT / 'data/processed/movie_item_user_data.npz'
    split_dir = ROOT / 'data/processed/hybrid_validation'
    out = ROOT / 'results/item_based/hybrid_update'
    if out.exists():
        raise FileExistsError('Output directory already exists: ' + str(out))
    info = json.loads((split_dir / 'split_info.json').read_text(encoding='utf-8'))
    selected = json.loads((ROOT / 'results/item_based/validation_a/selected_k.json').read_text(encoding='utf-8'))
    k = selected['selected_k']
    source_hash = sha256(source_path)
    if source_hash != info['source_sha256'] or source_hash != selected['source_sha256']:
        raise ValueError('Source hash differs')
    if k != 50 or info['seed'] != 42 or selected['split_seed'] != 42:
        raise ValueError('Selected K or split seed differs')
    with np.load(split_dir / 'validation_b.npz', allow_pickle=False) as source:
        val = load_ratings(source, 'validation')
        val_indices = source['validation_indices']
    with np.load(split_dir / 'validation_split_indices.npz', allow_pickle=False) as indices:
        if not np.array_equal(val_indices, indices['b_indices']):
            raise ValueError('Validation B indices differ')
    with np.load(source_path, allow_pickle=False) as source:
        train = load_ratings(source, 'train')
        test = load_ratings(source, 'test')
        for column, suffix in [('user_id', 'user_ids'), ('item_id', 'movie_ids'), ('rating', 'ratings')]:
            if not np.array_equal(val[column].to_numpy(), source['validation_' + suffix][val_indices]):
                raise ValueError('Validation B rows differ from source: ' + column)
    if len(train) != 17401383 or len(train) != info['counts']['train'] or len(train) != selected['train_count']:
        raise ValueError('Train count differs')
    if len(val) != 1087587 or len(val) != info['counts']['validation_b']:
        raise ValueError('Validation B count differs')
    if len(test) != 2175173 or len(test) != info['counts']['test']:
        raise ValueError('Test count differs')
    print(json.dumps({'k': k, 'train_count': len(train), 'stage': 'fit'}), flush=True)
    item = ItemBasedCF(k).fit(train)
    out.mkdir(parents=True)
    save_predictions(item, val, out / 'item_validation_b_predictions.csv')
    save_predictions(item, test, out / 'item_test_predictions.csv')
    if sha256(source_path) != source_hash:
        raise ValueError('Source changed during the run')
    print('Verification passed', flush=True)


if __name__ == '__main__':
    main()
