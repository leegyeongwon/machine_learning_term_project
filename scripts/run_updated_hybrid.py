from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def load_predictions(path, count):
    frame = pd.read_csv(path, float_precision='round_trip')
    columns = ['user_id', 'item_id', 'rating', 'predicted_rating']
    if not set(columns).issubset(frame.columns) or len(frame) != count:
        raise ValueError('Missing columns or unexpected count: ' + str(path))
    frame = frame[columns]
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Missing or nonfinite values: ' + str(path))
    for column in ['user_id', 'item_id']:
        if not pd.api.types.is_integer_dtype(frame[column]) or (frame[column] < 0).any():
            raise ValueError('Invalid IDs: ' + str(path))
    if frame.duplicated(['user_id', 'item_id']).any():
        raise ValueError('Duplicate pairs: ' + str(path))
    if not frame.rating.between(.5, 5).all() or not frame.predicted_rating.between(.5, 5).all():
        raise ValueError('Ratings outside allowed range: ' + str(path))
    return frame


def paired_predictions(item_path, content_path, count):
    item = load_predictions(item_path, count)
    content = load_predictions(content_path, count)
    paired = item.merge(content, on=['user_id', 'item_id'], how='outer', sort=False,
                        suffixes=('_item', '_content'), indicator=True, validate='one_to_one')
    if len(paired) != count or not paired['_merge'].eq('both').all():
        raise ValueError('Item and Content prediction pairs differ')
    if not np.allclose(paired.rating_item, paired.rating_content, rtol=0, atol=1e-6):
        raise ValueError('Item and Content actual ratings differ (atol=1e-6)')
    return paired


def main():
    raise SystemExit(
        'This legacy CLI is disabled because its Content-based CSVs use an incompatible split. '
        'Use scripts/run_content_hybrid_local.py for the final experiment. '
        'load_predictions() and paired_predictions() remain available for import.'
    )

if __name__ == '__main__':
    main()
