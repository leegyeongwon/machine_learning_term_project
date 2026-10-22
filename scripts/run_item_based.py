"""MovieLens 1M 로딩, 학습/평가, 추천 및 결과 저장용 CLI."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.item_based import ItemBasedCF, split_by_user, validate_ratings


def load_movielens(data_dir):
    """:: 다중문자 구분자는 pandas Python 엔진, 제목은 Latin-1로 읽는다."""
    ratings = pd.read_csv(data_dir / 'ratings.dat', sep='::', engine='python',
                          names=['user_id', 'item_id', 'rating', 'timestamp'],
                          encoding='latin-1')
    movies = pd.read_csv(data_dir / 'movies.dat', sep='::', engine='python',
                         names=['item_id', 'title', 'genres'], encoding='latin-1')
    validate_ratings(ratings)
    if movies.isna().any().any() or movies.item_id.duplicated().any():
        raise ValueError('영화 메타데이터 결측 또는 중복 ID가 있습니다.')
    if not ratings.rating.between(1, 5).all():
        raise ValueError('MovieLens 평점 범위를 확인하세요.')
    if not ratings.item_id.isin(movies.item_id).all():
        raise ValueError('제목을 찾을 수 없는 영화 ID가 있습니다.')
    return ratings, movies


def evaluate(model, test):
    """전체 테스트를 평가하고 대체 예측까지 포함한 행별 근거를 남긴다."""
    details = [model.predict_with_details(u, i)
               for u, i in test[['user_id', 'item_id']].itertuples(index=False, name=None)]
    predictions = test[['user_id', 'item_id', 'rating']].reset_index(drop=True).copy()
    predictions[['predicted_rating', 'method', 'n_contributors']] = pd.DataFrame(details)
    error = predictions.predicted_rating - predictions.rating
    cf = predictions.method.eq('cf')
    metrics = {'n_test': len(test), 'n_evaluated': len(predictions),
               'n_cf': int(cf.sum()), 'n_fallback': int((~cf).sum()),
               'cf_coverage': float(cf.mean()), 'mae': float(error.abs().mean()),
               'rmse': float(np.sqrt(np.mean(error ** 2))),
               'cf_mae': float(error[cf].abs().mean()) if cf.any() else None,
               'cf_rmse': float(np.sqrt(np.mean(error[cf] ** 2))) if cf.any() else None,
               'unknown_test_items': int((~test.item_id.isin(model.item_ids_)).sum())}
    return metrics, predictions


def file_hash(path):
    """원본 데이터 재현성 확인을 위한 SHA-256을 계산한다."""
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    """K 비교는 동일 holdout 진단용이며 최적 K의 독립 최종 평가가 아니다."""
    # Windows 사용자 캐시 권한과 무관하게 프로젝트 내부에 그래프 캐시 저장.
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.venv/matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/raw/ml-1m')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/item_based')
    parser.add_argument('--k-values', type=int, nargs='+', default=[20, 30, 50])
    parser.add_argument('--recommend-k', type=int, default=30)
    parser.add_argument('--user-id', type=int, default=1)
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.recommend_k not in args.k_values:
        parser.error('--recommend-k는 --k-values에 포함되어야 합니다.')
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ratings, movies = load_movielens(args.data_dir)
    train, test = split_by_user(ratings, args.test_size, args.seed)
    summary = {'users': int(ratings.user_id.nunique()),
               'rated_items': int(ratings.item_id.nunique()), 'catalog_items': len(movies),
               'ratings': len(ratings), 'rating_distribution': ratings.rating.value_counts().sort_index().to_dict(),
               'ratings_missing': ratings.isna().sum().to_dict(),
               'movies_missing': movies.isna().sum().to_dict(),
               'n_train': len(train), 'n_test': len(test),
               'min_train_per_user': int(train.groupby('user_id').size().min()),
               'seed': args.seed, 'test_size': args.test_size,
               'split': 'per-user random holdout; floor(n * test_size), minimum one train',
               'recommend_k': args.recommend_k, 'recommend_user_id': args.user_id,
               'python': platform.python_version(),
               'versions': {name: importlib.metadata.version(name) for name in
                            ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'psutil']},
               'input_sha256': {name: file_hash(args.data_dir / name) for name in
                                ['ratings.dat', 'movies.dat', 'users.dat']}}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    # 동일 분할의 정확한 재생성을 위해 테스트 쌍과 원본 평점/타임스탬프 저장.
    test.to_csv(args.output_dir / 'test_split.csv', index=False)
    rows = []
    for k in args.k_values:
        tick = time.perf_counter()
        model = ItemBasedCF(n_neighbors=k).fit(train)
        fit_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        metrics, predictions = evaluate(model, test)
        metrics.update(k=k, fit_seconds=fit_seconds,
                       evaluation_seconds=time.perf_counter() - tick,
                       sparse_matrix_mib=sum(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
                                             for m in [model.user_item_, model.item_user_]) / 2**20,
                       process_rss_mib=psutil.Process().memory_info().rss / 2**20)
        rows.append(metrics)
        predictions.to_csv(args.output_dir / f'predictions_k{k}.csv', index=False)
        if k == args.recommend_k:
            recommendations = model.recommend(args.user_id, args.top_n).merge(
                movies[['item_id', 'title']], on='item_id', how='left')
            recommendations = recommendations[['item_id', 'title', 'predicted_rating', 'method', 'n_contributors']]
            recommendations.to_csv(args.output_dir / f'recommendations_user_{args.user_id}.csv',
                                   index=False, encoding='utf-8-sig')
            print(recommendations.to_string(index=False), flush=True)
        print(json.dumps(metrics), flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / 'metrics.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(results.k, results.mae, 'o-', label='MAE (all test)')
    ax.plot(results.k, results.rmse, 'o-', label='RMSE (all test)')
    ax.set(xlabel='K neighbors', ylabel='Rating error', title='MovieLens 1M: per-user 80/20 holdout')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / 'metrics.png', dpi=160)
    plt.close(fig)
    summary['total_seconds'] = time.perf_counter() - started
    summary['peak_working_set_mib'] = getattr(psutil.Process().memory_info(), 'peak_wset', 0) / 2**20
    (args.output_dir / 'run_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(results.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
