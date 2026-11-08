import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from src.item_based import ItemBasedCF
from src.hybrid import metrics
from src.transfer_data import load_splits, validate_splits
from scripts.run_full_recsys_pipeline import scored
from scripts.split_hybrid_validation import sha256


def aligned_positions(ids, values):
    positions = pd.Index(ids).get_indexer(values)
    if (positions < 0).any():
        raise ValueError('User or item IDs missing from content')
    return positions


def validate_profiles(domain, train, content_path):
    rows = aligned_positions(domain['user_ids'], train.user_id)
    cols = aligned_positions(domain['item_ids'], train.item_id)
    signed = csr_matrix((train.rating.to_numpy(dtype=np.float32)-3., (rows, cols)),
                        shape=(len(domain['user_ids']), len(domain['item_ids'])))
    positive = signed.maximum(0).tocsr()
    expected_genres = normalize(positive @ domain['item_genres'])
    delta = (expected_genres-domain['user_genres']).tocsr()
    genre_error = float(np.abs(delta.data).max()) if delta.nnz else 0.
    numeric = np.zeros_like(domain['user_numeric'])
    with np.load(content_path, allow_pickle=False) as z:
        length_missing = z['runtime_missing' if domain['item_key']=='movie_ids' else 'pages_missing']
        year_missing = z['year_missing']
        plot_valid = z['plot_valid_mask'].astype(bool)
    for j, missing in enumerate([length_missing, year_missing]):
        observed = ~missing.astype(bool)
        values = domain['item_numeric'][:, j].copy()
        values[~observed] = 0
        numerator = np.asarray(positive @ values).ravel()
        denominator = np.asarray(positive @ observed.astype(np.float32)).ravel()
        np.divide(numerator, denominator, out=numeric[:, j], where=denominator>0)
        numeric[denominator<=0, j] = .5
    numeric_error = float(np.abs(numeric-domain['user_numeric']).max())
    sample = np.sort(np.random.default_rng(42).choice(len(domain['user_ids']),
                                                    min(128, len(domain['user_ids'])), replace=False))
    expected_plot = normalize(signed[sample][:, plot_valid] @ domain['item_plot'][plot_valid])
    plot_error = float(np.abs(expected_plot-domain['user_plot'][sample]).max())
    if max(genre_error, numeric_error, plot_error) > 1e-5:
        raise ValueError('Stored profiles differ from Train-only calculation: ' +
                         str((genre_error, numeric_error, plot_error)))
    return {'genre_all_users_max_error': genre_error, 'numeric_all_users_max_error': numeric_error,
            'plot_sample_users': len(sample), 'plot_sample_max_error': plot_error,
            'limitation': 'Plot profiles sampled; original generation history not proven'}


def combine_predictions(item, content, alpha):
    columns = ['user_id', 'item_id', 'rating']
    if not item[columns].equals(content[columns]):
        raise ValueError('Prediction rows or actual ratings differ')
    if item.duplicated(['user_id', 'item_id']).any():
        raise ValueError('Duplicate prediction pairs')
    for frame in [item, content]:
        if not np.isfinite(frame.predicted_rating).all() or not frame.predicted_rating.between(1, 5).all():
            raise ValueError('Predictions outside Book rating range')
    result = item[columns].copy()
    result['predicted_rating'] = alpha*item.predicted_rating + (1-alpha)*content.predicted_rating
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--team-source', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/hybrid/rebuilt_content_book')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError('Choose a new output directory: ' + str(out))
    data = ROOT/'data/processed'
    movie_info = json.loads((ROOT/'results/content_based/hybrid_local_rebuild/rebuild_info.json').read_text())
    weight_path = ROOT/'results/hybrid/rebuilt_content/selected_hybrid_weight.json'
    selected = json.loads(weight_path.read_text())
    if selected['item_k'] != 50 or selected['item_weight'] != .98 or not np.isclose(selected['content_weight'], .02):
        raise ValueError('Movie-selected settings differ')
    paths = [data/(d+'_'+s+'.npz') for d in ['movie', 'book']
             for s in ['item_user_data', 'content', 'user_profiles']]
    split = data/'hybrid_validation'
    paths += [split/n for n in ['validation_a.npz', 'validation_b.npz', 'validation_split_indices.npz', 'split_info.json']]
    paths += [weight_path, ROOT/'results/hybrid/rebuilt_content/movie_test_metrics.csv']
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}
    for name, value in movie_info['input_sha256'].items():
        if name in hashes and hashes[name] != value:
            raise ValueError('Movie inputs differ from the verified experiment: ' + name)
    source = (args.team_source.read_text(encoding='utf-8') if args.team_source else
              subprocess.check_output(['git', 'show', movie_info['team_source_git_blob']], cwd=ROOT).decode('utf-8'))
    if hashlib.sha256(source.encode()).hexdigest() != movie_info['team_source_sha256']:
        raise ValueError('Team source version differs')
    team = types.ModuleType('team_book_eval')
    team.__file__ = str(ROOT/'scripts/cross_domain_eval_usermean_plot_ridge.py')
    exec(compile(source, team.__file__, 'exec'), team.__dict__)
    domains = {d: team.load_domain(d, d+'_ids') for d in ['movie', 'book']}
    checks = {}; feature_meta = {}
    for d, domain in domains.items():
        for s in ['item_plot', 'user_plot', 'item_numeric', 'user_numeric']:
            if not np.isfinite(domain[s]).all():
                raise ValueError('Nonfinite features: '+d+' '+s)
        if domain['item_plot'].shape[1] != 1024 or domain['user_plot'].shape[1] != 1024:
            raise ValueError('Plot embedding dimension differs')
        if not pd.Index(domain['user_ids']).is_unique or not pd.Index(domain['item_ids']).is_unique:
            raise ValueError('Duplicate content/profile IDs')
        parts = load_splits(data/(d+'_item_user_data.npz'), d, validate=False)
        checks[d] = {'splits': validate_splits(parts)}
        for s, frame in parts.items():
            for column, suffix in [('user_id', 'user_ids'), ('item_id', d+'_ids'), ('rating', 'ratings')]:
                if not np.array_equal(frame[column], domain['profile'][s+'_'+suffix]):
                    raise ValueError('Stored profile split differs: '+d+' '+s)
        checks[d]['profiles'] = validate_profiles(domain, parts['train'], data/(d+'_content.npz'))
        with np.load(data/(d+'_content.npz'), allow_pickle=False) as z, np.load(data/(d+'_user_profiles.npz'), allow_pickle=False) as pr:
            if not np.array_equal(z['genre_names'], pr['genre_names']):
                raise ValueError('Item/user genre axes differ')
            feature_meta[d] = {'genre_names': z['genre_names'].tolist(),
                               'numeric_names': z['numeric_feature_names'].tolist(),
                               'year_bounds': [float(z['year_min']), float(z['year_max'])]}
        if d == 'book':
            books = parts
        print(json.dumps({'stage': 'validated_'+d, 'profile_checks': checks[d]['profiles']}), flush=True)
    if feature_meta['movie']['genre_names'] != feature_meta['book']['genre_names']:
        raise ValueError('Movie/book genre axes differ')
    if feature_meta['movie']['year_bounds'] != feature_meta['book']['year_bounds']:
        raise ValueError('Movie/book year normalization differs')
    if feature_meta['movie']['numeric_names'] != ['runtime', 'release_year'] or feature_meta['book']['numeric_names'] != ['pages', 'publish_year']:
        raise ValueError('Unexpected numeric feature order')
    with np.load(split/'validation_split_indices.npz', allow_pickle=False) as z:
        a, b = z['a_indices'], z['b_indices']
    with np.load(data/'movie_item_user_data.npz', allow_pickle=False) as z:
        order = np.random.default_rng(42).permutation(len(z['validation_ratings']))
        if not np.array_equal(a, order[:len(a)]) or not np.array_equal(b, order[len(a):]):
            raise ValueError('Movie A/B indices differ')
        with np.load(split/'validation_a.npz', allow_pickle=False) as saved:
            for suffix in ['user_ids', 'movie_ids', 'ratings']:
                if not np.array_equal(saved['validation_'+suffix], z['validation_'+suffix][a]):
                    raise ValueError('Movie Validation A differs')
    movie = domains['movie']; book = domains['book']
    movie_means = team.train_user_means(movie)
    u, i, y, _, _ = team.load_hybrid_val_part(movie, 'a')
    if len(y) != 1087586:
        raise ValueError('Movie Ridge fit count differs')
    print('Fitting movie Validation A Ridge (alpha=10, nonnegative coefficients)', flush=True)
    model = team.fit_ridge_batched(movie, u, i, y-movie_means[u], alpha=10.)
    if model.coef_.shape != (1027,) or not np.isfinite(model.coef_).all() or (model.coef_<0).any() or not np.isfinite(model.intercept_):
        raise ValueError('Invalid movie Ridge solution')
    coef = model.coef_.copy(); intercept = model.intercept_
    print('Fitting Book Train Item-based (K=50)', flush=True)
    item_model = ItemBasedCF(selected['item_k']).fit(books['train'])
    test = books['test']
    if not test.rating.between(1,5).all() or item_model.rating_range_ != (1.,5.):
        raise ValueError('Unexpected Book rating range')
    item = scored(item_model, test)
    means = team.train_user_means(book)
    users = aligned_positions(book['user_ids'], test.user_id)
    items = aligned_positions(book['item_ids'], test.item_id)
    raw = means[users] + team.predict_residual(model, book, users, items)
    content = test.copy();content['predicted_rating'] = np.clip(raw, 1., 5.)
    hybrid = combine_predictions(item, content, selected['item_weight'])
    if not np.array_equal(model.coef_, coef) or model.intercept_ != intercept:
        raise ValueError('Movie coefficients changed during transfer')
    out.mkdir(parents=True)
    rows = []
    for name, frame in [('Item-based', item), ('Content-based', content), ('Hybrid', hybrid)]:
        path = out/(name.lower().replace('-', '_')+'_book_test_predictions.csv')
        frame.to_csv(path, index=False)
        saved = pd.read_csv(path, float_precision='round_trip', dtype={'rating': frame.rating.dtype})
        for column in frame:
            if not np.array_equal(saved[column], frame[column]):
                raise ValueError('Prediction CSV round-trip mismatch: '+column)
        measured = metrics(frame.rating, frame.predicted_rating)
        err = saved.predicted_rating.to_numpy()-saved.rating.to_numpy(dtype=float)
        independent = {'mae': float(np.sum(np.abs(err))/len(err)),
                       'rmse': float(np.sqrt(np.dot(err,err)/len(err)))}
        if not np.allclose(list(measured.values()), list(independent.values()), rtol=0, atol=1e-12):
            raise ValueError('Independent metric verification failed')
        rows.append({'model': name, **measured, 'count': len(frame)})
        print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(out/'book_test_metrics.csv', index=False)
    if pd.read_csv(out/'book_test_metrics.csv', float_precision='round_trip').to_dict('records') != rows:
        raise ValueError('Metric CSV round-trip mismatch')
    reference = pd.read_csv(ROOT/'docs/final_results/final_metrics.csv', float_precision='round_trip')
    previous_item = reference.loc[reference.domain.eq('book_support_adapted') & reference.model.eq('Item-based')].iloc[0]
    if not np.allclose([rows[0]['mae'],rows[0]['rmse']], [previous_item.mae,previous_item.rmse], rtol=0, atol=1e-12):
        raise ValueError('Item-based differs from the previous identical Book experiment')
    for path in paths:
        if sha256(path) != hashes[path.relative_to(ROOT).as_posix()]:
            raise ValueError('Input changed: '+str(path))
    info = {'evaluation': 'Support-adapted; not zero-shot', 'input_sha256': hashes,
            'team_source_git_blob': movie_info['team_source_git_blob'], 'team_source_sha256': movie_info['team_source_sha256'],
            'movie_training_rows': len(y), 'ridge_alpha': 10, 'feature_count': 1027,
            'feature_order': ['genre_cos']+['plot_product_'+str(j) for j in range(1024)]+['length_closeness','year_closeness'],
            'nonnegative_coefficients': True, 'intercept_constrained': False,
            'movie_coefficients': coef.tolist(), 'movie_intercept': intercept,
            'movie_coefficients_unchanged': True, 'item_k': 50,
            'item_weight': selected['item_weight'], 'content_weight': 1-selected['item_weight'],
            'data_checks': checks, 'feature_metadata': feature_meta,
            'embedding_evidence': 'Both preprocessing sources specify BAAI/bge-m3 dense vectors; NPZ has no model weight hash or embedding-axis provenance',
            'length_interpretation': 'Within-domain normalized length preference closeness; runtime minutes and book pages are not the same physical quantity',
            'genre_interpretation': 'Shared ordered axes; user/item similarity calculated within each domain, no movie-to-book genre equivalence asserted',
            'year_interpretation': 'Existing shared movie/book min-max normalization retained, including anomalous years',
            'content_clipping': [1.,5.], 'team_original_save_clipping': [.5,5.],
            'raw_content_below_one': int((raw<1).sum()), 'raw_content_above_five': int((raw>5).sum()),
            'prediction_checks': 'passed: exact rows/ratings, unique pairs, finite/range, round-trip and independent metrics',
            'previous_item_metrics_reproduced': True, 'metrics': rows, 'inputs_unchanged': True}
    (out/'book_eval_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Book evaluation and verification completed', flush=True)


if __name__ == '__main__':
    main()
