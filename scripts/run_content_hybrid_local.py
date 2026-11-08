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
from src.hybrid import metrics, tune_weight
from scripts.run_updated_hybrid import paired_predictions
from scripts.split_hybrid_validation import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--team-source', type=Path)
    parser.add_argument('--content-output', type=Path, default=ROOT / 'results/content_based/hybrid_local_rebuild')
    parser.add_argument('--hybrid-output', type=Path, default=ROOT / 'results/hybrid/rebuilt_content')
    args = parser.parse_args()
    data = ROOT / 'data/processed'
    split = data / 'hybrid_validation'
    content_out = args.content_output.resolve()
    hybrid_out = args.hybrid_output.resolve()
    if content_out.exists() or hybrid_out.exists():
        raise FileExistsError('A rebuild output directory already exists')
    expected = {
        data / 'movie_item_user_data.npz': '083b64d1de94f09405e08fad538ecad3ae4396e86ab481b08f7d35c44a344481',
        split / 'validation_split_indices.npz': 'e8fb6ad98d0b5dc2f44745bd7b3d47dac7216c72c8fa8a0ef931b57a054ab2b9',
    }
    inputs = list(expected) + [data / 'movie_user_profiles.npz', data / 'movie_content.npz',
                              split / 'validation_a.npz', split / 'validation_b.npz', split / 'split_info.json']
    item_dir = ROOT / 'results/item_based/hybrid_update'
    inputs += [item_dir / 'item_validation_b_predictions.csv', item_dir / 'item_test_predictions.csv']
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs}
    for path, value in expected.items():
        if hashes[path.relative_to(ROOT).as_posix()] != value:
            raise ValueError('Unexpected input hash: ' + str(path))
    if args.team_source is not None:
        source = args.team_source.read_text(encoding='utf-8')
    else:
        source = subprocess.check_output(['git', 'show', '7b2eb0c306e64a8aff6935a99b03109ff2d40b07'], cwd=ROOT).decode('utf-8')
    if hashlib.sha256(source.encode()).hexdigest() != '4cf085df20da03b755f6b3ecc583cafd774c378f12507cbfacce10ce8aea7a42':
        raise ValueError('Team source differs from the version used in this experiment')
    team = types.ModuleType('team_content_model')
    team.__file__ = str(ROOT / 'scripts/cross_domain_eval_usermean_plot_ridge.py')
    exec(compile(source, team.__file__, 'exec'), team.__dict__)
    with np.load(split / 'validation_split_indices.npz', allow_pickle=False) as z:
        a, b = z['a_indices'], z['b_indices']
    with np.load(data / 'movie_item_user_data.npz', allow_pickle=False) as z:
        validation = {s: z['validation_' + s] for s in ['user_ids', 'movie_ids', 'ratings']}
        test = {s: z['test_' + s] for s in ['user_ids', 'movie_ids', 'ratings']}
        with np.load(data / 'movie_user_profiles.npz', allow_pickle=False) as profiles:
            for s in ['user_ids', 'movie_ids', 'ratings']:
                if not np.array_equal(z['train_' + s], profiles['train_' + s]):
                    raise ValueError('Profile Train differs: ' + s)
    order = np.random.default_rng(42).permutation(len(validation['ratings']))
    if not np.array_equal(a, order[:len(a)]) or not np.array_equal(b, order[len(a):]):
        raise ValueError('Unexpected A/B indices')
    for part, indices in [('a', a), ('b', b)]:
        with np.load(split / ('validation_' + part + '.npz'), allow_pickle=False) as z:
            if not np.array_equal(z['validation_indices'], indices):
                raise ValueError('Split indices differ')
            for s, values in validation.items():
                if not np.array_equal(z['validation_' + s], values[indices]):
                    raise ValueError('Split rows differ')
    vk = pd.MultiIndex.from_arrays([validation['user_ids'], validation['movie_ids']])
    tk = pd.MultiIndex.from_arrays([test['user_ids'], test['movie_ids']])
    if not vk.is_unique or not tk.is_unique or vk.intersection(tk).size:
        raise ValueError('Validation/Test overlap or duplicates')
    movie = team.load_domain('movie', 'movie_ids')
    if movie['user_plot'].shape != (len(movie['user_ids']), 1024):
        raise ValueError('User plot dimension differs')
    if movie['item_plot'].shape != (len(movie['item_ids']), 1024):
        raise ValueError('Item plot dimension differs')
    means = team.train_user_means(movie)
    u, i, y, _, _ = team.load_hybrid_val_part(movie, 'a')
    if len(y) != 1087586:
        raise ValueError('Training count differs')
    print(json.dumps({'stage': 'content_fit', 'rows': len(y), 'ridge_alpha': 10,
                      'positive_coefficients': True}), flush=True)
    model = team.fit_ridge_batched(movie, u, i, y - means[u], alpha=10.)
    if not np.isfinite(model.coef_).all() or (model.coef_ < 0).any() or not np.isfinite(model.intercept_):
        raise ValueError('Invalid Ridge solution')
    content_out.mkdir(parents=True)
    content_metrics = {}
    for label in ['val_b', 'test']:
        if label == 'val_b':
            u, i, y, users, items = team.load_hybrid_val_part(movie, 'b')
            expected_rows = {s: values[b] for s, values in validation.items()}
        else:
            users, items, y = test['user_ids'], test['movie_ids'], test['ratings'].astype(np.float32)
            u = np.searchsorted(movie['user_ids'], users)
            i = np.searchsorted(movie['item_ids'], items)
            expected_rows = test
        for actual, s in [(users, 'user_ids'), (items, 'movie_ids'), (y, 'ratings')]:
            if not np.array_equal(actual, expected_rows[s]):
                raise ValueError('Prediction target differs: ' + label)
        predicted = means[u] + team.predict_residual(model, movie, u, i)
        path = content_out / ('movie_' + label + '_predictions.csv')
        team.save_predictions(path, users, items, y, predicted)
        saved = pd.read_csv(path, float_precision='round_trip')
        for column, actual in [('user_id', users), ('item_id', items), ('rating', y),
                               ('predicted_rating', np.clip(predicted, .5, 5.))]:
            if not np.array_equal(saved[column], actual):
                raise ValueError('Saved prediction mismatch: ' + column)
        content_metrics[label] = metrics(y, saved.predicted_rating)
        print(json.dumps({'stage': label, 'count': len(saved), **content_metrics[label]}), flush=True)
    val = paired_predictions(item_dir / 'item_validation_b_predictions.csv',
                             content_out / 'movie_val_b_predictions.csv', 1087587)
    paired_test = paired_predictions(item_dir / 'item_test_predictions.csv',
                                     content_out / 'movie_test_predictions.csv', 2175173)
    if pd.MultiIndex.from_frame(val[['user_id', 'item_id']]).intersection(
            pd.MultiIndex.from_frame(paired_test[['user_id', 'item_id']])).size:
        raise ValueError('Prediction splits overlap')
    alpha, grid = tune_weight(val.rating_item, val.predicted_rating_item, val.predicted_rating_content)
    best = grid.loc[grid.alpha.eq(alpha)].iloc[0]
    predictions = {'Item-based': paired_test.predicted_rating_item,
                   'Content-based': paired_test.predicted_rating_content,
                   'Hybrid': alpha * paired_test.predicted_rating_item + (1-alpha) * paired_test.predicted_rating_content}
    rows = [{'model': name, **metrics(paired_test.rating_item, values), 'count': len(paired_test)}
            for name, values in predictions.items()]
    selected = {'item_weight': alpha, 'content_weight': 1-alpha, 'item_k': 50,
                'content_model': 'Nonnegative Ridge; alpha=10; Validation A residual training',
                'validation_b_mae': float(best.mae), 'validation_b_rmse': float(best.rmse),
                'validation_b_count': len(val), 'test_count': len(paired_test), 'split_seed': 42}
    hybrid_out.mkdir(parents=True)
    grid.to_csv(hybrid_out / 'hybrid_validation_b_grid.csv', index=False)
    pd.DataFrame(rows).to_csv(hybrid_out / 'movie_test_metrics.csv', index=False)
    (hybrid_out / 'selected_hybrid_weight.json').write_text(json.dumps(selected, indent=2)+'\n', encoding='utf-8')
    report = '\n'.join(['# Content-based 재학습 후 Hybrid 결과', '',
        '기존 Hybrid는 SharedContentCF와 전체 Validation을 사용하였다. 이번에는 팀원의 1027개 피처와 비음수 Ridge 구현을 그대로 사용하고 공통 Validation A 1,087,586행의 사용자 평균 대비 잔차를 학습하였다. 사용자 평균과 선호 프로필은 Train을 기준으로 사용하였다.', '',
        f'Validation B에서 101개 가중치를 비교하여 Item-based {alpha:.2f}, Content-based {1-alpha:.2f}를 선택하였다. MAE 우선, RMSE, 작은 Item 비중 순으로 선택했으며 Test로 다시 조정하지 않았다.',
        f'선택된 조합의 Validation B MAE는 {best.mae:.6f}, RMSE는 {best.rmse:.6f}이다.', '',
        '| 모델 | Test MAE | Test RMSE |', '|---|---:|---:|', '| 기존 Hybrid | 0.633949 | 0.848977 |',
        *[f"| {r['model']} | {r['mae']:.6f} | {r['rmse']:.6f} |" for r in rows], '',
        f"기존 Hybrid 대비 새 Hybrid의 오차 차이는 MAE {rows[-1]['mae']-.633949:+.6f}, RMSE {rows[-1]['rmse']-.848977:+.6f}이다. 기존 수치는 소수점 여섯 자리 기준이며 음수는 오차 감소를 뜻한다.", '',
        '결합 비중은 Validation B의 오차를 기준으로 결정되었다. Content-based 구현과 Validation 사용 방식이 동시에 달라져 성능 차이를 한 가지 원인으로 단정할 수 없다. 원인을 분리하는 실험과 도서 평가는 수행하지 않았다.', '',
        '프로필 생성 코드는 Train 평점만 참조한다. 저장된 장르·수치 프로필 전체와 줄거리 프로필 128명을 Train 계산 결과와 비교하였다. 줄거리 프로필 전체와 과거 파일 생성 이력까지 입증한 것은 아니다. 콘텐츠 전처리의 기존 변환 기준을 유지했으며 전처리 방식을 새로 검증하거나 학습하지 않았다.', ''])
    (hybrid_out / 'hybrid_comparison.md').write_text(report, encoding='utf-8')
    for path in inputs:
        if sha256(path) != hashes[path.relative_to(ROOT).as_posix()]:
            raise ValueError('Input changed: ' + str(path))
    if not pd.read_csv(hybrid_out/'hybrid_validation_b_grid.csv', float_precision='round_trip').equals(grid):
        raise ValueError('Saved grid differs')
    if pd.read_csv(hybrid_out/'movie_test_metrics.csv', float_precision='round_trip').to_dict('records') != rows:
        raise ValueError('Saved metrics differ')
    if json.loads((hybrid_out/'selected_hybrid_weight.json').read_text()) != selected:
        raise ValueError('Saved selection differs')
    provenance = {'input_sha256': hashes, 'team_source_git_blob': '7b2eb0c306e64a8aff6935a99b03109ff2d40b07',
                  'team_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                  'reused_functions': ['load_domain', 'load_hybrid_val_part', 'train_user_means', 'pair_features',
                                       'fit_ridge_batched', 'predict_residual', 'save_predictions'],
                  'training_rows': 1087586, 'training_split': 'Common Validation A', 'ridge_alpha': 10,
                  'nonnegative_coefficients': True, 'intercept_constrained': False,
                  'profile_checks': {'train_records_exact_match': True,
                                     'preference_profile_check_performed': False,
                                     'preference_profile_check_result': None,
                                     'limitation': 'Genre, numeric and plot profiles are not reconstructed or compared in this run; historical manual check results are not included.'},
                  'content_metrics': content_metrics, 'selected': selected, 'test_metrics': rows,
                  'prediction_pair_validation': 'passed', 'output_round_trip': 'passed', 'inputs_unchanged': True}
    (content_out/'rebuild_info.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'selected': selected, 'test_metrics': rows, 'verification': 'passed'}), flush=True)


if __name__ == '__main__':
    main()
