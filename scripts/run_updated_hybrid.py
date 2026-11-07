import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.hybrid import metrics, tune_weight


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
    out = ROOT / 'results/hybrid/updated_content'
    if out.exists():
        raise FileExistsError('Output directory already exists: ' + str(out))
    item_dir = ROOT / 'results/item_based/hybrid_update'
    content_dir = ROOT / 'results/content_based/cross_domain_eval_usermean_plot_ridge'
    selected_k = json.loads((ROOT / 'results/item_based/validation_a/selected_k.json').read_text())
    info = json.loads((ROOT / 'data/processed/hybrid_validation/split_info.json').read_text())
    if selected_k['selected_k'] != 50 or info['seed'] != 42:
        raise ValueError('Unexpected K or split seed')
    val = paired_predictions(item_dir / 'item_validation_b_predictions.csv',
                             content_dir / 'movie_val_b_predictions.csv', 1087587)
    test = paired_predictions(item_dir / 'item_test_predictions.csv',
                              content_dir / 'movie_test_predictions.csv', 2175173)
    keys = ['user_id', 'item_id']
    if pd.MultiIndex.from_frame(val[keys]).intersection(pd.MultiIndex.from_frame(test[keys])).size:
        raise ValueError('Validation B and Test overlap')
    alpha, grid = tune_weight(val.rating_item, val.predicted_rating_item, val.predicted_rating_content)
    best = grid.loc[grid.alpha.eq(alpha)].iloc[0]
    selected = {
        'item_weight': alpha, 'content_weight': 1 - alpha,
        'validation_b_mae': float(best.mae), 'validation_b_rmse': float(best.rmse),
        'item_k': selected_k['selected_k'], 'content_model': 'Nonnegative Ridge trained on Validation A',
        'split_seed': info['seed'], 'validation_b_count': len(val), 'test_count': len(test),
        'selection': 'Validation B MAE, then RMSE, then smaller item weight',
        'rating_comparison_atol': 1e-6,
        'content_predictions_directory': content_dir.relative_to(ROOT).as_posix(),
    }
    predictions = {
        'Item-based': test.predicted_rating_item,
        'Content-based': test.predicted_rating_content,
        'Hybrid': alpha * test.predicted_rating_item + (1 - alpha) * test.predicted_rating_content,
    }
    rows = [{'model': model, **metrics(test.rating_item, prediction), 'count': len(test)}
            for model, prediction in predictions.items()]
    hybrid = rows[-1]
    old_mae, old_rmse = .633949, .848977
    report = '\n'.join([
        '# 새로운 Content-based 모델을 적용한 Hybrid 결과', '',
        '기존 Hybrid는 SharedContentCF와 기존 Validation 전체를 사용하였다. 이번에는 팀원의 비음수 Ridge 모델을 사용하였다. Validation A는 Content-based 학습과 Item-based K 선택에 사용하고, Validation B에서 두 예측을 결합할 비중을 선택하였다. 추가 모델 학습은 수행하지 않았다.', '',
        f'Validation B에서 0~1을 0.01 간격으로 비교한 결과 Item-based 비중은 {alpha:.2f}, Content-based 비중은 {1-alpha:.2f}로 선택되었다. MAE를 우선하고 동률이면 RMSE, 작은 Item 비중 순으로 선택하였다.',
        f'선택된 조합의 Validation B MAE는 {best.mae:.6f}, RMSE는 {best.rmse:.6f}이다. 이 비중을 고정하여 Test를 평가하였다.', '',
        '| 모델 | 영화 Test MAE | 영화 Test RMSE |', '|---|---:|---:|',
        f'| 기존 Hybrid | {old_mae:.6f} | {old_rmse:.6f} |',
        *[f"| {r['model']} | {r['mae']:.6f} | {r['rmse']:.6f} |" for r in rows], '',
        f"새 Hybrid와 기존 Hybrid의 오차 차이(새 결과−기존 결과)는 MAE {hybrid['mae']-old_mae:+.6f}, RMSE {hybrid['rmse']-old_rmse:+.6f}이다. 차이는 제공된 기존 결과의 소수점 여섯 자리 값을 기준으로 계산하였다. 음수이면 오차 감소, 양수이면 오차 증가이다.", '',
        '두 예측의 결합 비중은 Validation B에서 관측한 오차에 따라 결정되었다. 기존 실험과는 Content-based 구현과 Validation 사용 방식이 함께 달라졌으므로, 성능 차이를 어느 한 변경의 효과로 단정할 수 없다. 원인을 분리하는 추가 실험은 수행하지 않았다.', '',
        f'Validation B {len(val):,}행과 Test {len(test):,}행에서 모델 간 ID와 정답을 대조하였으며, 두 분할 사이의 사용자·영화 쌍 중복은 없었다. 정답 비교에는 절대 허용 오차 1e-6을 적용하고 지표 계산에는 Item-based CSV의 정답을 사용하였다. Test는 가중치 선택에 사용하지 않았다.', '',
        '이번 실험은 제공된 영화 예측 CSV의 결합 평가이다. Content-based 학습 과정 자체를 재검증하거나 도서 도메인의 성능을 평가한 결과는 아니다.', '',
    ])
    out.mkdir(parents=True)
    grid_path = out / 'hybrid_validation_b_grid.csv'
    metrics_path = out / 'movie_test_metrics.csv'
    grid.to_csv(grid_path, index=False)
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    json_path = out / 'selected_hybrid_weight.json'
    json_path.write_text(json.dumps(selected, indent=2) + '\n', encoding='utf-8')
    (out / 'hybrid_comparison.md').write_text(report, encoding='utf-8')
    if not pd.read_csv(grid_path, float_precision='round_trip').equals(grid):
        raise ValueError('Saved grid differs')
    if pd.read_csv(metrics_path, float_precision='round_trip').to_dict('records') != rows:
        raise ValueError('Saved metrics differ')
    if json.loads(json_path.read_text()) != selected:
        raise ValueError('Saved selection differs')
    print(json.dumps({'selected': selected, 'test_metrics': rows, 'verification': 'passed'}), flush=True)


if __name__ == '__main__':
    main()
