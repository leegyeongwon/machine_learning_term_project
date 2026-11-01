"""공통 NPZ 원점수 로딩 및 분할·누수 검증.

Z-score는 누수 검증용이며 모델 입력은 원점수이다.
Test는 분할·마스크 무결성 검사에만 사용한다.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from src.item_based import validate_ratings


def load_common_splits(npz_path):
    """원점수 배열 로딩 및 movie_id를 item_id로 매핑."""
    parts = {}
    with np.load(npz_path, allow_pickle=False) as data:
        for name in ['train', 'validation', 'test']:
            parts[name] = pd.DataFrame({
                'user_id': data[f'{name}_user_ids'],
                'item_id': data[f'{name}_movie_ids'],
                'rating': data[f'{name}_ratings']})
            validate_ratings(parts[name])
            if not parts[name].rating.between(1, 5).all():
                raise ValueError(f'{name}: 원점수 범위가 1~5가 아닙니다.')
    return parts


def source_splitter(source_path):
    """최상위 I/O 실행 없이 원문의 split_ratings 함수와 seed 반환."""
    tree = ast.parse(Path(source_path).read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'split_ratings')
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'RANDOM_STATE'
                              for target in node.targets))
    seed = ast.literal_eval(assignment.value)
    namespace = {'train_test_split': train_test_split, 'RANDOM_STATE': seed}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(source_path), 'exec'), namespace)
    return namespace['split_ratings'], seed


def restore_csr(data, prefix):
    """NPZ의 CSR 구성 요소로 희소 행렬 복원."""
    matrix = csr_matrix((data[f'{prefix}_data'], data[f'{prefix}_indices'],
                         data[f'{prefix}_indptr']), shape=tuple(data[f'{prefix}_shape']))
    matrix.check_format(full_check=True)
    return matrix


def validate_common_data(npz_path, raw_path, source_path, content_paths=()):
    """원본 평점, 분할, Train 통계·마스크·Z-score 검증.

    중복 기준은 평점 triple과 사용자-아이템 쌍이다.
    반환값은 분할별 원점수 DataFrame과 검증 결과이다.
    """
    parts = load_common_splits(npz_path)
    raw = pd.read_csv(raw_path, sep='::', engine='python', encoding='latin-1',
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])
    raw = raw.sort_values(['user_id', 'movie_id']).reset_index(drop=True)
    original = raw.rename(columns={'movie_id': 'item_id'})[['user_id', 'item_id', 'rating']]
    validate_ratings(original)
    combined = pd.concat(parts.values()).sort_values(['user_id', 'item_id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(combined, original)
    overlap = {}
    triple = ['user_id', 'item_id', 'rating']
    for a, b in [('train','validation'), ('train','test'), ('validation','test')]:
        overlap[f'{a}_{b}_triples'] = len(parts[a].merge(parts[b], on=triple))
        overlap[f'{a}_{b}_pairs'] = len(parts[a][triple[:2]].merge(parts[b][triple[:2]], on=triple[:2]))
    if any(overlap.values()):
        raise ValueError(f'분할 중복: {overlap}')
    splitter, seed = source_splitter(source_path)
    if seed != 42:
        raise ValueError('공통 seed가 42가 아닙니다.')
    # 원문 분할 함수로 저장 결과와 재현성 확인
    expected = splitter(raw)
    repeated = splitter(raw)
    for name, a, b in zip(parts, expected, repeated):
        pd.testing.assert_frame_equal(a, b)
        expected_frame = a.rename(columns={'movie_id':'item_id'})[triple]
        pd.testing.assert_frame_equal(parts[name], expected_frame)
    with np.load(npz_path, allow_pickle=False) as data:
        if int(data['random_state']) != seed:
            raise ValueError('NPZ의 seed가 원문과 다릅니다.')
        keys = list(data.files)
        users = pd.Index(data['user_ids'])
        items = pd.Index(data['movie_ids'])
        np.testing.assert_array_equal(users.to_numpy(), np.sort(original.user_id.unique()))
        np.testing.assert_array_equal(items.to_numpy(), np.sort(original.item_id.unique()))
        train = parts['train']
        for frame in parts.values():
            if not frame.user_id.isin(users).all() or not frame.item_id.isin(items).all():
                raise ValueError('잘못된 ID가 있습니다.')
        rows = users.get_indexer(train.user_id)
        cols = items.get_indexer(train.item_id)
        shape = (len(users), len(items))
        mask = restore_csr(data, 'mask')
        normalized = restore_csr(data, 'rating')
        expected_mask = csr_matrix((np.ones(len(train), dtype=np.int8), (rows, cols)), shape=shape)
        if mask.shape != shape or normalized.shape != shape or (mask != expected_mask).nnz:
            raise ValueError('Train 관측 마스크가 원본 Train과 다릅니다.')
        means = train.groupby('user_id').rating.mean().reindex(users).fillna(0).to_numpy()
        stds = train.groupby('user_id').rating.std(ddof=0).reindex(users).fillna(1).replace(0, 1).to_numpy()
        np.testing.assert_allclose(data['user_means'], means, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(data['user_stds'], stds, rtol=1e-12, atol=1e-12)
        expected_z = ((train.rating.to_numpy() - means[rows]) / stds[rows]).astype(np.float32)
        expected_matrix = csr_matrix((expected_z, (rows, cols)), shape=shape)
        difference = normalized - expected_matrix
        max_difference = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
        if not np.isfinite(normalized.data).all() or max_difference > 1e-6:
            raise ValueError('Train 기반 Z-score가 기대값과 다릅니다.')
        if (normalized - normalized.multiply(mask)).nnz:
            raise ValueError('Train 관측 이외의 Z-score 값이 있습니다.')
        for name in ['validation', 'test']:
            frame = parts[name]
            r, c = users.get_indexer(frame.user_id), items.get_indexer(frame.item_id)
            if np.asarray(mask[r, c]).any() or np.asarray(normalized[r, c]).any():
                raise ValueError(f'{name} 평점이 학습 행렬에 있습니다.')
        matrix_report = {'shape': list(shape), 'mask_nnz': int(mask.nnz),
                         'normalized_nnz': int(normalized.nnz),
                         'train_observed_zscore_zero_count': int((expected_z == 0).sum()),
                         'train_only_means_stds_verified': True,
                         'train_only_mask_verified': True, 'holdout_mask_and_matrix_zero': True,
                         'zscore_max_absolute_difference': max_difference}
    content_report = []
    for path in content_paths:
        path = Path(path)
        # 원점수 분할 배열이 있는 Content 파일만 비교
        with np.load(path, allow_pickle=False) as other:
            needed = {f'{name}_{key}' for name in parts for key in ['user_ids','movie_ids','ratings']}
            if not needed.issubset(other.files):
                content_report.append({'path': str(path), 'status':'not_comparable_missing_split_arrays'})
                continue
        other_parts = load_common_splits(path)
        for name in parts:
            # 행 순서와 무관하게 평점 triple 비교
            pd.testing.assert_frame_equal(parts[name].sort_values(triple).reset_index(drop=True),
                                          other_parts[name].sort_values(triple).reset_index(drop=True))
        content_report.append({'path': str(path), 'status':'all_split_triples_identical'})
    result = {'all_checks_passed':True, 'raw_count':len(raw), 'sum_split_count':len(combined),
              'split_method':'global shuffled two-stage 80/10/10 after user/movie sorting',
              'random_state':seed, 'source_split_exact_order_match':True, 'seed_reproducible':True,
              'original_triples_exactly_restored':True, 'overlap_counts':overlap,
              'npz_keys':keys, 'splits':{name:{'count':len(frame),'ratio':len(frame)/len(raw),
                     'users':int(frame.user_id.nunique()),'items':int(frame.item_id.nunique()),
                     'min_per_user':int(frame.groupby('user_id').size().min())} for name,frame in parts.items()},
              'users_without_train':int(len(set(original.user_id)-set(parts['train'].user_id))),
              'matrix_checks':matrix_report,
              'content_comparison':content_report or [{'status':'not_compared_no_local_content_npz'}]}
    return parts, result
