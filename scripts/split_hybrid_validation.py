import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/processed/movie_item_user_data.npz'
OUTPUT = ROOT / 'data/processed/hybrid_validation'
SEED = 42


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def split_counts(path):
    counts = {}
    # Train/Test 배열은 읽지 않고 NPY 헤더에서 행 수만 확인
    with ZipFile(path) as archive:
        for split in ('train', 'validation', 'test'):
            with archive.open(split + '_ratings.npy') as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, _ = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unsupported NPY header: ' + str(version))
                counts[split] = shape[0]
    return counts


def save_arrays(path, arrays):
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays):
                raise ValueError('Existing output has different keys: ' + str(path))
            for key, values in arrays.items():
                if saved[key].dtype != values.dtype or not np.array_equal(saved[key], values):
                    raise ValueError('Existing output differs: ' + str(path))
        return
    np.savez_compressed(path, **arrays)


def main():
    source_hash = sha256(SOURCE)
    counts = split_counts(SOURCE)
    keys = ('validation_user_ids', 'validation_movie_ids', 'validation_ratings')
    with np.load(SOURCE, allow_pickle=False) as source:
        validation = {key: source[key] for key in keys}
    n = counts['validation']
    if any(values.shape != (n,) for values in validation.values()):
        raise ValueError('Validation array lengths differ')

    order = np.random.default_rng(SEED).permutation(n)
    a_indices, b_indices = order[:n // 2], order[n // 2:]
    if not np.array_equal(np.sort(np.concatenate([a_indices, b_indices])), np.arange(n)):
        raise ValueError('Validation indices are missing or duplicated')

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, indices in [('a', a_indices), ('b', b_indices)]:
        arrays = {key: values[indices] for key, values in validation.items()}
        arrays['validation_indices'] = indices
        path = OUTPUT / ('validation_' + name + '.npz')
        save_arrays(path, arrays)
        with np.load(path, allow_pickle=False) as saved:
            for key in keys:
                if not np.array_equal(saved[key], validation[key][saved['validation_indices']]):
                    raise ValueError('Saved rows differ from source: ' + key)

    save_arrays(OUTPUT / 'validation_split_indices.npz',
                {'a_indices': a_indices, 'b_indices': b_indices})
    if sha256(SOURCE) != source_hash:
        raise ValueError('Source file changed')
    info = {
        'source_file': SOURCE.relative_to(ROOT).as_posix(),
        'source_sha256': source_hash,
        'seed': SEED,
        'method': 'numpy.random.default_rng(42).permutation; A = first n//2 rows, B = remaining rows',
        'index_basis': 'Zero-based rows in the original Validation arrays',
        'counts': {'train': counts['train'], 'validation': n,
                   'validation_a': len(a_indices), 'validation_b': len(b_indices),
                   'test': counts['test']},
        'purpose': {'validation_a': 'Content-based regression training',
                    'validation_b': 'Hybrid weight selection'},
    }
    info_path = OUTPUT / 'split_info.json'
    if info_path.exists():
        if json.loads(info_path.read_text(encoding='utf-8')) != info:
            raise ValueError('Existing split_info.json differs')
    else:
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'counts': info['counts'], 'seed': SEED, 'verification': 'passed'}))


if __name__ == '__main__':
    main()
