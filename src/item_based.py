"""희소 행렬 기반 Item-based CF. 데이터셋의 제목/장르 로직과 독립적이다."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def validate_ratings(ratings):
    """공통 입력 컬럼, 양수 유한 평점, 중복 사용자-아이템을 검증한다."""
    required = ['user_id', 'item_id', 'rating']
    if not set(required).issubset(ratings.columns) or ratings.empty:
        raise ValueError('비어 있지 않은 user_id, item_id, rating 데이터가 필요합니다.')
    if ratings[required].isna().any().any():
        raise ValueError('결측값은 허용하지 않습니다.')
    values = ratings.rating.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('평점은 유한한 양수여야 합니다. 0은 미평가입니다.')
    if ratings.duplicated(['user_id', 'item_id']).any():
        raise ValueError('중복 사용자-아이템 평점은 먼저 정리해야 합니다.')


def split_by_user(ratings, test_size=0.2, random_state=42):
    """사용자별 무작위 holdout. 단일 평점 사용자는 학습에만 남긴다."""
    validate_ratings(ratings)
    if not 0 < test_size < 1:
        raise ValueError('test_size는 0과 1 사이여야 합니다.')
    frame = ratings.reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    mask = np.zeros(len(frame), dtype=bool)
    for positions in frame.groupby('user_id', sort=False).indices.values():
        if len(positions) > 1:
            count = min(len(positions) - 1, max(1, int(len(positions) * test_size)))
            mask[rng.choice(positions, count, replace=False)] = True
    return frame.loc[~mask].copy(), frame.loc[mask].copy()


class ItemBasedCF:
    """Cosine KNN의 이웃 중 사용자가 평가한 항목만 가중평균한다.

    n_neighbors: 대상 아이템의 전체 이웃 수(기본 30).
    batch_size: KNN 조회 배치 크기. 전체 아이템 제곱 행렬 보관을 피한다.
    n_jobs: sklearn 병렬 작업 수. 기본 1로 메모리 급증을 방지한다.
    """

    def __init__(self, n_neighbors=30, batch_size=128, n_jobs=1):
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError('n_neighbors는 양의 정수여야 합니다.')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size는 양의 정수여야 합니다.')
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.n_jobs = n_jobs

    def fit(self, train_ratings):
        """학습 평점만 CSR 사용자×아이템 행렬 및 전치 행렬로 변환한다."""
        validate_ratings(train_ratings)
        frame = train_ratings
        self.user_ids_ = pd.Index(frame.user_id.unique())
        self.item_ids_ = pd.Index(frame.item_id.unique())
        rows = self.user_ids_.get_indexer(frame.user_id)
        cols = self.item_ids_.get_indexer(frame.item_id)
        self.user_item_ = csr_matrix((frame.rating.to_numpy(dtype=np.float64),
                                     (rows, cols)),
                                    shape=(len(self.user_ids_), len(self.item_ids_)))
        self.item_user_ = self.user_item_.T.tocsr()
        self.global_mean_ = float(frame.rating.mean())
        self.user_means_ = frame.groupby('user_id').rating.mean().to_dict()
        self.item_means_ = frame.groupby('item_id').rating.mean().to_dict()
        self.rating_range_ = (float(frame.rating.min()), float(frame.rating.max()))
        self.knn_ = NearestNeighbors(metric='cosine', algorithm='brute',
                                     n_jobs=self.n_jobs).fit(self.item_user_)
        count = len(self.item_ids_)
        k = min(self.n_neighbors, count - 1)
        self.neighbor_indices_ = np.empty((count, k), dtype=np.int32)
        self.neighbor_similarities_ = np.empty((count, k), dtype=float)
        for start in range(0, count, self.batch_size):
            distances, indices = self.knn_.kneighbors(
                self.item_user_[start:start + self.batch_size], n_neighbors=k + 1)
            for offset, (dist, idx) in enumerate(zip(distances, indices)):
                keep = idx != start + offset
                self.neighbor_indices_[start + offset] = idx[keep][:k]
                self.neighbor_similarities_[start + offset] = np.clip(1 - dist[keep][:k], 0, 1)
        return self

    def _check_fit(self):
        """미학습 모델 사용을 명확한 오류로 알린다."""
        if not hasattr(self, 'knn_'):
            raise ValueError('fit을 먼저 호출하세요.')

    def predict_with_details(self, user_id, item_id):
        """예측값, CF/대체 방식, 기여 이웃 수를 반환한다."""
        self._check_fit()
        u = self.user_ids_.get_indexer([user_id])[0]
        i = self.item_ids_.get_indexer([item_id])[0]
        if u >= 0 and i >= 0:
            neighbors = self.neighbor_indices_[i]
            weights = self.neighbor_similarities_[i]
            ratings = self.user_item_[u, neighbors].toarray().ravel()
            valid = (ratings > 0) & (weights > 1e-12)
            if valid.any():
                value = np.dot(weights[valid], ratings[valid]) / weights[valid].sum()
                return float(np.clip(value, *self.rating_range_)), 'cf', int(valid.sum())
        if user_id in self.user_means_:
            return float(self.user_means_[user_id]), 'user_mean', 0
        if item_id in self.item_means_:
            return float(self.item_means_[item_id]), 'item_mean', 0
        return self.global_mean_, 'global_mean', 0

    def predict(self, user_id, item_id):
        """Hybrid 조합에 사용할 단일 실수 평점을 반환한다."""
        return self.predict_with_details(user_id, item_id)[0]

    def recommend(self, user_id, top_n=10):
        """학습에서 미평가한 아이템을 점수 내림차순으로 반환한다.

        반환 DataFrame: item_id, predicted_rating, method, n_contributors.
        제목은 데이터셋 어댑터에서 별도로 결합한다.
        """
        self._check_fit()
        if not isinstance(top_n, int) or top_n < 1:
            raise ValueError('top_n은 양의 정수여야 합니다.')
        u = self.user_ids_.get_indexer([user_id])[0]
        seen = set(self.user_item_[u].indices) if u >= 0 else set()
        records = [(item, *self.predict_with_details(user_id, item))
                   for i, item in enumerate(self.item_ids_) if i not in seen]
        return pd.DataFrame(records, columns=['item_id', 'predicted_rating',
                                              'method', 'n_contributors']).sort_values(
            ['predicted_rating', 'item_id'], ascending=[False, True]).head(top_n).reset_index(drop=True)
