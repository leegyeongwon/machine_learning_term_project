"""데이터 로딩과 독립적인 Validation 가중 결합."""
import numpy as np
import pandas as pd


def metrics(actual, predicted):
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    if a.shape != p.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError('빈 값, 차원 불일치 또는 비유한 값')
    e = p - a
    return {'mae': float(np.abs(e).mean()), 'rmse': float(np.sqrt((e * e).mean()))}


def align_predictions(left, right):
    keys = ['user_id', 'item_id']
    for f in (left, right):
        if f.empty or f.duplicated(keys).any() or f[keys + ['rating', 'predicted_rating']].isna().any().any():
            raise ValueError('예측 중복, 결측 또는 빈 데이터')
        if not np.isfinite(f.predicted_rating).all() or not f.predicted_rating.between(.5, 5).all():
            raise ValueError('예측 범위 오류')
    paired = left.merge(right, on=keys, how='outer', suffixes=('_item', '_content'), indicator=True, validate='one_to_one')
    if not paired['_merge'].eq('both').all() or not np.array_equal(paired.rating_item, paired.rating_content):
        raise ValueError('평가 쌍 또는 정답 불일치')
    return paired


def tune_weight(actual, item, content):
    rows = [{'alpha': a, **metrics(actual, a * np.asarray(item) + (1-a) * np.asarray(content))}
            for a in np.linspace(0, 1, 101)]
    table = pd.DataFrame(rows)
    return float(table.sort_values(['mae', 'rmse', 'alpha']).iloc[0].alpha), table


class HybridCF:
    def __init__(self, alpha=.5, item_model=None, content_model=None):
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('alpha는 0~1')
        self.alpha, self.item_model, self.content_model = alpha, item_model, content_model

    def fit(self, train_ratings):
        if self.item_model is None or self.content_model is None:
            raise ValueError('두 기본 모델 필요')
        self.item_model.fit(train_ratings)
        self.content_model.fit(train_ratings)
        self.seen_ = train_ratings.groupby('user_id').item_id.apply(set).to_dict()
        self.item_ids_ = np.union1d(self.item_model.item_ids_, self.content_model.item_ids_)
        return self

    def predict(self, user_id, item_id):
        return float(np.clip(self.alpha*self.item_model.predict(user_id, item_id) +
                             (1-self.alpha)*self.content_model.predict(user_id, item_id), .5, 5))

    def predict_many(self, user_ids, item_ids):
        if len(user_ids) != len(item_ids):
            raise ValueError('ID 길이 불일치')
        return np.array([self.predict(u, i) for u, i in zip(user_ids, item_ids)])

    def recommend(self, user_id, top_n=10):
        if not isinstance(top_n, int) or top_n < 1:
            raise ValueError('top_n은 양의 정수')
        ids = [i for i in self.item_ids_ if i not in self.seen_.get(user_id, set())]
        return pd.DataFrame({'item_id': ids, 'predicted_rating': self.predict_many([user_id]*len(ids), ids)}).sort_values(
            ['predicted_rating', 'item_id'], ascending=[False, True]).head(top_n).reset_index(drop=True)
