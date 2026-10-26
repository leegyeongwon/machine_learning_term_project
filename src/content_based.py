"""콘텐츠 기반 필터링. 팀원 영화 장르 벡터와 학습 평점으로 평점을 예측한다."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ITEM_FEATURES = ROOT / "data" / "processed" / "movies_multihot_data.npz"


class ContentBasedCF:
    """장르 벡터 코사인 유사도로 평점을 예측한다.

    하이브리드 조합을 위해 ItemBasedCF와 같은 공개 API를 제공한다.
    학습 평점 컬럼: user_id, item_id, rating.
    영화 벡터는 기본으로 data/processed/movies_multihot_data.npz 를 쓴다.
    유저 프로필은 fit의 학습 평점만으로 다시 계산한다.
    예측 점수는 학습 평점 범위(MovieLens면 1~5)로 자른다.
    """

    def __init__(self, item_features_path=None):
        """콘텐츠 기반 모델을 생성한다.

        입력:
            item_features_path: 영화 장르 npz 경로. None이면 기본 경로를 쓴다.
        """
        self.item_features_path = Path(item_features_path) if item_features_path else DEFAULT_ITEM_FEATURES

    def fit(self, train_ratings):
        """학습 평점과 영화 장르 벡터로 모델을 학습한다.

        입력:
            train_ratings: user_id, item_id, rating 컬럼의 pandas DataFrame.
        출력:
            self
        하는 일:
            npz에서 영화 벡터를 읽고, train만으로 유저 프로필을 만든다.
        """
        frame = train_ratings.copy()
        if "item_id" not in frame.columns and "movie_id" in frame.columns:
            frame = frame.rename(columns={"movie_id": "item_id"})
        required = ["user_id", "item_id", "rating"]
        if not set(required).issubset(frame.columns) or frame.empty:
            raise ValueError("비어 있지 않은 user_id, item_id, rating 데이터가 필요합니다.")

        if not self.item_features_path.exists():
            raise FileNotFoundError(
                f"영화 벡터 파일이 없습니다: {self.item_features_path}\n"
                "preprocessing_and_data/movielens-1m 에서 "
                "python preprocess_content_based.py 를 먼저 실행하세요."
            )

        loaded = np.load(self.item_features_path, allow_pickle=True)
        self.item_ids_ = np.asarray(loaded["movie_ids"])
        self.titles_ = np.asarray(loaded["titles"])
        self.genre_cols_ = list(loaded["classes"])
        self.item_vecs_ = np.asarray(loaded["multi_hot_norm"], dtype=float)
        self.item_index_ = {int(item_id): i for i, item_id in enumerate(self.item_ids_)}
        movie_multihot = (self.item_vecs_ > 1e-12).astype(int)
        movie_feature_df = pd.DataFrame(
            movie_multihot, index=self.item_ids_, columns=self.genre_cols_
        )

        frame = frame.copy()
        frame["adjusted_rating"] = frame["rating"].astype(float) - 3.0
        merged = frame.merge(movie_feature_df, left_on="item_id", right_index=True, how="inner")
        for col in self.genre_cols_:
            merged[col] = merged[col] * merged["adjusted_rating"]

        user_profiles_raw = merged.groupby("user_id")[self.genre_cols_].sum()
        raw_values = user_profiles_raw.to_numpy(dtype=float)
        norms = np.linalg.norm(raw_values, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.user_ids_ = user_profiles_raw.index.to_numpy()
        self.user_index_ = {int(u): i for i, u in enumerate(self.user_ids_)}
        self.user_profiles_norm_ = raw_values / norms
        self.seen_ = frame.groupby("user_id")["item_id"].apply(set).to_dict()
        self.global_mean_ = float(frame["rating"].mean())
        self.user_means_ = frame.groupby("user_id")["rating"].mean().to_dict()
        self.item_means_ = frame.groupby("item_id")["rating"].mean().to_dict()
        self.rating_range_ = (float(frame["rating"].min()), float(frame["rating"].max()))
        return self

    def _check_fit(self):
        """미학습 모델 사용을 명확한 오류로 알린다."""
        if not hasattr(self, "item_vecs_"):
            raise ValueError("fit을 먼저 호출하세요.")

    def predict_with_details(self, user_id, item_id):
        """예측값, 방식, 기여 장르 수를 반환한다."""
        self._check_fit()
        u = self.user_index_.get(int(user_id), -1)
        i = self.item_index_.get(int(item_id), -1)
        if u >= 0 and i >= 0:
            user_vec = self.user_profiles_norm_[u]
            item_vec = self.item_vecs_[i]
            sim = float(np.dot(user_vec, item_vec))
            score = float(np.clip(3.0 + 2.0 * sim, *self.rating_range_))
            contributors = int(np.sum((user_vec != 0) & (item_vec != 0)))
            return score, "content", contributors
        if user_id in self.user_means_:
            return float(self.user_means_[user_id]), "user_mean", 0
        if item_id in self.item_means_:
            return float(self.item_means_[item_id]), "item_mean", 0
        return self.global_mean_, "global_mean", 0

    def predict(self, user_id, item_id):
        """Hybrid 조합에 사용할 단일 실수 평점을 반환한다.

        입력:
            user_id: 사용자 ID
            item_id: 아이템 ID
        출력:
            float. 학습 평점 범위로 clip한 예측 점수(MovieLens면 1~5).
        하는 일:
            유저 프로필과 영화 장르 벡터의 코사인을 1~5점으로 바꾼다.
        """
        return self.predict_with_details(user_id, item_id)[0]

    def recommend(self, user_id, top_n=10):
        """학습에서 미평가한 아이템을 점수 내림차순으로 반환한다.

        입력:
            user_id: 사용자 ID
            top_n: 반환할 추천 개수. 양의 정수.
        출력:
            pandas.DataFrame. 컬럼: item_id, predicted_rating, method, n_contributors.
            점수는 내림차순, 동점은 item_id 오름차순.
        하는 일:
            학습에서 이미 평가한 아이템은 제외하고 top_n개를 고른다.
        """
        self._check_fit()
        if not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n은 양의 정수여야 합니다.")

        seen = self.seen_.get(user_id, set())
        u = self.user_index_.get(int(user_id), -1)
        if u >= 0:
            sims = self.item_vecs_ @ self.user_profiles_norm_[u]
            scores = np.clip(3.0 + 2.0 * sims, *self.rating_range_)
            methods = np.full(len(self.item_ids_), "content")
            contributors = np.sum((self.item_vecs_ != 0) & (self.user_profiles_norm_[u] != 0), axis=1)
        else:
            fallback = float(self.user_means_.get(user_id, self.global_mean_))
            scores = np.full(len(self.item_ids_), fallback)
            methods = np.full(len(self.item_ids_), "user_mean" if user_id in self.user_means_ else "global_mean")
            contributors = np.zeros(len(self.item_ids_), dtype=int)

        rec = pd.DataFrame({
            "item_id": self.item_ids_,
            "predicted_rating": scores,
            "method": methods,
            "n_contributors": contributors,
        })
        rec = rec[~rec["item_id"].isin(seen)]
        return rec.sort_values(
            ["predicted_rating", "item_id"], ascending=[False, True]
        ).head(top_n).reset_index(drop=True)
