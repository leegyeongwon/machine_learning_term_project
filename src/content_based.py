"""콘텐츠 기반 필터링. 유저 평균 + Ridge로 평점을 예측한다."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ITEM_FEATURES = ROOT / "data" / "processed" / "movies_multihot_data.npz"
N_OCCUPATIONS = 21
YEAR_RE = re.compile(r"\((\d{4})\)")


def parse_year(title):
    """제목에서 개봉연도를 뽑는다. 없으면 NaN."""
    match = YEAR_RE.search(str(title))
    return float(match.group(1)) if match else np.nan


def split_by_user(ratings, test_size=0.2, random_state=42):
    """사용자별 무작위 holdout. 단일 평점 사용자는 학습에만 남긴다."""
    if not 0 < test_size < 1:
        raise ValueError("test_size는 0과 1 사이여야 합니다.")
    frame = ratings.reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    mask = np.zeros(len(frame), dtype=bool)
    for positions in frame.groupby("user_id", sort=False).indices.values():
        if len(positions) > 1:
            count = min(len(positions) - 1, max(1, int(len(positions) * test_size)))
            mask[rng.choice(positions, count, replace=False)] = True
    return frame.loc[~mask].copy(), frame.loc[mask].copy()


def split_by_user_three(ratings, val_size=0.2, test_size=0.2, random_state=42):
    """사용자별 train/validation/test. 학습은 최소 1개를 남긴다."""
    if not 0 < val_size < 1 or not 0 < test_size < 1:
        raise ValueError("val_size와 test_size는 0과 1 사이여야 합니다.")
    if val_size + test_size >= 1:
        raise ValueError("val_size와 test_size의 합은 1보다 작아야 합니다.")
    frame = ratings.reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    val_mask = np.zeros(len(frame), dtype=bool)
    test_mask = np.zeros(len(frame), dtype=bool)
    for positions in frame.groupby("user_id", sort=False).indices.values():
        n = len(positions)
        if n <= 1:
            continue
        n_test = min(n - 1, max(1, int(n * test_size)))
        n_val = min(n - 1 - n_test, max(0, int(n * val_size)))
        if n - n_test > 1 and n_val == 0:
            n_val = 1
        chosen = rng.choice(positions, n_test + n_val, replace=False)
        test_mask[chosen[:n_test]] = True
        if n_val:
            val_mask[chosen[n_test:]] = True
    train_mask = ~(val_mask | test_mask)
    return frame.loc[train_mask].copy(), frame.loc[val_mask].copy(), frame.loc[test_mask].copy()


class _RidgeModel:
    """배치 정규방정식으로 맞춘 Ridge. sklearn과 같은 predict 인터페이스."""

    def __init__(self, coef, intercept):
        self.coef_ = np.asarray(coef, dtype=float)
        self.intercept_ = float(intercept)
        self.n_features_in_ = int(self.coef_.shape[0])

    def predict(self, X):
        """(n, p) 피처 행렬에 대해 잔차 예측값을 반환한다."""
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


class ContentBasedCF:
    """아이템 내용과 이 유저의 학습 평점으로 평점을 예측한다.

    하이브리드 조합을 위해 ItemBasedCF와 같은 공개 API를 제공한다.
    학습 평점 컬럼: user_id, item_id, rating. timestamp가 있으면 최근 평점에 더 큰 가중을 준다.
    아이템 벡터는 movies.dat, movies_multihot_data.npz, 또는 book_content.npz 를 쓴다.
    유저 프로필·Ridge는 fit의 학습 평점만으로 다시 계산한다.
    다른 유저가 그 아이템에 준 평점은 피처로 쓰지 않는다.
    예측 점수는 학습 평점 범위(MovieLens·Goodbooks면 1~5)로 자른다.
    """

    def __init__(
        self,
        item_features_path=None,
        movies_path=None,
        users_path=None,
        ridge_alpha=10.0,
        recency_half_life_days=365.0,
    ):
        """콘텐츠 기반 모델을 생성한다.

        입력:
            item_features_path: 영화 장르 npz 또는 book_content.npz 경로.
            movies_path: movies.dat 경로. 있으면 장르·제목·연도를 여기서 만든다.
            users_path: users.dat 경로. 있으면 성별·나이·직업을 상호작용 피처로 쓴다.
            ridge_alpha: Ridge 규제 세기.
            recency_half_life_days: 평점 가중치가 절반이 되는 일 수. timestamp가 있을 때만 쓴다.
        """
        self.item_features_path = Path(item_features_path) if item_features_path else DEFAULT_ITEM_FEATURES
        self.movies_path = Path(movies_path) if movies_path else None
        self.users_path = Path(users_path) if users_path else None
        self.ridge_alpha = float(ridge_alpha)
        self.recency_half_life_days = float(recency_half_life_days)

    def fit(self, train_ratings):
        """학습 평점과 영화·유저 속성으로 Ridge를 학습한다.

        입력:
            train_ratings: user_id, item_id, rating 컬럼의 pandas DataFrame.
        출력:
            self
        하는 일:
            영화 장르 TF-IDF와 연도를 만들고, train만으로 유저 평균·취향 벡터를 만든 뒤
            (평점 - 유저 평균)을 Ridge로 학습한다.
        """
        frame = train_ratings.copy()
        if "item_id" not in frame.columns and "movie_id" in frame.columns:
            frame = frame.rename(columns={"movie_id": "item_id"})
        required = ["user_id", "item_id", "rating"]
        if not set(required).issubset(frame.columns) or frame.empty:
            raise ValueError("비어 있지 않은 user_id, item_id, rating 데이터가 필요합니다.")

        self._load_item_content()
        movie_multihot = self.item_multihot_
        n_movies, n_genres = movie_multihot.shape
        doc_freq = movie_multihot.sum(axis=0)
        idf = np.log((n_movies + 1.0) / (doc_freq + 1.0)) + 1.0
        tfidf = movie_multihot * idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.item_tfidf_ = tfidf / norms

        if getattr(self, "year_norm_", None) is None:
            years = np.array([parse_year(title) for title in self.titles_], dtype=float)
            year_mean = np.nanmean(years) if np.isfinite(years).any() else 0.0
            years = np.where(np.isfinite(years), years, year_mean)
            year_std = np.nanstd(years)
            if not np.isfinite(year_std) or year_std < 1e-12:
                year_std = 1.0
            self.year_norm_ = ((years - year_mean) / year_std).astype(float)

        frame["user_id"] = frame["user_id"].astype(int)
        frame["item_id"] = frame["item_id"].astype(int)
        frame["rating"] = frame["rating"].astype(float)
        frame = frame[frame["item_id"].isin(self.item_index_)].copy()
        if frame.empty:
            raise ValueError("학습 평점에 아이템 벡터와 맞는 item_id가 없습니다.")

        self.global_mean_ = float(frame["rating"].mean())
        self.user_means_ = frame.groupby("user_id")["rating"].mean().to_dict()
        self.item_means_ = frame.groupby("item_id")["rating"].mean().to_dict()
        self.rating_range_ = (float(frame["rating"].min()), float(frame["rating"].max()))
        self.seen_ = frame.groupby("user_id")["item_id"].apply(set).to_dict()

        if "timestamp" in frame.columns:
            age_days = (float(frame["timestamp"].max()) - frame["timestamp"].astype(float)) / 86400.0
            frame["recency_w"] = 0.5 ** (age_days / self.recency_half_life_days)
        else:
            frame["recency_w"] = 1.0

        frame["user_mean"] = frame["user_id"].map(self.user_means_)
        frame["residual"] = frame["rating"] - frame["user_mean"]
        frame["weighted_residual"] = frame["recency_w"] * frame["residual"]

        self.user_ids_ = np.array(sorted(self.user_means_), dtype=int)
        self.user_index_ = {int(u): i for i, u in enumerate(self.user_ids_)}
        self.user_mean_arr_ = np.array([self.user_means_[int(u)] for u in self.user_ids_], dtype=float)

        n_users = len(self.user_ids_)
        profiles = np.zeros((n_users, n_genres), dtype=float)
        u_rows = frame["user_id"].map(self.user_index_).to_numpy()
        i_rows = frame["item_id"].map(self.item_index_).to_numpy()
        np.add.at(
            profiles,
            u_rows,
            self.item_tfidf_[i_rows] * frame["weighted_residual"].to_numpy()[:, None],
        )
        profile_norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        profile_norms[profile_norms == 0] = 1.0
        self.user_profiles_ = profiles / profile_norms
        self._build_extra_user_profiles(
            u_rows,
            i_rows,
            frame["weighted_residual"].to_numpy(),
            frame["recency_w"].to_numpy(),
        )

        self._load_user_demographics()
        self.ridge_ = self._fit_ridge_batched(
            u_rows,
            i_rows,
            frame["residual"].to_numpy(dtype=float),
            frame["recency_w"].to_numpy(dtype=float),
        )
        return self

    def _fit_ridge_batched(self, user_idx, item_idx, y, sample_weight, batch_size=8192):
        """전체 행렬을 만들지 않고 X'WX w = X'Wy 를 배치로 풀어 Ridge를 맞춘다."""
        n_features = self._features_from_indices(user_idx[:1], item_idx[:1]).shape[1]
        dim = n_features + 1
        xtx = np.zeros((dim, dim), dtype=np.float64)
        xty = np.zeros(dim, dtype=np.float64)
        n = len(y)
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            features = self._features_from_indices(user_idx[sl], item_idx[sl]).astype(np.float64, copy=False)
            weights = np.sqrt(np.clip(sample_weight[sl], 1e-12, None))
            weighted_x = features * weights[:, None]
            weighted_y = y[sl] * weights
            xtx[0, 0] += np.dot(weights, weights)
            xtx[0, 1:] += weights @ weighted_x
            xtx[1:, 1:] += weighted_x.T @ weighted_x
            xty[0] += np.dot(weights, weighted_y)
            xty[1:] += weighted_x.T @ weighted_y
        xtx[1:, 0] = xtx[0, 1:]
        xtx.flat[:: dim + 1] += self.ridge_alpha
        xtx[0, 0] -= self.ridge_alpha
        coef_aug = np.linalg.solve(xtx, xty)
        return _RidgeModel(coef_aug[1:], coef_aug[0])

    def _reset_extra_item_features(self):
        """책 전용 피처를 비운다. 영화 경로에서 이전 값이 남지 않게 한다."""
        self.year_norm_ = None
        self.item_plot_ = None
        self.item_numeric_ = None
        self.author_csr_ = None
        self.user_plot_ = None
        self.user_numeric_ = None
        self.user_author_csr_ = None

    def _csr_from_npz(self, loaded, prefix):
        """npz에 저장된 CSR 구성 요소를 희소행렬로 복원한다."""
        keys = (f"{prefix}_data", f"{prefix}_indices", f"{prefix}_indptr", f"{prefix}_shape")
        if not all(key in loaded for key in keys):
            return None
        return csr_matrix(
            (loaded[f"{prefix}_data"], loaded[f"{prefix}_indices"], loaded[f"{prefix}_indptr"]),
            shape=tuple(loaded[f"{prefix}_shape"]),
        )

    def _load_book_content(self, loaded):
        """Goodbooks book_content.npz에서 장르·작가·줄거리·수치 특성을 읽는다."""
        self.item_ids_ = np.asarray(loaded["book_ids"])
        self.titles_ = np.asarray(loaded["titles"]) if "titles" in loaded else np.array([""] * len(self.item_ids_))
        genre_csr = self._csr_from_npz(loaded, "genres")
        if genre_csr is None:
            raise ValueError("book_content.npz에 장르 희소행렬이 없습니다.")
        self.item_multihot_ = (genre_csr.toarray() > 1e-12).astype(float)
        if "genre_names" in loaded:
            self.genre_cols_ = [str(name) for name in loaded["genre_names"]]
        else:
            self.genre_cols_ = [f"genre_{j}" for j in range(self.item_multihot_.shape[1])]
        self.author_csr_ = self._csr_from_npz(loaded, "authors")
        if self.author_csr_ is not None:
            self.author_csr_ = normalize(self.author_csr_.astype(np.float32), norm="l2", axis=1)
        if "plot_embeddings" in loaded:
            plot = np.asarray(loaded["plot_embeddings"], dtype=np.float32)
            norms = np.linalg.norm(plot, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.item_plot_ = plot / norms
        if "numeric_features" in loaded:
            numeric = np.asarray(loaded["numeric_features"], dtype=float)
            if numeric.ndim == 1:
                numeric = numeric[:, None]
            self.item_numeric_ = numeric
            if numeric.shape[1] >= 2:
                self.year_norm_ = numeric[:, 1].astype(float)
            elif numeric.shape[1] == 1:
                self.year_norm_ = numeric[:, 0].astype(float)

    def _load_item_content(self):
        """movies.dat이 있으면 그걸 쓰고, 없으면 영화/책 npz를 쓴다."""
        self._reset_extra_item_features()
        if self.movies_path is not None and self.movies_path.exists():
            movies = pd.read_csv(
                self.movies_path,
                sep="::",
                engine="python",
                encoding="ISO-8859-1",
                names=["item_id", "title", "genres"],
            )
            genre_lists = movies["genres"].fillna("").str.split("|")
            genre_cols = sorted({g for row in genre_lists for g in row if g})
            multi_hot = np.zeros((len(movies), len(genre_cols)), dtype=float)
            genre_index = {g: j for j, g in enumerate(genre_cols)}
            for i, row in enumerate(genre_lists):
                for genre in row:
                    if genre in genre_index:
                        multi_hot[i, genre_index[genre]] = 1.0
            self.item_ids_ = movies["item_id"].to_numpy()
            self.titles_ = movies["title"].to_numpy()
            self.genre_cols_ = genre_cols
            self.item_multihot_ = multi_hot
        elif self.item_features_path.exists():
            loaded = np.load(self.item_features_path, allow_pickle=True)
            if "book_ids" in loaded:
                self._load_book_content(loaded)
            else:
                self.item_ids_ = np.asarray(loaded["movie_ids"])
                self.titles_ = np.asarray(loaded["titles"])
                self.genre_cols_ = list(loaded["classes"])
                item_vecs = np.asarray(loaded["multi_hot_norm"], dtype=float)
                self.item_multihot_ = (item_vecs > 1e-12).astype(float)
        else:
            raise FileNotFoundError(
                "아이템 데이터가 없습니다. movies.dat 또는 "
                f"{self.item_features_path} 를 준비하세요."
            )
        self.item_index_ = {int(item_id): i for i, item_id in enumerate(self.item_ids_)}

    def _build_extra_user_profiles(self, user_idx, item_idx, weighted_residual, recency_w):
        """줄거리·수치·작가 취향 벡터를 train 평점만으로 만든다."""
        n_users = len(self.user_ids_)
        weights = weighted_residual.astype(float)
        recency = np.asarray(recency_w, dtype=float)
        if self.item_plot_ is not None:
            profiles = np.zeros((n_users, self.item_plot_.shape[1]), dtype=np.float32)
            np.add.at(profiles, user_idx, self.item_plot_[item_idx] * weights[:, None])
            norms = np.linalg.norm(profiles, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.user_plot_ = profiles / norms
        if self.item_numeric_ is not None:
            profiles = np.zeros((n_users, self.item_numeric_.shape[1]), dtype=float)
            weight_sum = np.zeros(n_users, dtype=float)
            np.add.at(profiles, user_idx, self.item_numeric_[item_idx] * recency[:, None])
            np.add.at(weight_sum, user_idx, recency)
            weight_sum[weight_sum == 0] = 1.0
            self.user_numeric_ = profiles / weight_sum[:, None]
        if self.author_csr_ is not None:
            pref = csr_matrix(
                (weights, (user_idx, item_idx)),
                shape=(n_users, self.author_csr_.shape[0]),
            )
            user_author = pref @ self.author_csr_
            self.user_author_csr_ = normalize(user_author, norm="l2", axis=1)

    def _load_user_demographics(self):
        """users.dat이 있으면 성별·나이·직업을 읽고, 없으면 인구통계 피처를 끈다."""
        self.has_demo_ = False
        n_users = len(self.user_ids_)
        self.is_male_ = np.zeros(n_users, dtype=float)
        self.age_scaled_ = np.zeros(n_users, dtype=float)
        self.occupation_ = np.zeros((n_users, N_OCCUPATIONS), dtype=float)
        if self.users_path is None or not self.users_path.exists():
            return
        users = pd.read_csv(
            self.users_path,
            sep="::",
            engine="python",
            encoding="ISO-8859-1",
            names=["user_id", "gender", "age", "occupation", "zip"],
        )
        users["user_id"] = users["user_id"].astype(int)
        by_id = users.set_index("user_id")
        by_id.index = by_id.index.astype(int)
        for i, user_id in enumerate(self.user_ids_):
            if int(user_id) not in by_id.index:
                continue
            row = by_id.loc[int(user_id)]
            self.is_male_[i] = 1.0 if str(row["gender"]).upper() == "M" else 0.0
            self.age_scaled_[i] = (float(row["age"]) - 25.0) / 25.0
            occ = int(row["occupation"])
            if 0 <= occ < N_OCCUPATIONS:
                self.occupation_[i, occ] = 1.0
        self.has_demo_ = True

    def _features_from_indices(self, user_idx, item_idx):
        """유저·아이템 위치 배열로 Ridge 입력 행렬을 만든다."""
        tfidf = self.item_tfidf_[item_idx]
        year = self.year_norm_[item_idx][:, None]
        profile = self.user_profiles_[user_idx]
        parts = [tfidf, year, profile * tfidf]
        if self.item_plot_ is not None and self.user_plot_ is not None:
            item_plot = self.item_plot_[item_idx]
            plot_cos = np.sum(self.user_plot_[user_idx] * item_plot, axis=1, keepdims=True)
            parts.extend([item_plot, plot_cos])
        if self.item_numeric_ is not None:
            item_num = self.item_numeric_[item_idx]
            parts.append(item_num)
            if self.user_numeric_ is not None:
                parts.append(item_num - self.user_numeric_[user_idx])
        if self.author_csr_ is not None and self.user_author_csr_ is not None:
            ua = self.user_author_csr_[user_idx]
            ia = self.author_csr_[item_idx]
            author_cos = np.asarray(ua.multiply(ia).sum(axis=1)).reshape(-1, 1)
            parts.append(author_cos)
        if self.has_demo_:
            male = self.is_male_[user_idx][:, None]
            age = self.age_scaled_[user_idx][:, None]
            occ = self.occupation_[user_idx]
            parts.extend([male * tfidf, age * year, age * tfidf, occ * year])
        return np.hstack(parts).astype(np.float32)

    def _check_fit(self):
        """미학습 모델 사용을 명확한 오류로 알린다."""
        if not hasattr(self, "ridge_"):
            raise ValueError("fit을 먼저 호출하세요.")

    def _content_scores(self, user_idx, item_idx, batch_size=8192):
        """학습된 Ridge로 콘텐츠 점수를 계산한다."""
        user_idx = np.asarray(user_idx)
        item_idx = np.asarray(item_idx)
        n = len(user_idx)
        out = np.empty(n, dtype=float)
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            features = self._features_from_indices(user_idx[sl], item_idx[sl])
            raw = self.user_mean_arr_[user_idx[sl]] + self.ridge_.predict(features)
            out[sl] = np.clip(raw, *self.rating_range_)
        return out

    def _contributors(self, user_idx, item_idx):
        """유저 취향과 영화가 같이 가진 장르 수를 센다."""
        profile = self.user_profiles_[np.asarray(user_idx)]
        item = self.item_multihot_[np.asarray(item_idx)]
        return np.sum((profile != 0) & (item != 0), axis=1).astype(int)

    def predict_with_details(self, user_id, item_id):
        """예측값, 방식, 기여 장르 수를 반환한다."""
        self._check_fit()
        u = self.user_index_.get(int(user_id), -1)
        i = self.item_index_.get(int(item_id), -1)
        if u >= 0 and i >= 0:
            score = float(self._content_scores([u], [i])[0])
            contributors = int(self._contributors([u], [i])[0])
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
            유저 평균에, 영화 장르·연도·취향 매칭(과 인구통계 상호작용) Ridge 보정을 더한다.
        """
        return self.predict_with_details(user_id, item_id)[0]

    def predict_many(self, user_ids, item_ids):
        """여러 (user, item)을 한 번에 예측한다."""
        self._check_fit()
        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)
        n = len(user_ids)
        scores = np.empty(n, dtype=float)
        methods = np.empty(n, dtype=object)
        contributors = np.zeros(n, dtype=int)
        u_idx = np.fromiter((self.user_index_.get(int(u), -1) for u in user_ids), dtype=int, count=n)
        i_idx = np.fromiter((self.item_index_.get(int(i), -1) for i in item_ids), dtype=int, count=n)
        valid = (u_idx >= 0) & (i_idx >= 0)
        if valid.any():
            scores[valid] = self._content_scores(u_idx[valid], i_idx[valid])
            methods[valid] = "content"
            contributors[valid] = self._contributors(u_idx[valid], i_idx[valid])
        for k in np.flatnonzero(~valid):
            score, method, n_c = self.predict_with_details(user_ids[k], item_ids[k])
            scores[k] = score
            methods[k] = method
            contributors[k] = n_c
        return scores, methods, contributors

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
        n_items = len(self.item_ids_)
        if u >= 0:
            item_idx = np.arange(n_items)
            scores = self._content_scores(np.full(n_items, u), item_idx)
            methods = np.full(n_items, "content")
            contributors = self._contributors(np.full(n_items, u), item_idx)
        else:
            fallback = float(self.user_means_.get(user_id, self.global_mean_))
            scores = np.full(n_items, fallback)
            methods = np.full(
                n_items, "user_mean" if user_id in self.user_means_ else "global_mean"
            )
            contributors = np.zeros(n_items, dtype=int)

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
