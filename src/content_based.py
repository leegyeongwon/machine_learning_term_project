"""콘텐츠 기반 필터링. 아이템 메타데이터로 사용자-아이템 평점을 예측한다."""


class ContentBasedCF:
    """아이템 특성(제목, 장르 등)을 이용해 평점을 예측한다.

    하이브리드 조합을 위해 ItemBasedCF와 같은 공개 API를 제공한다.
    학습 평점 컬럼: user_id, item_id, rating.
    예측 점수는 학습 평점 범위(MovieLens면 1~5)로 자른다.
    """

    def __init__(self):
        """콘텐츠 기반 모델을 생성한다."""
        raise NotImplementedError

    def fit(self, train_ratings):
        """학습 평점으로 모델을 학습한다.

        입력:
            train_ratings: user_id, item_id, rating 컬럼의 pandas DataFrame.
                구현 시 item_id에 대응하는 아이템 메타데이터(title, genres)가
                추가로 필요하다.
        출력:
            self
        하는 일:
            아이템 특성을 학습하고 이후 predict/recommend에 사용할 상태를 저장한다.
        """
        raise NotImplementedError

    def predict(self, user_id, item_id):
        """Hybrid 조합에 사용할 단일 실수 평점을 반환한다.

        입력:
            user_id: 사용자 ID
            item_id: 아이템 ID
        출력:
            float. 학습 평점 범위로 clip한 예측 점수(MovieLens면 1~5).
        하는 일:
            사용자와 아이템의 콘텐츠 유사도를 바탕으로 평점을 예측한다.
        """
        raise NotImplementedError

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
        raise NotImplementedError
