"""아이템 기반 CF와 콘텐츠 기반 점수를 결합하는 하이브리드 추천."""


class HybridCF:
    """alpha * cf + (1 - alpha) * content 로 평점을 예측한다.

    ItemBasedCF, ContentBasedCF와 같은 공개 API를 제공한다.
    학습 평점 컬럼: user_id, item_id, rating.
    예측 점수는 학습 평점 범위(MovieLens면 1~5)로 자른다.
    """

    def __init__(self, alpha=0.5):
        """하이브리드 모델을 생성한다.

        입력:
            alpha: CF 점수 비중. 0~1. content 비중은 (1 - alpha).
        """
        raise NotImplementedError

    def fit(self, train_ratings):
        """학습 평점으로 CF와 콘텐츠 모델을 함께 학습한다.

        입력:
            train_ratings: user_id, item_id, rating 컬럼의 pandas DataFrame.
        출력:
            self
        하는 일:
            같은 train 분할로 두 모델을 학습한다. 테스트 평점은 쓰지 않는다.
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
            cf_score와 content_score를 alpha로 가중합한다.
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
