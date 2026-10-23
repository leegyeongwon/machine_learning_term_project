"""공통 데이터 로딩과 분할. 모델 구현과 독립적이다."""


def validate_ratings(ratings):
    """공통 입력 컬럼, 양수 유한 평점, 중복 사용자-아이템을 검증한다.

    입력:
        ratings: user_id, item_id, rating 컬럼의 pandas DataFrame.
    출력:
        없음. 실패 시 ValueError.
    하는 일:
        비어 있지 않은 필수 컬럼, 결측 없음, 평점은 유한한 양수,
        (user_id, item_id) 중복 없음을 확인한다. 0은 미평가이다.
    """
    raise NotImplementedError


def split_by_user(ratings, test_size=0.2, random_state=42):
    """사용자별 무작위 holdout. 단일 평점 사용자는 학습에만 남긴다.

    입력:
        ratings: user_id, item_id, rating 컬럼의 pandas DataFrame.
        test_size: 사용자별 테스트 비율. 0과 1 사이.
        random_state: 난수 시드.
    출력:
        (train_df, test_df)
    하는 일:
        사용자마다 floor(n * test_size)개를 테스트로 보내고 학습은 최소 1개를 남긴다.
    """
    raise NotImplementedError


def load_movielens(data_dir):
    """MovieLens 1M ratings.dat, movies.dat을 공통 컬럼명으로 읽는다.

    입력:
        data_dir: ratings.dat, movies.dat이 있는 경로 (기본 data/raw/ml-1m).
    출력:
        (ratings, movies)
        ratings 컬럼: user_id, item_id, rating, timestamp
        movies 컬럼: item_id, title, genres
    하는 일:
        sep='::', engine='python', encoding='latin-1'로 읽고 validate_ratings를 적용한다.
        평점 범위는 1~5여야 한다.
    """
    raise NotImplementedError
