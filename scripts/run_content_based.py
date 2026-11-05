"""MovieLens/Goodbooks 로딩, Ridge alpha 선택, 학습/평가, 추천 저장 CLI."""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.content_based import ContentBasedCF, split_by_user_three

GOODBOOKS_RATINGS_URL = (
    "https://raw.githubusercontent.com/"
    "malcolmosh/goodbooks-10k-extended/master/ratings.csv"
)
DEFAULT_RIDGE_ALPHAS = (1.0, 10.0, 100.0)


def load_movielens_ratings(data_dir):
    """MovieLens ratings.dat을 user_id, item_id, rating, timestamp로 읽는다."""
    path = Path(data_dir) / "ratings.dat"
    ratings = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="ISO-8859-1",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    if ratings.empty:
        raise ValueError("평점 데이터가 비어 있습니다.")
    return ratings


def load_goodbooks_ratings(data_dir):
    """Goodbooks ratings.csv을 읽고 book_id를 item_id로 바꾼다. 없으면 팀원과 같은 URL에서 받는다."""
    data_dir = Path(data_dir)
    path = data_dir / "ratings.csv"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Goodbooks ratings가 없어 다운로드합니다: {GOODBOOKS_RATINGS_URL}")
        urllib.request.urlretrieve(GOODBOOKS_RATINGS_URL, path)
    ratings = pd.read_csv(path)
    if "book_id" in ratings.columns and "item_id" not in ratings.columns:
        ratings = ratings.rename(columns={"book_id": "item_id"})
    required = {"user_id", "item_id", "rating"}
    if not required.issubset(ratings.columns):
        raise ValueError(f"Goodbooks 평점에 {sorted(required)} 컬럼이 필요합니다.")
    ratings = ratings[["user_id", "item_id", "rating"]].copy()
    if ratings.empty:
        raise ValueError("평점 데이터가 비어 있습니다.")
    return ratings


def evaluate(model, test):
    """전체 테스트를 평가하고 행별 근거를 남긴다.

    입력:
        model: fit이 끝난 ContentBasedCF.
        test: user_id, item_id, rating 컬럼의 DataFrame.
    출력:
        (metrics: dict, predictions: DataFrame)
        predictions 컬럼: user_id, item_id, rating, predicted_rating, method, n_contributors
    하는 일:
        테스트 행을 버리지 않고 MAE, RMSE를 계산한다.
    """
    scores, methods, contributors = model.predict_many(
        test["user_id"].to_numpy(), test["item_id"].to_numpy()
    )
    predictions = test[["user_id", "item_id", "rating"]].reset_index(drop=True).copy()
    predictions["predicted_rating"] = scores
    predictions["method"] = methods
    predictions["n_contributors"] = contributors
    error = predictions.predicted_rating - predictions.rating
    content = predictions.method.eq("content")
    metrics = {
        "n_test": len(test),
        "n_evaluated": len(predictions),
        "n_content": int(content.sum()),
        "n_fallback": int((~content).sum()),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }
    return metrics, predictions


def parse_alphas(text):
    """쉼표로 나눈 ridge_alpha 후보를 실수 튜플로 바꾼다."""
    values = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError("ridge_alpha 후보가 비어 있습니다.")
    return values


def select_ridge_alpha(train, val, alphas, model_kwargs):
    """train만으로 각 alpha를 맞추고 val MAE가 가장 작은 값을 고른다."""
    rows = []
    best_alpha = None
    best_mae = float("inf")
    for alpha in alphas:
        model = ContentBasedCF(ridge_alpha=alpha, **model_kwargs).fit(train)
        metrics, _ = evaluate(model, val)
        row = {
            "ridge_alpha": float(alpha),
            "val_mae": metrics["mae"],
            "val_rmse": metrics["rmse"],
            "n_content": metrics["n_content"],
        }
        rows.append(row)
        print(json.dumps({"tune": row}, ensure_ascii=False))
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_alpha = float(alpha)
    return best_alpha, best_mae, rows


def save_run(output_dir, model, train, val, test, metrics, predictions, recommendations, summary):
    """metrics, 예측, 추천, run_summary를 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    recommendations.to_csv(
        output_dir / f"recommendations_user_{summary['user_id']}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    summary = {
        **summary,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "used_demographics": bool(model.has_demo_),
        "n_features": int(getattr(model.ridge_, "n_features_in_", 0)),
        "metrics": metrics,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_recommendations(model, user_id, top_n):
    """추천에 제목 컬럼을 붙인다."""
    recommendations = model.recommend(user_id, top_n)
    title_map = dict(zip(model.item_ids_.astype(int), model.titles_))
    recommendations["title"] = recommendations["item_id"].map(lambda i: title_map.get(int(i), ""))
    return recommendations[["item_id", "title", "predicted_rating", "method", "n_contributors"]]


def main():
    """콘텐츠 기반 모델을 학습·평가하고 results/content_based/<dataset>/에 저장한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["movielens", "books"], default="movielens")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--item-features", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=None)
    parser.add_argument(
        "--ridge-alphas",
        default="1,10,100",
        help="--tune일 때 탐색할 ridge_alpha 후보. 쉼표로 구분.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="train으로 맞추고 val MAE로 ridge_alpha를 고른다. test로는 고르지 않는다.",
    )
    args = parser.parse_args()

    if args.dataset == "books":
        data_dir = args.data_dir or (ROOT / "data" / "raw" / "goodbooks-10k")
        item_features = args.item_features or (ROOT / "data" / "processed" / "book_content.npz")
        output_dir = args.output_dir or (ROOT / "results" / "content_based" / "goodbooks")
        movies_path = None
        users_path = None
        ratings = load_goodbooks_ratings(data_dir)
    else:
        data_dir = args.data_dir or (ROOT / "preprocessing_and_data" / "movielens-1m")
        item_features = args.item_features or (ROOT / "data" / "processed" / "movies_multihot_data.npz")
        output_dir = args.output_dir or (ROOT / "results" / "content_based" / "movielens")
        movies_path = data_dir / "movies.dat"
        users_path = data_dir / "users.dat"
        movies_path = movies_path if movies_path.exists() else None
        users_path = users_path if users_path.exists() else None
        ratings = load_movielens_ratings(data_dir)

    train, val, test = split_by_user_three(
        ratings, val_size=args.val_size, test_size=args.test_size, random_state=args.seed
    )
    model_kwargs = {
        "item_features_path": item_features,
        "movies_path": movies_path,
        "users_path": users_path,
    }

    val_table = []
    best_val_mae = None
    if args.tune:
        if args.dataset == "books":
            raise ValueError("Goodbooks에서는 ridge_alpha를 다시 고르지 않습니다. MovieLens --tune 결과를 --ridge-alpha로 넘기세요.")
        alphas = parse_alphas(args.ridge_alphas)
        ridge_alpha, best_val_mae, val_table = select_ridge_alpha(train, val, alphas, model_kwargs)
        fit_frame = pd.concat([train, val], ignore_index=True)
    else:
        ridge_alpha = 10.0 if args.ridge_alpha is None else float(args.ridge_alpha)
        fit_frame = pd.concat([train, val], ignore_index=True)

    model = ContentBasedCF(ridge_alpha=ridge_alpha, **model_kwargs).fit(fit_frame)
    metrics, predictions = evaluate(model, test)
    recommendations = build_recommendations(model, args.user_id, args.top_n)
    summary = save_run(
        output_dir,
        model,
        train,
        val,
        test,
        metrics,
        predictions,
        recommendations,
        {
            "dataset": args.dataset,
            "seed": args.seed,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "ridge_alpha": ridge_alpha,
            "best_ridge_alpha": ridge_alpha if args.tune else None,
            "best_val_mae": best_val_mae,
            "val_mae_by_alpha": val_table,
            "tuned": bool(args.tune),
            "movies_path": str(movies_path) if movies_path else None,
            "users_path": str(users_path) if users_path else None,
            "item_features": str(item_features),
            "user_id": args.user_id,
        },
    )
    print(recommendations.to_string(index=False))
    print(json.dumps({"metrics": metrics, "ridge_alpha": ridge_alpha, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
