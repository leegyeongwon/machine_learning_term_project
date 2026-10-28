"""MovieLens 1M 로딩, 학습/평가, 추천 및 결과 저장용 CLI (콘텐츠 기반)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.content_based import ContentBasedCF


def load_ratings(data_dir):
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


def main():
    """콘텐츠 기반 모델을 학습·평가하고 results/content_based/에 저장한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "preprocessing_and_data" / "movielens-1m",
    )
    parser.add_argument(
        "--item-features",
        type=Path,
        default=ROOT / "data" / "processed" / "movies_multihot_data.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "content_based")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    movies_path = args.data_dir / "movies.dat"
    users_path = args.data_dir / "users.dat"
    ratings = load_ratings(args.data_dir)
    train, test = split_by_user(ratings, args.test_size, args.seed)
    model = ContentBasedCF(
        item_features_path=args.item_features,
        movies_path=movies_path if movies_path.exists() else None,
        users_path=users_path if users_path.exists() else None,
        ridge_alpha=args.ridge_alpha,
    ).fit(train)
    metrics, predictions = evaluate(model, test)
    recommendations = model.recommend(args.user_id, args.top_n)
    title_map = dict(zip(model.item_ids_.astype(int), model.titles_))
    recommendations["title"] = recommendations["item_id"].map(lambda i: title_map.get(int(i), ""))
    recommendations = recommendations[
        ["item_id", "title", "predicted_rating", "method", "n_contributors"]
    ]

    metrics_path = args.output_dir / "metrics.csv"
    rec_path = args.output_dir / f"recommendations_user_{args.user_id}.csv"
    pred_path = args.output_dir / "predictions.csv"
    summary_path = args.output_dir / "run_summary.json"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    recommendations.to_csv(rec_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(pred_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "n_train": len(train),
                "n_test": len(test),
                "seed": args.seed,
                "test_size": args.test_size,
                "ridge_alpha": args.ridge_alpha,
                "movies_path": str(movies_path) if movies_path.exists() else None,
                "users_path": str(users_path) if users_path.exists() else None,
                "item_features": str(args.item_features),
                "used_demographics": bool(model.has_demo_),
                "n_features": int(getattr(model.ridge_, "n_features_in_", 0)),
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(recommendations.to_string(index=False))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
