"""공통 8:1:1 Train으로 Content 학습, Validation으로 Ridge alpha 선택.

기존 run_content_based.py는 수정하지 않는다.
Test는 평가하지 않으며 Train+Validation 재학습도 하지 않는다.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.common_item_data import assert_identical_splits, source_splitter
from src.content_based import ContentBasedCF
from scripts.run_content_based import evaluate, parse_alphas


CONTENT_PREPROCESS = ROOT / "preprocessing_and_data" / "movielens-1m" / "preprocess_content_based.py"
MOVIELENS_DIR = ROOT / "preprocessing_and_data" / "movielens-1m"


def sha256(path):
    """파일 SHA-256 반환."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path, value):
    """결과를 UTF-8 JSON으로 저장."""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def triple_fingerprint(frame):
    """행 순서에 독립적인 평점 triple SHA-256 반환."""
    text = (
        frame[["user_id", "item_id", "rating"]]
        .sort_values(["user_id", "item_id"])
        .to_csv(index=False, lineterminator="\n")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choose_best_alpha(metrics):
    """Validation MAE, RMSE, 작은 alpha 순으로 최적 규제 세기를 고른다."""
    return float(metrics.sort_values(["mae", "rmse", "ridge_alpha"]).iloc[0]["ridge_alpha"])


def validate_hybrid_predictions(predictions, validation):
    """Validation 일대일 대응, 중복·누락, 평점 범위 검증."""
    keys = ["user_id", "item_id"]
    if predictions.duplicated(keys).any() or len(predictions) != len(validation):
        raise ValueError("Hybrid 예측에 중복 또는 누락이 있습니다.")
    paired = validation.merge(
        predictions,
        on=keys,
        how="outer",
        suffixes=("_actual", "_saved"),
        indicator=True,
        validate="one_to_one",
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Validation ID가 다릅니다.")
    np.testing.assert_array_equal(paired.rating_actual, paired.rating_saved)
    values = predictions.predicted_rating.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 1) & (values <= 5)).all():
        raise ValueError("예측은 유한한 1~5점이어야 합니다.")
    if not predictions.n_contributors.ge(0).all():
        raise ValueError("잘못된 기여 특성 수입니다.")
    allowed = {"content", "user_mean", "item_mean", "global_mean"}
    if not set(predictions.method).issubset(allowed):
        raise ValueError(f"알 수 없는 method가 있습니다: {sorted(set(predictions.method) - allowed)}")
    fallback = predictions.method.ne("content")
    if fallback.any() and not predictions.loc[fallback, "n_contributors"].eq(0).all():
        raise ValueError("대체 예측의 기여 수는 0이어야 합니다.")
    return {
        "rows": len(predictions),
        "one_prediction_per_validation_pair": True,
        "original_ids_and_ratings_match": True,
        "finite_and_in_1_to_5": True,
        "method_contributors_consistent": True,
    }


def replay_source_split(ratings_path, source_path, parts):
    """전처리 원문의 split_ratings로 NPZ 분할을 재현한다."""
    raw = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        encoding="ISO-8859-1",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    raw = raw.sort_values(["user_id", "movie_id"]).reset_index(drop=True)
    splitter, seed = source_splitter(source_path)
    if seed != 42:
        raise ValueError("공통 seed가 42가 아닙니다.")
    expected = splitter(raw)
    triple = ["user_id", "item_id", "rating"]
    for name, frame in zip(parts, expected):
        expected_frame = frame.rename(columns={"movie_id": "item_id"})[triple]
        pd.testing.assert_frame_equal(parts[name], expected_frame)
    return {"source_split_exact_order_match": True, "random_state": seed, "raw_count": len(raw)}


def assert_train_only(model, train, validation):
    """Validation 쌍이 학습 seen 집합에 없는지 확인한다."""
    seen_pairs = set(zip(train.user_id.astype(int), train.item_id.astype(int)))
    val_pairs = set(zip(validation.user_id.astype(int), validation.item_id.astype(int)))
    overlap = seen_pairs & val_pairs
    if overlap:
        raise ValueError(f"Validation이 학습 평점에 있습니다: {len(overlap)}")
    leaked = [
        (int(user_id), int(item_id))
        for user_id, items in model.seen_.items()
        for item_id in items
        if (int(user_id), int(item_id)) in val_pairs
    ]
    if leaked:
        raise ValueError(f"Validation이 모델 seen에 있습니다: {len(leaked)}")


def main():
    """공통 Train 학습 및 Validation 기반 Ridge 선택·예측 저장."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "content_based" / "common_split_811",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=ROOT / "data" / "processed" / "user_profiles_data.npz",
    )
    parser.add_argument(
        "--item-npz",
        type=Path,
        default=ROOT / "data" / "processed" / "item_user_data.npz",
    )
    parser.add_argument(
        "--item-features",
        type=Path,
        default=ROOT / "data" / "processed" / "movies_multihot_data.npz",
    )
    parser.add_argument("--movies-path", type=Path, default=MOVIELENS_DIR / "movies.dat")
    parser.add_argument("--users-path", type=Path, default=MOVIELENS_DIR / "users.dat")
    parser.add_argument("--ratings-path", type=Path, default=MOVIELENS_DIR / "ratings.dat")
    parser.add_argument("--ridge-alphas", default="1,10,100")
    args = parser.parse_args()

    out = args.output_dir.resolve()
    npz_path = args.npz.resolve()
    item_npz = args.item_npz.resolve()
    if (out / "validation_metrics.csv").exists():
        parser.error("기존 결과를 보존합니다. 새 --output-dir을 지정하세요.")
    if not npz_path.exists():
        parser.error(f"콘텐츠 NPZ가 없습니다: {npz_path}")
    if not item_npz.exists():
        parser.error(f"아이템 NPZ가 없습니다: {item_npz}")

    out.mkdir(parents=True, exist_ok=True)
    if not (out / ".gitignore").exists():
        (out / ".gitignore").write_text("*\n", encoding="utf-8")

    started = time.perf_counter()
    parts = assert_identical_splits(npz_path, item_npz)
    split_checks = {
        "content_item_split_identical": True,
        "train_count": len(parts["train"]),
        "validation_count": len(parts["validation"]),
        "test_count": len(parts["test"]),
    }
    if args.ratings_path.exists():
        split_checks.update(replay_source_split(args.ratings_path, CONTENT_PREPROCESS, parts))
    save_json(out / "data_validation_report.json", split_checks)
    print(json.dumps(split_checks, ensure_ascii=False, indent=2), flush=True)

    train, validation = parts["train"], parts["validation"]
    del parts

    movies_path = args.movies_path if args.movies_path.exists() else None
    users_path = args.users_path if args.users_path.exists() else None
    model_kwargs = {
        "item_features_path": args.item_features,
        "movies_path": movies_path,
        "users_path": users_path,
    }

    rows = []
    predictions_by_alpha = {}
    for alpha in parse_alphas(args.ridge_alphas):
        tick = time.perf_counter()
        model = ContentBasedCF(ridge_alpha=alpha, **model_kwargs).fit(train)
        fit_seconds = time.perf_counter() - tick
        assert_train_only(model, train, validation)
        tick = time.perf_counter()
        metrics, predictions = evaluate(model, validation)
        evaluation_seconds = time.perf_counter() - tick
        metrics["n_validation"] = metrics.pop("n_test")
        metrics.update(
            ridge_alpha=float(alpha),
            fit_seconds=fit_seconds,
            evaluation_seconds=evaluation_seconds,
            fallback_methods=predictions.loc[predictions.method.ne("content"), "method"]
            .value_counts()
            .to_dict(),
        )
        validate_hybrid_predictions(predictions, validation)
        alpha_tag = str(alpha).replace(".", "p")
        predictions.to_csv(out / f"validation_predictions_alpha_{alpha_tag}.csv", index=False)
        predictions_by_alpha[float(alpha)] = predictions
        rows.append(metrics)
        print(json.dumps({k: v for k, v in metrics.items() if k != "fallback_methods"}, ensure_ascii=False), flush=True)
        del model

    table = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "fallback_methods"} for row in rows]
    )
    table.to_csv(out / "validation_metrics.csv", index=False)
    save_json(out / "validation_metrics.json", rows)
    best_alpha = choose_best_alpha(table)
    selected = predictions_by_alpha[best_alpha]
    selected.to_csv(out / "hybrid_ready_validation_predictions.csv", index=False)
    saved = pd.read_csv(out / "hybrid_ready_validation_predictions.csv")
    hybrid_checks = validate_hybrid_predictions(saved, validation)
    error = saved.predicted_rating - saved.rating
    best = table.set_index("ridge_alpha").loc[best_alpha]
    np.testing.assert_allclose(
        [error.abs().mean(), np.sqrt(np.mean(error ** 2))],
        [best.mae, best.rmse],
        rtol=1e-12,
    )
    save_json(
        out / "hybrid_contract.json",
        {
            "best_ridge_alpha": best_alpha,
            "selection": "Validation MAE, then RMSE, then smaller ridge_alpha",
            "model": "existing raw-rating ContentBasedCF",
            "fit_data": "common Train only",
            "train_plus_validation_refit": False,
            "test_evaluated": False,
            "merge_keys": ["user_id", "item_id"],
            "rating_scale": [1, 5],
            "method_values": sorted(saved.method.unique().tolist()),
            "train_triple_sha256": triple_fingerprint(train),
            "validation_triple_sha256": triple_fingerprint(validation),
            "npz_sha256": sha256(npz_path),
            "item_npz_sha256": sha256(item_npz),
            "csv_sha256": sha256(out / "hybrid_ready_validation_predictions.csv"),
            "checks": hybrid_checks,
            "content_split_verified": True,
        },
    )
    os.environ["MPLCONFIGDIR"] = str(out / "matplotlib_cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table.ridge_alpha, table.mae, "o-", label="Validation MAE")
    ax.plot(table.ridge_alpha, table.rmse, "o-", label="Validation RMSE")
    ax.set(xlabel="ridge_alpha", ylabel="Rating error", title="Team common global 80/10/10 split")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "validation_metrics.png", dpi=160)
    plt.close(fig)
    save_json(
        out / "run_summary.json",
        {
            "npz_sha256": sha256(npz_path),
            "item_npz_sha256": sha256(item_npz),
            "model_sha256": sha256(ROOT / "src" / "content_based.py"),
            "best_ridge_alpha": best_alpha,
            "test_evaluated": False,
            "used_demographics": bool(users_path),
            "movies_path": str(movies_path) if movies_path else None,
            "users_path": str(users_path) if users_path else None,
            "item_features": str(args.item_features),
            "total_seconds": time.perf_counter() - started,
        },
    )
    print(table.to_string(index=False), flush=True)
    print(
        f"Best ridge_alpha={best_alpha}; Hybrid rows={len(saved)}; Test not evaluated.",
        flush=True,
    )


if __name__ == "__main__":
    main()
