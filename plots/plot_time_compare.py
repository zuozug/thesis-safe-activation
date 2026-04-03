from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_inputs(items: List[str]) -> List[Tuple[str, Path]]:
    """
    输入格式示例：
        Baseline=outputs/logs/train_baseline_metrics.json
        ApproxReLU=outputs/logs/train_approx_relu_metrics.json
        ApproxGELU=outputs/logs/train_approx_gelu_metrics.json
    """
    parsed: List[Tuple[str, Path]] = []

    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid input item: {item}. Expected format: Label=path/to/file.json"
            )

        label, path_text = item.split("=", 1)
        label = label.strip()
        path = Path(path_text.strip())

        if not label:
            raise ValueError(f"Invalid label in input item: {item}")

        parsed.append((label, path))

    if not parsed:
        raise ValueError("No valid input metrics files provided.")

    return parsed


def extract_time_metrics(metrics: dict, label: str) -> dict:
    summary = metrics.get("summary")
    inference = metrics.get("inference")

    if summary is None:
        raise KeyError(f"Metrics file for {label} does not contain 'summary'.")
    if inference is None:
        raise KeyError(f"Metrics file for {label} does not contain 'inference'.")

    required_summary_keys = [
        "total_training_seconds",
        "average_epoch_seconds",
        "final_test_accuracy",
    ]
    required_inference_keys = [
        "seconds_per_batch",
        "seconds_per_sample",
        "num_batches",
        "num_samples",
    ]

    for key in required_summary_keys:
        if key not in summary:
            raise KeyError(f"Metrics file for {label} missing summary key: {key}")

    for key in required_inference_keys:
        if key not in inference:
            raise KeyError(f"Metrics file for {label} missing inference key: {key}")

    return {
        "label": label,
        "model_name": metrics.get("model_name", label),
        "experiment_name": metrics.get("experiment_name", label),
        "total_training_seconds": float(summary["total_training_seconds"]),
        "average_epoch_seconds": float(summary["average_epoch_seconds"]),
        "final_test_accuracy": float(summary["final_test_accuracy"]),
        "seconds_per_batch": float(inference["seconds_per_batch"]),
        "seconds_per_sample": float(inference["seconds_per_sample"]),
        "num_batches": int(inference["num_batches"]),
        "num_samples": int(inference["num_samples"]),
    }


def plot_training_time(rows: List[dict], output_path: Path) -> None:
    labels = [row["label"] for row in rows]
    values = [row["total_training_seconds"] for row in rows]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title("Total Training Time Comparison")
    plt.xlabel("Model")
    plt.ylabel("Seconds")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_inference_time(rows: List[dict], output_path: Path) -> None:
    labels = [row["label"] for row in rows]
    values = [row["seconds_per_sample"] for row in rows]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title("Inference Time Per Sample Comparison")
    plt.xlabel("Model")
    plt.ylabel("Seconds per sample")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_summary_rows(rows: List[dict]) -> List[dict]:
    return [
        {
            "label": row["label"],
            "model_name": row["model_name"],
            "experiment_name": row["experiment_name"],
            "final_test_accuracy": row["final_test_accuracy"],
            "total_training_seconds": row["total_training_seconds"],
            "average_epoch_seconds": row["average_epoch_seconds"],
            "seconds_per_batch": row["seconds_per_batch"],
            "seconds_per_sample": row["seconds_per_sample"],
            "num_batches_used_for_inference": row["num_batches"],
            "num_samples_used_for_inference": row["num_samples"],
        }
        for row in rows
    ]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot training time and inference time comparison figures."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "Baseline=outputs/logs/train_baseline_metrics.json",
            "ApproxReLU=outputs/logs/train_approx_relu_metrics.json",
            "ApproxGELU=outputs/logs/train_approx_gelu_metrics.json",
        ],
        help="One or more items in the form Label=path/to/metrics.json",
    )
    parser.add_argument("--figure-dir", type=str, default="outputs/figures")
    parser.add_argument("--table-dir", type=str, default="outputs/tables")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    figure_dir = ensure_dir(args.figure_dir)
    table_dir = ensure_dir(args.table_dir)

    parsed_inputs = parse_inputs(args.inputs)

    rows: List[dict] = []
    for label, path in parsed_inputs:
        metrics = load_json(path)
        rows.append(extract_time_metrics(metrics, label))

    training_time_path = figure_dir / "training_time_compare.png"
    inference_time_path = figure_dir / "inference_time_compare.png"
    summary_path = table_dir / "time_compare_summary.json"

    plot_training_time(rows, training_time_path)
    plot_inference_time(rows, inference_time_path)

    output = {
        "inputs": [
            {"label": label, "path": str(path)}
            for label, path in parsed_inputs
        ],
        "summary_rows": build_summary_rows(rows),
        "artifacts": {
            "training_time_figure": str(training_time_path),
            "inference_time_figure": str(inference_time_path),
            "summary_json": str(summary_path),
        },
    }
    save_json(output, summary_path)

    print("Time comparison plotting finished.")
    print(f"Saved training time figure:  {training_time_path}")
    print(f"Saved inference time figure: {inference_time_path}")
    print(f"Saved summary json:          {summary_path}")


if __name__ == "__main__":
    main()