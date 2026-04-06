from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_time_info(metrics: Dict[str, Any], label: str) -> Dict[str, Any]:
    summary = metrics.get("summary")
    if not isinstance(summary, dict):
        raise KeyError(f"{label}: missing 'summary' field.")

    inference = metrics.get("inference")
    if not isinstance(inference, dict):
        raise KeyError(f"{label}: missing 'inference' field.")

    total_training_seconds = summary.get("total_training_seconds")
    average_epoch_seconds = summary.get("average_epoch_seconds")
    inference_seconds_per_batch = inference.get("seconds_per_batch")
    inference_seconds_per_sample = inference.get("seconds_per_sample")

    required = {
        "summary.total_training_seconds": total_training_seconds,
        "summary.average_epoch_seconds": average_epoch_seconds,
        "inference.seconds_per_batch": inference_seconds_per_batch,
        "inference.seconds_per_sample": inference_seconds_per_sample,
    }
    for key, value in required.items():
        if value is None:
            raise KeyError(f"{label}: cannot find '{key}' in metrics file.")

    return {
        "label": label,
        "total_training_seconds": float(total_training_seconds),
        "average_epoch_seconds": float(average_epoch_seconds),
        "inference_seconds_per_batch": float(inference_seconds_per_batch),
        "inference_seconds_per_sample": float(inference_seconds_per_sample),
    }


def annotate_bars(ax, bars, fmt: str, offset_ratio: float = 0.02) -> None:
    heights = [bar.get_height() for bar in bars]
    if not heights:
        return
    max_height = max(heights)
    offset = max(max_height * offset_ratio, 0.00001)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_training_time(ax, items: List[Dict[str, Any]]) -> None:
    labels = [item["label"] for item in items]
    xs = list(range(len(labels)))
    total_times = [item["total_training_seconds"] for item in items]
    avg_epoch_times = [item["average_epoch_seconds"] for item in items]

    width = 0.36
    bars1 = ax.bar([x - width / 2 for x in xs], total_times, width=width, label="Total Train (s)")
    bars2 = ax.bar([x + width / 2 for x in xs], avg_epoch_times, width=width, label="Avg Epoch (s)")

    ax.set_title("Training Time Comparison")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Seconds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    annotate_bars(ax, bars1, "{:.2f}")
    annotate_bars(ax, bars2, "{:.2f}")


def plot_inference_time(ax, items: List[Dict[str, Any]]) -> None:
    labels = [item["label"] for item in items]
    xs = list(range(len(labels)))
    per_batch = [item["inference_seconds_per_batch"] for item in items]
    per_sample = [item["inference_seconds_per_sample"] for item in items]

    ax.plot(xs, per_batch, marker="o", linewidth=2, label="Infer / Batch (s)")
    ax.set_title("Inference Time Comparison")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Seconds / Batch")
    ax.grid(True, alpha=0.3)

    y_offset = max((max(per_batch) - min(per_batch)) * 0.15, 0.00005)
    for x, y in zip(xs, per_batch):
        ax.text(x, y + y_offset, f"{y:.4f}", ha="center", va="bottom", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(xs, per_sample, marker="s", linewidth=2, linestyle="--", label="Infer / Sample (s)")
    ax2.set_ylabel("Seconds / Sample")

    y2_offset = max((max(per_sample) - min(per_sample)) * 0.15, 0.0000005)
    for x, y in zip(xs, per_sample):
        ax2.text(x, y + y2_offset, f"{y:.6f}", ha="center", va="bottom", fontsize=8)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")


def build_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "models": items,
        "fastest_training_total": min(items, key=lambda x: x["total_training_seconds"])["label"],
        "fastest_training_epoch": min(items, key=lambda x: x["average_epoch_seconds"])["label"],
        "fastest_inference_batch": min(items, key=lambda x: x["inference_seconds_per_batch"])["label"],
        "fastest_inference_sample": min(items, key=lambda x: x["inference_seconds_per_sample"])["label"],
    }


def save_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def plot_figure(items: List[Dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    plot_training_time(axes[0], items)
    plot_inference_time(axes[1], items)

    fig.suptitle("Main Model Time Cost Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot main model time cost comparison.")
    parser.add_argument(
        "--baseline",
        type=str,
        default="outputs/logs/exp1_baseline/train_baseline_metrics.json",
        help="Path to baseline metrics json.",
    )
    parser.add_argument(
        "--approx-relu",
        type=str,
        default="outputs/logs/exp2_approx_relu/train_approx_relu_metrics.json",
        help="Path to approx relu metrics json.",
    )
    parser.add_argument(
        "--approx-gelu",
        type=str,
        default="outputs/logs/exp3_approx_gelu/train_approx_gelu_metrics.json",
        help="Path to approx gelu metrics json.",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default="outputs/figures",
        help="Directory to save figure.",
    )
    parser.add_argument(
        "--table-dir",
        type=str,
        default="outputs/tables",
        help="Directory to save summary json.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    figure_dir = ensure_dir(resolve_path(args.figure_dir))
    table_dir = ensure_dir(resolve_path(args.table_dir))

    baseline_metrics = load_json(resolve_path(args.baseline))
    approx_relu_metrics = load_json(resolve_path(args.approx_relu))
    approx_gelu_metrics = load_json(resolve_path(args.approx_gelu))

    items = [
        extract_time_info(baseline_metrics, "Baseline"),
        extract_time_info(approx_relu_metrics, "ApproxReLU"),
        extract_time_info(approx_gelu_metrics, "ApproxGELU"),
    ]

    figure_path = figure_dir / "main_model_time_compare.png"
    summary_path = table_dir / "main_model_time_compare_summary.json"

    plot_figure(items, figure_path)
    save_json(build_summary(items), summary_path)

    print("Plot finished.")
    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()