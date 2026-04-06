from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA = {
    "degree_ablation": [
        {
            "label": "deg2",
            "best_val_accuracy": 0.9835,
            "final_test_accuracy": 0.9867,
            "final_test_loss": 0.04179073411840945,
            "total_training_seconds": 61.184544299962,
            "inference_seconds_per_batch": 0.005125074996612966,
        },
        {
            "label": "deg4",
            "best_val_accuracy": 0.9863333333333333,
            "final_test_accuracy": 0.9870,
            "final_test_loss": 0.03908233933374286,
            "total_training_seconds": 66.5906679998152,
            "inference_seconds_per_batch": 0.005959550000261516,
        },
        {
            "label": "deg6",
            "best_val_accuracy": 0.9860,
            "final_test_accuracy": 0.9880,
            "final_test_loss": 0.03681513642184436,
            "total_training_seconds": 94.33730349992402,
            "inference_seconds_per_batch": 0.006683695001993328,
        },
    ],
    "interval_ablation": [
        {
            "label": "int[-2,2]",
            "best_val_accuracy": 0.9883333333333333,
            "final_test_accuracy": 0.9895,
            "final_test_loss": 0.03251980026997626,
            "total_training_seconds": 67.780756900087,
            "inference_seconds_per_batch": 0.006446284987032413,
        },
        {
            "label": "int[-3,3]",
            "best_val_accuracy": 0.9890,
            "final_test_accuracy": 0.9884,
            "final_test_loss": 0.03486963073192164,
            "total_training_seconds": 76.59119439986534,
            "inference_seconds_per_batch": 0.006121110008098185,
        },
        {
            "label": "int[-4,4]",
            "best_val_accuracy": 0.9851666666666666,
            "final_test_accuracy": 0.9859,
            "final_test_loss": 0.043682854986190796,
            "total_training_seconds": 81.92908070003614,
            "inference_seconds_per_batch": 0.006777865008916706,
        },
    ],
}


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
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def validate_items(items: List[Dict[str, Any]], group_name: str) -> None:
    required_keys = {
        "label",
        "best_val_accuracy",
        "final_test_accuracy",
        "final_test_loss",
        "total_training_seconds",
        "inference_seconds_per_batch",
    }
    for idx, item in enumerate(items):
        missing = required_keys - set(item.keys())
        if missing:
            raise KeyError(f"{group_name}[{idx}] missing keys: {sorted(missing)}")


def annotate_points(ax, xs, ys, fmt: str = "{:.4f}", y_offset_ratio: float = 0.03) -> None:
    if not ys:
        return
    y_min = min(ys)
    y_max = max(ys)
    delta = max((y_max - y_min) * y_offset_ratio, 0.001)
    for x, y in zip(xs, ys):
        ax.text(x, y + delta, fmt.format(y), ha="center", va="bottom", fontsize=8)


def plot_accuracy_panel(ax, items: List[Dict[str, Any]], title: str) -> None:
    labels = [item["label"] for item in items]
    xs = list(range(len(labels)))
    best_vals = [float(item["best_val_accuracy"]) for item in items]
    test_vals = [float(item["final_test_accuracy"]) for item in items]

    ax.plot(xs, best_vals, marker="o", linewidth=2, label="Best Val Acc")
    ax.plot(xs, test_vals, marker="o", linewidth=2, label="Final Test Acc")
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    all_vals = best_vals + test_vals
    ax.set_ylim(min(all_vals) - 0.003, max(all_vals) + 0.003)

    annotate_points(ax, xs, best_vals)
    annotate_points(ax, xs, test_vals)


def plot_efficiency_panel(ax, items: List[Dict[str, Any]], title: str) -> None:
    labels = [item["label"] for item in items]
    xs = list(range(len(labels)))
    train_times = [float(item["total_training_seconds"]) for item in items]
    infer_times = [float(item["inference_seconds_per_batch"]) for item in items]

    ax.plot(xs, train_times, marker="o", linewidth=2, label="Train Time (s)")
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Training Seconds")
    ax.grid(True, alpha=0.3)

    annotate_points(ax, xs, train_times, fmt="{:.2f}", y_offset_ratio=0.02)

    ax2 = ax.twinx()
    ax2.plot(xs, infer_times, marker="s", linewidth=2, linestyle="--", label="Infer / Batch (s)")
    ax2.set_ylabel("Inference Seconds / Batch")

    infer_offset = max((max(infer_times) - min(infer_times)) * 0.15, 0.00005)
    for x, y in zip(xs, infer_times):
        ax2.text(x, y + infer_offset, f"{y:.4f}", ha="center", va="bottom", fontsize=8)

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")


def build_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    degree_best_test = max(data["degree_ablation"], key=lambda x: x["final_test_accuracy"])
    interval_best_test = max(data["interval_ablation"], key=lambda x: x["final_test_accuracy"])

    return {
        "degree_ablation": data["degree_ablation"],
        "interval_ablation": data["interval_ablation"],
        "recommended_default": {
            "degree_label": "deg4",
            "interval_label": "int[-3,3]",
            "reason": "balanced accuracy-efficiency tradeoff",
        },
        "best_test_accuracy": {
            "degree_ablation": {
                "label": degree_best_test["label"],
                "value": degree_best_test["final_test_accuracy"],
            },
            "interval_ablation": {
                "label": interval_best_test["label"],
                "value": interval_best_test["final_test_accuracy"],
            },
        },
    }


def plot_figure(data: Dict[str, Any], output_path: Path) -> None:
    degree_items = data["degree_ablation"]
    interval_items = data["interval_ablation"]

    validate_items(degree_items, "degree_ablation")
    validate_items(interval_items, "interval_ablation")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    plot_accuracy_panel(axes[0, 0], degree_items, "Degree Ablation: Accuracy")
    plot_efficiency_panel(axes[0, 1], degree_items, "Degree Ablation: Efficiency")
    plot_accuracy_panel(axes[1, 0], interval_items, "Interval Ablation: Accuracy")
    plot_efficiency_panel(axes[1, 1], interval_items, "Interval Ablation: Efficiency")

    fig.suptitle("ReLU Ablation Results", fontsize=14)
    fig.text(
        0.98,
        0.01,
        "Default recommended config: deg4 + int[-3,3]",
        ha="right",
        va="bottom",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot ReLU ablation result figure.")
    parser.add_argument(
        "--data-json",
        type=str,
        default="",
        help="Optional custom JSON path. If omitted, built-in experiment-6 values are used.",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default="outputs/figures",
        help="Directory to save the figure.",
    )
    parser.add_argument(
        "--table-dir",
        type=str,
        default="outputs/tables",
        help="Directory to save the summary JSON.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    figure_dir = ensure_dir(resolve_path(args.figure_dir))
    table_dir = ensure_dir(resolve_path(args.table_dir))

    if args.data_json:
        data_path = resolve_path(args.data_json)
        data = load_json(data_path)
    else:
        data = DEFAULT_DATA

    figure_path = figure_dir / "relu_ablation_result.png"
    summary_path = table_dir / "relu_ablation_result_summary.json"

    plot_figure(data, figure_path)
    save_json(build_summary(data), summary_path)

    print("Plot finished.")
    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()