from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def find_metrics_json(exp_path: Path) -> Path:
    """
    允许传入：
    1. 直接的 json 文件路径
    2. 实验目录路径
    """
    if exp_path.is_file():
        return exp_path

    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment path not found: {exp_path}")

    candidates = [
        "eval_softmax_metrics.json",
        "softmax_eval_metrics.json",
        "metrics.json",
        "summary.json",
        "results.json",
    ]

    for name in candidates:
        candidate = exp_path / name
        if candidate.exists():
            return candidate

    json_files = sorted(exp_path.glob("*.json"))
    if len(json_files) == 1:
        return json_files[0]
    if len(json_files) > 1:
        # 优先选名字里更像 softmax/eval/metrics 的
        keywords = ["softmax", "eval", "metric", "result", "summary"]
        scored: List[Tuple[int, Path]] = []
        for p in json_files:
            score = sum(1 for kw in keywords if kw in p.name.lower())
            scored.append((score, p))
        scored.sort(key=lambda x: (-x[0], x[1].name))
        return scored[0][1]

    raise FileNotFoundError(f"No json metrics file found in: {exp_path}")


def get_nested_value(data: Dict[str, Any], key: str) -> Any:
    if key in data:
        return data[key]
    summary = data.get("summary")
    if isinstance(summary, dict) and key in summary:
        return summary[key]
    return None


def get_first_available(data: Dict[str, Any], keys: List[str], required: bool = True) -> Any:
    for key in keys:
        value = get_nested_value(data, key)
        if value is not None:
            return value
    if required:
        raise KeyError(f"Cannot find any of keys: {keys}")
    return None


def parse_input_item(item: str) -> Tuple[str, str]:
    """
    形如：
    3阶[-4,4]=outputs/logs/exp5_softmax
    """
    if "=" not in item:
        raise ValueError(f"Invalid input item: {item}. Expected LABEL=PATH")
    label, path = item.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid input item: {item}. Expected LABEL=PATH")
    return label, path


def extract_metrics(label: str, metrics: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    probability_mae = get_first_available(metrics, ["probability_mae", "probability MAE"])
    probability_mse = get_first_available(metrics, ["probability_mse", "probability MSE"])
    probability_max_abs_error = get_first_available(
        metrics,
        ["probability_max_abs_error", "probability max abs error"],
    )
    accuracy_drop = get_first_available(metrics, ["accuracy_drop", "accuracy drop"])
    approx_accuracy = get_first_available(metrics, ["approx_accuracy", "approx accuracy"], required=False)
    exact_accuracy = get_first_available(metrics, ["exact_accuracy", "exact accuracy"], required=False)
    is_stable = get_first_available(metrics, ["is_stable", "is stable"], required=False)

    return {
        "label": label,
        "source_path": str(source_path),
        "probability_mae": float(probability_mae),
        "probability_mse": float(probability_mse),
        "probability_max_abs_error": float(probability_max_abs_error),
        "accuracy_drop": float(accuracy_drop),
        "approx_accuracy": None if approx_accuracy is None else float(approx_accuracy),
        "exact_accuracy": None if exact_accuracy is None else float(exact_accuracy),
        "is_stable": is_stable,
    }


def annotate_bars(ax, bars, fmt: str = "{:.4f}") -> None:
    max_height = max((bar.get_height() for bar in bars), default=0.0)
    offset = max(max_height * 0.01, 0.001)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_metric(ax, labels: List[str], values: List[float], title: str, ylabel: str) -> None:
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    annotate_bars(ax, bars)
    ax.tick_params(axis="x", rotation=15)


def plot_figure(items: List[Dict[str, Any]], output_path: Path) -> None:
    labels = [item["label"] for item in items]
    mae_values = [item["probability_mae"] for item in items]
    mse_values = [item["probability_mse"] for item in items]
    maxerr_values = [item["probability_max_abs_error"] for item in items]
    accdrop_values = [item["accuracy_drop"] for item in items]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    plot_metric(axes[0, 0], labels, mae_values, "Probability MAE", "MAE")
    plot_metric(axes[0, 1], labels, mse_values, "Probability MSE", "MSE")
    plot_metric(axes[1, 0], labels, maxerr_values, "Probability Max Abs Error", "Max Abs Error")
    plot_metric(axes[1, 1], labels, accdrop_values, "Accuracy Drop", "Accuracy Drop")

    stable_count = sum(1 for item in items if item["is_stable"] is True)
    note_text = f"Stable configs: {stable_count}/{len(items)}"
    fig.text(0.98, 0.01, note_text, ha="right", va="bottom", fontsize=9)

    fig.suptitle("Softmax Approximation Probability Distribution Error Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_summary(items: List[Dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Softmax approximation probability distribution error comparison."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "deg3[-4,4]=outputs/logs/exp5_softmax",
            "deg5[-4,4]=outputs/logs/exp5a_softmax_deg5",
            "deg5[-6,6]-exp=outputs/logs/exp5b_softmax_deg5_wide",
            "deg5[-6,6]-cheb=outputs/logs/exp5c_softmax_chebyshev",
            "deg5[-6,6]-ls=outputs/logs/exp5d_softmax_ls",
        ],
        help="Input items in LABEL=PATH form.",
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

    items: List[Dict[str, Any]] = []

    for raw_item in args.inputs:
        label, path_str = parse_input_item(raw_item)
        exp_path = resolve_path(path_str)
        json_path = find_metrics_json(exp_path)
        metrics = load_json(json_path)
        item = extract_metrics(label, metrics, json_path)
        items.append(item)

    figure_path = figure_dir / "softmax_error_comparison.png"
    summary_path = table_dir / "softmax_error_comparison_summary.json"

    plot_figure(items, figure_path)
    save_summary(items, summary_path)

    print("Plot finished.")
    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()