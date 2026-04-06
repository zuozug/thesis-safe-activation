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


def get_history(metrics: Dict[str, Any], key: str) -> List[float]:
    history = metrics.get("history")
    if not isinstance(history, dict):
        raise KeyError("Metrics file does not contain a valid 'history' field.")
    values = history.get(key)
    if not isinstance(values, list) or len(values) == 0:
        raise KeyError(f"Metrics file missing or empty history key: {key}")
    return values


def get_summary_value(metrics: Dict[str, Any], key: str, fallback: Any = None) -> Any:
    summary = metrics.get("summary", {})
    if isinstance(summary, dict) and key in summary:
        return summary[key]
    if key in metrics:
        return metrics[key]
    return fallback


def extract_core_info(metrics: Dict[str, Any], label: str) -> Dict[str, Any]:
    train_loss = get_history(metrics, "train_loss")
    val_accuracy = get_history(metrics, "val_accuracy")

    total_training_seconds = get_summary_value(metrics, "total_training_seconds")
    final_test_accuracy = get_summary_value(metrics, "final_test_accuracy")

    if total_training_seconds is None:
        total_training_seconds = get_summary_value(metrics, "training_seconds")
    if final_test_accuracy is None:
        final_test_accuracy = get_summary_value(metrics, "test_accuracy")

    if total_training_seconds is None:
        raise KeyError(f"{label}: cannot find total_training_seconds in metrics summary.")
    if final_test_accuracy is None:
        raise KeyError(f"{label}: cannot find final_test_accuracy in metrics summary.")

    return {
        "label": label,
        "epochs": list(range(1, len(train_loss) + 1)),
        "train_loss": train_loss,
        "val_accuracy": val_accuracy,
        "total_training_seconds": float(total_training_seconds),
        "final_test_accuracy": float(final_test_accuracy),
    }


def plot_comparison(baseline: Dict[str, Any], approx_relu: Dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1) 训练损失曲线
    ax = axes[0, 0]
    ax.plot(baseline["epochs"], baseline["train_loss"], marker="o", linewidth=2, label=baseline["label"])
    ax.plot(approx_relu["epochs"], approx_relu["train_loss"], marker="o", linewidth=2, label=approx_relu["label"])
    ax.set_title("Training Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2) 验证精度曲线
    ax = axes[0, 1]
    ax.plot(baseline["epochs"], baseline["val_accuracy"], marker="o", linewidth=2, label=baseline["label"])
    ax.plot(approx_relu["epochs"], approx_relu["val_accuracy"], marker="o", linewidth=2, label=approx_relu["label"])
    ax.set_title("Validation Accuracy Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3) 最终测试准确率柱状图
    ax = axes[1, 0]
    model_names = [baseline["label"], approx_relu["label"]]
    test_acc = [baseline["final_test_accuracy"], approx_relu["final_test_accuracy"]]
    bars = ax.bar(model_names, test_acc)
    ax.set_title("Final Test Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(min(test_acc) - 0.01, max(test_acc) + 0.01)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, test_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.0005,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    # 4) 总训练时间柱状图
    ax = axes[1, 1]
    train_times = [baseline["total_training_seconds"], approx_relu["total_training_seconds"]]
    bars = ax.bar(model_names, train_times)
    ax.set_title("Total Training Time")
    ax.set_ylabel("Seconds")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, train_times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(train_times) * 0.01,
            f"{value:.2f}s",
            ha="center",
            va="bottom",
        )

    fig.suptitle("Baseline vs ApproxReLU Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_summary(baseline: Dict[str, Any], approx_relu: Dict[str, Any], output_path: Path) -> None:
    summary = {
        "baseline": {
            "final_test_accuracy": baseline["final_test_accuracy"],
            "total_training_seconds": baseline["total_training_seconds"],
        },
        "approx_relu": {
            "final_test_accuracy": approx_relu["final_test_accuracy"],
            "total_training_seconds": approx_relu["total_training_seconds"],
        },
        "delta": {
            "test_accuracy_diff": approx_relu["final_test_accuracy"] - baseline["final_test_accuracy"],
            "training_time_diff_seconds": approx_relu["total_training_seconds"] - baseline["total_training_seconds"],
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Baseline vs ApproxReLU comparison figure."
    )
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

    baseline_path = resolve_path(args.baseline)
    approx_relu_path = resolve_path(args.approx_relu)
    figure_dir = ensure_dir(resolve_path(args.figure_dir))
    table_dir = ensure_dir(resolve_path(args.table_dir))

    baseline_metrics = load_json(baseline_path)
    approx_relu_metrics = load_json(approx_relu_path)

    baseline = extract_core_info(baseline_metrics, "Baseline")
    approx_relu = extract_core_info(approx_relu_metrics, "ApproxReLU")

    figure_path = figure_dir / "baseline_vs_approx_relu_comparison.png"
    summary_path = table_dir / "baseline_vs_approx_relu_summary.json"

    plot_comparison(baseline, approx_relu, figure_path)
    save_summary(baseline, approx_relu, summary_path)

    print("Plot finished.")
    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()