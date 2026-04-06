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


def extract_core_info(metrics: Dict[str, Any]) -> Dict[str, Any]:
    train_loss = get_history(metrics, "train_loss")
    val_accuracy = get_history(metrics, "val_accuracy")

    final_val_accuracy = get_summary_value(metrics, "final_val_accuracy")
    best_val_accuracy = get_summary_value(metrics, "best_val_accuracy")
    final_test_accuracy = get_summary_value(metrics, "final_test_accuracy")

    total_training_seconds = get_summary_value(metrics, "total_training_seconds")
    average_epoch_seconds = get_summary_value(metrics, "average_epoch_seconds")
    inference_seconds_per_batch = get_summary_value(metrics, "inference_seconds_per_batch")

    if final_val_accuracy is None:
        final_val_accuracy = val_accuracy[-1]
    if best_val_accuracy is None:
        best_val_accuracy = max(val_accuracy)

    required = {
        "final_test_accuracy": final_test_accuracy,
        "total_training_seconds": total_training_seconds,
        "average_epoch_seconds": average_epoch_seconds,
    }
    for key, value in required.items():
        if value is None:
            raise KeyError(f"Cannot find '{key}' in metrics summary.")

    return {
        "epochs": list(range(1, len(train_loss) + 1)),
        "train_loss": [float(x) for x in train_loss],
        "val_accuracy": [float(x) for x in val_accuracy],
        "final_val_accuracy": float(final_val_accuracy),
        "best_val_accuracy": float(best_val_accuracy),
        "final_test_accuracy": float(final_test_accuracy),
        "total_training_seconds": float(total_training_seconds),
        "average_epoch_seconds": float(average_epoch_seconds),
        "inference_seconds_per_batch": (
            None if inference_seconds_per_batch is None else float(inference_seconds_per_batch)
        ),
    }


def plot_result(info: Dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1. 训练损失曲线
    ax = axes[0, 0]
    ax.plot(info["epochs"], info["train_loss"], marker="o", linewidth=2)
    ax.set_title("ApproxSigmoid Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.grid(True, alpha=0.3)

    # 2. 验证精度曲线
    ax = axes[0, 1]
    ax.plot(info["epochs"], info["val_accuracy"], marker="o", linewidth=2)
    ax.set_title("ApproxSigmoid Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.grid(True, alpha=0.3)

    # 3. 精度指标柱状图
    ax = axes[1, 0]
    acc_names = ["Best Val", "Final Val", "Final Test"]
    acc_values = [
        info["best_val_accuracy"],
        info["final_val_accuracy"],
        info["final_test_accuracy"],
    ]
    bars = ax.bar(acc_names, acc_values)
    ax.set_title("Accuracy Summary")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(min(acc_values) - 0.02, max(acc_values) + 0.02)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, acc_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.002,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    # 4. 时间指标柱状图
    ax = axes[1, 1]
    time_names = ["Total Train", "Avg Epoch"]
    time_values = [
        info["total_training_seconds"],
        info["average_epoch_seconds"],
    ]
    bars = ax.bar(time_names, time_values)
    ax.set_title("Training Time Summary")
    ax.set_ylabel("Seconds")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, time_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(time_values) * 0.02,
            f"{value:.2f}s",
            ha="center",
            va="bottom",
        )

    if info["inference_seconds_per_batch"] is not None:
        note_text = f"Inference / batch: {info['inference_seconds_per_batch']:.6f}s"
    else:
        note_text = "Inference / batch: N/A"

    ax.text(
        0.98,
        0.05,
        note_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    fig.suptitle("ApproxSigmoid Model Results", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_summary(info: Dict[str, Any], output_path: Path) -> None:
    summary = {
        "best_val_accuracy": info["best_val_accuracy"],
        "final_val_accuracy": info["final_val_accuracy"],
        "final_test_accuracy": info["final_test_accuracy"],
        "total_training_seconds": info["total_training_seconds"],
        "average_epoch_seconds": info["average_epoch_seconds"],
        "inference_seconds_per_batch": info["inference_seconds_per_batch"],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot ApproxSigmoid model result figure."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/logs/exp4_approx_sigmoid/train_approx_sigmoid_metrics.json",
        help="Path to approx sigmoid metrics json.",
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

    input_path = resolve_path(args.input)
    figure_dir = ensure_dir(resolve_path(args.figure_dir))
    table_dir = ensure_dir(resolve_path(args.table_dir))

    metrics = load_json(input_path)
    info = extract_core_info(metrics)

    figure_path = figure_dir / "approx_sigmoid_result.png"
    summary_path = table_dir / "approx_sigmoid_result_summary.json"

    plot_result(info, figure_path)
    save_summary(info, summary_path)

    print("Plot finished.")
    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()