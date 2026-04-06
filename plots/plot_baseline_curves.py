from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


# 项目根目录：.../thesis-safe-activation
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str) -> Path:
    """
    将相对路径解析为相对于项目根目录的绝对路径。
    """
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


def extract_history(metrics: Dict[str, Any]) -> Dict[str, List[float]]:
    if "history" not in metrics:
        raise KeyError("Metrics file does not contain 'history'.")

    history = metrics["history"]
    required_keys = ["train_loss", "val_accuracy"]

    for key in required_keys:
        if key not in history:
            raise KeyError(f"Metrics file missing history key: {key}")

    train_loss = history["train_loss"]
    val_accuracy = history["val_accuracy"]

    if len(train_loss) != len(val_accuracy):
        raise ValueError(
            f"Inconsistent history length: "
            f"train_loss={len(train_loss)}, val_accuracy={len(val_accuracy)}"
        )

    if len(train_loss) == 0:
        raise ValueError("History is empty.")

    return {
        "epochs": list(range(1, len(train_loss) + 1)),
        "train_loss": train_loss,
        "val_accuracy": val_accuracy,
    }


def plot_train_loss(epochs: List[int], train_loss: List[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", linewidth=2)
    plt.title("Baseline CNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_val_accuracy(epochs: List[int], val_accuracy: List[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_accuracy, marker="o", linewidth=2)
    plt.title("Baseline CNN Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_combined(
    epochs: List[int],
    train_loss: List[float],
    val_accuracy: List[float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    axes[0].plot(epochs, train_loss, marker="o", linewidth=2)
    axes[0].set_title("Baseline CNN Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_accuracy, marker="o", linewidth=2)
    axes[1].set_title("Baseline CNN Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_summary(
    epochs: List[int],
    train_loss: List[float],
    val_accuracy: List[float],
    metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    summary = {
        "input_metrics": str(metrics.get("_input_path", "")),
        "model_name": metrics.get("model_name", "Baseline CNN"),
        "experiment_name": metrics.get("experiment_name", "train_baseline"),
        "num_epochs": len(epochs),
        "final_train_loss": train_loss[-1],
        "final_val_accuracy": val_accuracy[-1],
        "best_val_accuracy": max(val_accuracy),
    }

    if "summary" in metrics and isinstance(metrics["summary"], dict):
        summary["reported_best_val_accuracy"] = metrics["summary"].get("best_val_accuracy")
        summary["final_test_accuracy"] = metrics["summary"].get("final_test_accuracy")
        summary["total_training_seconds"] = metrics["summary"].get("total_training_seconds")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Baseline CNN training loss and validation accuracy curves."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/logs/exp1_baseline/train_baseline_metrics.json",
        help="Path to baseline metrics json.",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default="outputs/figures",
        help="Directory to save figures.",
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
    metrics["_input_path"] = str(input_path)

    history = extract_history(metrics)
    epochs = history["epochs"]
    train_loss = history["train_loss"]
    val_accuracy = history["val_accuracy"]

    train_loss_path = figure_dir / "baseline_training_loss.png"
    val_accuracy_path = figure_dir / "baseline_validation_accuracy.png"
    combined_path = figure_dir / "baseline_training_curves_combined.png"
    summary_path = table_dir / "baseline_curve_summary.json"

    plot_train_loss(epochs, train_loss, train_loss_path)
    plot_val_accuracy(epochs, val_accuracy, val_accuracy_path)
    plot_combined(epochs, train_loss, val_accuracy, combined_path)
    save_summary(epochs, train_loss, val_accuracy, metrics, summary_path)

    print("Baseline curve plotting finished.")
    print(f"Input metrics: {input_path}")
    print(f"Saved training loss figure: {train_loss_path}")
    print(f"Saved validation accuracy figure: {val_accuracy_path}")
    print(f"Saved combined figure: {combined_path}")
    print(f"Saved summary json: {summary_path}")


if __name__ == "__main__":
    main()