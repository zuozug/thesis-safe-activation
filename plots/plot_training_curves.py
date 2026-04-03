from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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
    输入格式：
        Baseline=outputs/logs/train_baseline_metrics.json
        ApproxReLU=outputs/logs/train_approx_relu_metrics.json
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


def extract_history(metrics: dict, label: str) -> dict:
    if "history" not in metrics:
        raise KeyError(f"Metrics file for {label} does not contain 'history'.")

    history = metrics["history"]

    required_keys = [
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    ]
    for key in required_keys:
        if key not in history:
            raise KeyError(f"Metrics file for {label} missing history key: {key}")

    lengths = {key: len(history[key]) for key in required_keys}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(
            f"Inconsistent history lengths for {label}: {lengths}"
        )

    num_epochs = next(iter(unique_lengths))
    if num_epochs == 0:
        raise ValueError(f"Empty history for {label}")

    return {
        "label": label,
        "epochs": list(range(1, num_epochs + 1)),
        "train_loss": history["train_loss"],
        "train_accuracy": history["train_accuracy"],
        "val_loss": history["val_loss"],
        "val_accuracy": history["val_accuracy"],
        "summary": metrics.get("summary", {}),
        "model_name": metrics.get("model_name", label),
        "experiment_name": metrics.get("experiment_name", label),
    }


def plot_train_loss(
    histories: List[dict],
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    for item in histories:
        plt.plot(
            item["epochs"],
            item["train_loss"],
            label=item["label"],
            linewidth=2,
        )

    plt.title("Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_val_accuracy(
    histories: List[dict],
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    for item in histories:
        plt.plot(
            item["epochs"],
            item["val_accuracy"],
            label=item["label"],
            linewidth=2,
        )

    plt.title("Validation Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_summary_table(histories: List[dict]) -> List[dict]:
    rows = []

    for item in histories:
        summary = item.get("summary", {})
        rows.append(
            {
                "label": item["label"],
                "model_name": item["model_name"],
                "experiment_name": item["experiment_name"],
                "num_epochs": len(item["epochs"]),
                "final_train_loss": item["train_loss"][-1],
                "final_train_accuracy": item["train_accuracy"][-1],
                "final_val_loss": item["val_loss"][-1],
                "final_val_accuracy": item["val_accuracy"][-1],
                "best_val_accuracy": summary.get("best_val_accuracy"),
                "final_test_accuracy": summary.get("final_test_accuracy"),
                "total_training_seconds": summary.get("total_training_seconds"),
            }
        )

    return rows


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot training loss and validation accuracy curves from experiment metric files."
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

    histories: List[dict] = []
    for label, path in parsed_inputs:
        metrics = load_json(path)
        histories.append(extract_history(metrics, label))

    train_loss_path = figure_dir / "training_loss_curves.png"
    val_acc_path = figure_dir / "validation_accuracy_curves.png"
    summary_path = table_dir / "training_curve_summary.json"

    plot_train_loss(histories, train_loss_path)
    plot_val_accuracy(histories, val_acc_path)

    output = {
        "inputs": [
            {"label": label, "path": str(path)}
            for label, path in parsed_inputs
        ],
        "summary_rows": build_summary_table(histories),
        "artifacts": {
            "train_loss_figure": str(train_loss_path),
            "val_accuracy_figure": str(val_acc_path),
            "summary_json": str(summary_path),
        },
    }
    save_json(output, summary_path)

    print("Training curve plotting finished.")
    print(f"Saved train loss figure:   {train_loss_path}")
    print(f"Saved val accuracy figure: {val_acc_path}")
    print(f"Saved summary json:        {summary_path}")


if __name__ == "__main__":
    main()