from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cnn import build_baseline_cnn
from safe_activations.approx import approx_softmax
from safe_activations.exact import exact_softmax


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def prepare_test_loader(
    *,
    data_dir: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def accuracy_from_probs(probs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = probs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


@torch.no_grad()
def evaluate_softmax_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    degree: int,
    interval: Tuple[float, float],
    method: str,
) -> Dict[str, float]:
    """
    用 baseline 模型输出的 logits 比较 exact softmax 与 approx softmax。

    指标包括：
    - exact / approx 分类准确率
    - 概率分布 MAE / MSE / Max Error
    - 概率和偏离 1 的程度
    - 数值稳定性（是否出现 NaN / Inf）
    """
    model.eval()

    total_samples = 0
    exact_correct = 0
    approx_correct = 0

    total_abs_error = 0.0
    total_sq_error = 0.0
    max_abs_error = 0.0

    total_sum_abs_error = 0.0
    total_sum_sq_error = 0.0
    max_sum_abs_error = 0.0

    invalid_value_count = 0

    total_forward_exact_seconds = 0.0
    total_forward_approx_seconds = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)

        # exact softmax
        if device.type == "cuda":
            torch.cuda.synchronize()
        exact_start = time.perf_counter()
        exact_probs = exact_softmax(logits, dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        exact_end = time.perf_counter()

        # approx softmax
        if device.type == "cuda":
            torch.cuda.synchronize()
        approx_start = time.perf_counter()
        approx_probs = approx_softmax(
            logits,
            dim=-1,
            degree=degree,
            interval=interval,
            method=method,
            clip_input=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        approx_end = time.perf_counter()

        total_forward_exact_seconds += exact_end - exact_start
        total_forward_approx_seconds += approx_end - approx_start

        batch_size = targets.size(0)
        total_samples += batch_size

        exact_correct += (exact_probs.argmax(dim=1) == targets).sum().item()
        approx_correct += (approx_probs.argmax(dim=1) == targets).sum().item()

        abs_diff = torch.abs(exact_probs - approx_probs)
        sq_diff = (exact_probs - approx_probs) ** 2

        total_abs_error += abs_diff.sum().item()
        total_sq_error += sq_diff.sum().item()
        max_abs_error = max(max_abs_error, abs_diff.max().item())

        approx_prob_sum = approx_probs.sum(dim=1)
        sum_abs_error = torch.abs(approx_prob_sum - 1.0)
        sum_sq_error = (approx_prob_sum - 1.0) ** 2

        total_sum_abs_error += sum_abs_error.sum().item()
        total_sum_sq_error += sum_sq_error.sum().item()
        max_sum_abs_error = max(max_sum_abs_error, sum_abs_error.max().item())

        invalid_mask = ~torch.isfinite(approx_probs)
        invalid_value_count += invalid_mask.sum().item()

    num_prob_values = total_samples * 10  # MNIST 10 类
    probability_mae = total_abs_error / num_prob_values
    probability_mse = total_sq_error / num_prob_values

    prob_sum_mae = total_sum_abs_error / total_samples
    prob_sum_mse = total_sum_sq_error / total_samples

    is_stable = invalid_value_count == 0 and not math.isnan(probability_mae)

    return {
        "num_samples": total_samples,
        "exact_accuracy": exact_correct / total_samples,
        "approx_accuracy": approx_correct / total_samples,
        "accuracy_drop": (exact_correct - approx_correct) / total_samples,
        "probability_mae": probability_mae,
        "probability_mse": probability_mse,
        "probability_max_abs_error": max_abs_error,
        "probability_sum_mae": prob_sum_mae,
        "probability_sum_mse": prob_sum_mse,
        "probability_sum_max_abs_error": max_sum_abs_error,
        "invalid_value_count": invalid_value_count,
        "is_stable": is_stable,
        "exact_softmax_total_seconds": total_forward_exact_seconds,
        "approx_softmax_total_seconds": total_forward_approx_seconds,
        "exact_softmax_seconds_per_sample": total_forward_exact_seconds / total_samples,
        "approx_softmax_seconds_per_sample": total_forward_approx_seconds / total_samples,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate approx softmax on logits from a trained baseline CNN."
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/logs")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="outputs/logs/baseline_best.pt",
        help="已训练 baseline 模型的权重路径。",
    )
    parser.add_argument(
        "--hidden-activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "sigmoid"],
        help="baseline 模型训练时使用的隐藏层激活函数。",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--interval-left", type=float, default=-4.0)
    parser.add_argument("--interval-right", type=float, default=4.0)
    parser.add_argument("--method", type=str, default="exp_poly_norm")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.interval_left >= args.interval_right:
        raise ValueError("interval-left must be smaller than interval-right.")

    set_seed(args.seed)

    device = torch.device(args.device)
    output_dir = ensure_dir(args.output_dir)
    checkpoint_path = Path(args.checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Please run train_baseline.py first."
        )

    interval = (args.interval_left, args.interval_right)

    print("Preparing test dataloader...")
    test_loader = prepare_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Loading baseline model...")
    model = build_baseline_cnn(hidden_activation=args.hidden_activation).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print("Evaluating exact softmax vs approx softmax on model logits...")
    metrics = evaluate_softmax_outputs(
        model=model,
        loader=test_loader,
        device=device,
        degree=args.degree,
        interval=interval,
        method=args.method,
    )

    result = {
        "experiment_name": "eval_softmax",
        "model_name": "Baseline CNN logits -> Softmax evaluation",
        "checkpoint_path": str(checkpoint_path),
        "activation": {
            "hidden_activation": args.hidden_activation,
        },
        "softmax_approximation": {
            "degree": args.degree,
            "interval": [args.interval_left, args.interval_right],
            "method": args.method,
        },
        "config": {
            "data_dir": args.data_dir,
            "output_dir": str(output_dir),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "device": str(device),
        },
        "summary": metrics,
    }

    metrics_path = output_dir / "eval_softmax_metrics.json"
    save_json(result, metrics_path)

    print("\nSoftmax evaluation finished.")
    print(f"Exact accuracy:      {metrics['exact_accuracy']:.4f}")
    print(f"Approx accuracy:     {metrics['approx_accuracy']:.4f}")
    print(f"Accuracy drop:       {metrics['accuracy_drop']:.4f}")
    print(f"Probability MAE:     {metrics['probability_mae']:.8f}")
    print(f"Probability MSE:     {metrics['probability_mse']:.8f}")
    print(f"Prob-sum MAE:        {metrics['probability_sum_mae']:.8f}")
    print(f"Stable:              {metrics['is_stable']}")
    print(f"Saved metrics to:    {metrics_path}")


if __name__ == "__main__":
    main()