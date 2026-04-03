from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models.cnn import build_approx_cnn


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


def prepare_mnist_loaders(
    *,
    data_dir: str,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.ToTensor()

    full_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    total_size = len(full_train_dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def measure_inference_time(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    warmup_batches: int = 5,
    measure_batches: int = 20,
) -> Dict[str, float]:
    model.eval()

    batches = list(loader)
    if not batches:
        raise ValueError("The dataloader is empty.")

    warmup_batches = min(warmup_batches, len(batches))
    measure_batches = min(measure_batches, len(batches))

    for i in range(warmup_batches):
        images, _ = batches[i]
        images = images.to(device, non_blocking=True)
        _ = model(images)

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = 0.0
    total_samples = 0

    for i in range(measure_batches):
        images, _ = batches[i]
        images = images.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        total_time += end - start
        total_samples += images.size(0)

    return {
        "total_seconds": total_time,
        "num_batches": measure_batches,
        "num_samples": total_samples,
        "seconds_per_batch": total_time / measure_batches,
        "seconds_per_sample": total_time / total_samples,
    }


def train_single_configuration(
    *,
    hidden_activation: str,
    degree: int,
    interval: Tuple[float, float],
    method: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Dict[str, float]:
    model = build_approx_cnn(
        hidden_activation=hidden_activation,
        degree=degree,
        interval=interval,
        method=method,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_seconds": [],
    }

    best_val_accuracy = -1.0
    best_state_dict = None

    total_train_start = time.perf_counter()

    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_seconds = time.perf_counter() - epoch_start

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["epoch_seconds"].append(epoch_seconds)

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    total_training_seconds = time.perf_counter() - total_train_start

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    final_val_metrics = evaluate(model, val_loader, criterion, device)
    final_test_metrics = evaluate(model, test_loader, criterion, device)
    inference_metrics = measure_inference_time(model, test_loader, device)

    return {
        "degree": degree,
        "interval": [interval[0], interval[1]],
        "method": method,
        "best_val_accuracy": best_val_accuracy,
        "final_val_loss": final_val_metrics["loss"],
        "final_val_accuracy": final_val_metrics["accuracy"],
        "final_test_loss": final_test_metrics["loss"],
        "final_test_accuracy": final_test_metrics["accuracy"],
        "total_training_seconds": total_training_seconds,
        "average_epoch_seconds": sum(history["epoch_seconds"]) / len(history["epoch_seconds"]),
        "inference_seconds_per_batch": inference_metrics["seconds_per_batch"],
        "inference_seconds_per_sample": inference_metrics["seconds_per_sample"],
        "history": history,
    }


def parse_intervals(intervals_text: str) -> List[Tuple[float, float]]:
    """
    输入格式示例：
        "-2,2;-3,3;-4,4"
    """
    results: List[Tuple[float, float]] = []

    for item in intervals_text.split(";"):
        item = item.strip()
        if not item:
            continue

        left_text, right_text = item.split(",")
        left = float(left_text.strip())
        right = float(right_text.strip())

        if left >= right:
            raise ValueError(f"Invalid interval: {(left, right)}")

        results.append((left, right))

    if not results:
        raise ValueError("No valid intervals provided.")

    return results


def parse_degrees(degrees_text: str) -> List[int]:
    results = [int(x.strip()) for x in degrees_text.split(",") if x.strip()]
    if not results:
        raise ValueError("No valid degrees provided.")
    for degree in results:
        if degree <= 0:
            raise ValueError(f"Invalid degree: {degree}")
    return results


def build_default_method(hidden_activation: str) -> str:
    key = hidden_activation.lower()
    if key in {"relu", "gelu"}:
        return "chebyshev"
    if key == "sigmoid":
        return "least_squares"
    raise ValueError(f"Unsupported hidden activation: {hidden_activation}")


def build_default_degrees(hidden_activation: str) -> List[int]:
    key = hidden_activation.lower()
    if key == "relu":
        return [2, 4, 6]
    if key == "gelu":
        return [3, 5, 7]
    if key == "sigmoid":
        return [3, 5, 7]
    raise ValueError(f"Unsupported hidden activation: {hidden_activation}")


def build_default_intervals(hidden_activation: str) -> List[Tuple[float, float]]:
    key = hidden_activation.lower()
    if key in {"relu", "gelu"}:
        return [(-2.0, 2.0), (-3.0, 3.0), (-4.0, 4.0)]
    if key == "sigmoid":
        return [(-3.0, 3.0), (-4.0, 4.0), (-5.0, 5.0)]
    raise ValueError(f"Unsupported hidden activation: {hidden_activation}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ablation experiments on MNIST.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/logs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--hidden-activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "sigmoid"],
        help="要做消融的隐藏层激活函数。",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="",
        help="不填则按函数自动选择默认方法。",
    )

    parser.add_argument(
        "--degrees",
        type=str,
        default="",
        help='阶数列表，例如 "2,4,6"。',
    )
    parser.add_argument(
        "--intervals",
        type=str,
        default="",
        help='区间列表，例如 "-2,2;-3,3;-4,4"。',
    )

    parser.add_argument(
        "--fixed-degree",
        type=int,
        default=-1,
        help="跑区间消融时固定的阶数；默认取 degrees 的中间值。",
    )
    parser.add_argument(
        "--fixed-interval-left",
        type=float,
        default=float("nan"),
        help="跑阶数消融时固定区间左端点；不填则自动取默认中间区间。",
    )
    parser.add_argument(
        "--fixed-interval-right",
        type=float,
        default=float("nan"),
        help="跑阶数消融时固定区间右端点；不填则自动取默认中间区间。",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)
    output_dir = ensure_dir(args.output_dir)

    hidden_activation = args.hidden_activation.lower()
    method = args.method.strip() or build_default_method(hidden_activation)

    degrees = (
        parse_degrees(args.degrees)
        if args.degrees.strip()
        else build_default_degrees(hidden_activation)
    )
    intervals = (
        parse_intervals(args.intervals)
        if args.intervals.strip()
        else build_default_intervals(hidden_activation)
    )

    if args.fixed_degree > 0:
        fixed_degree = args.fixed_degree
    else:
        fixed_degree = degrees[len(degrees) // 2]

    if math.isnan(args.fixed_interval_left) or math.isnan(args.fixed_interval_right):
        fixed_interval = intervals[len(intervals) // 2]
    else:
        if args.fixed_interval_left >= args.fixed_interval_right:
            raise ValueError("fixed interval is invalid.")
        fixed_interval = (args.fixed_interval_left, args.fixed_interval_right)

    print("Preparing MNIST dataloaders...")
    train_loader, val_loader, test_loader = prepare_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print("Running degree ablation...")
    degree_ablation_results = []
    for degree in degrees:
        print(f"  -> degree={degree}, interval={fixed_interval}, method={method}")
        result = train_single_configuration(
            hidden_activation=hidden_activation,
            degree=degree,
            interval=fixed_interval,
            method=method,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
        )
        degree_ablation_results.append(result)

    print("Running interval ablation...")
    interval_ablation_results = []
    for interval in intervals:
        print(f"  -> degree={fixed_degree}, interval={interval}, method={method}")
        result = train_single_configuration(
            hidden_activation=hidden_activation,
            degree=fixed_degree,
            interval=interval,
            method=method,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
        )
        interval_ablation_results.append(result)

    output = {
        "experiment_name": "run_ablation",
        "hidden_activation": hidden_activation,
        "method": method,
        "config": {
            "data_dir": args.data_dir,
            "output_dir": str(output_dir),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "val_ratio": args.val_ratio,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "device": str(device),
        },
        "degree_ablation": {
            "fixed_interval": [fixed_interval[0], fixed_interval[1]],
            "degrees": degrees,
            "results": degree_ablation_results,
        },
        "interval_ablation": {
            "fixed_degree": fixed_degree,
            "intervals": [[x[0], x[1]] for x in intervals],
            "results": interval_ablation_results,
        },
    }

    output_path = output_dir / f"run_ablation_{hidden_activation}_metrics.json"
    save_json(output, output_path)

    print("\nAblation finished.")
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    import math
    main()