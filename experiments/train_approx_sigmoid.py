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

        elapsed = end - start
        total_time += elapsed
        total_samples += images.size(0)

    return {
        "total_seconds": total_time,
        "num_batches": measure_batches,
        "num_samples": total_samples,
        "seconds_per_batch": total_time / measure_batches,
        "seconds_per_sample": total_time / total_samples,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ApproxSigmoid CNN on MNIST.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/logs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--interval-left", type=float, default=-4.0)
    parser.add_argument("--interval-right", type=float, default=4.0)
    parser.add_argument("--method", type=str, default="least_squares")
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
    interval = (args.interval_left, args.interval_right)

    print("Preparing MNIST dataloaders...")
    train_loader, val_loader, test_loader = prepare_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print("Building ApproxSigmoid model...")
    model = build_approx_cnn(
        hidden_activation="sigmoid",
        degree=args.degree,
        interval=interval,
        method=args.method,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_seconds": [],
    }

    best_val_accuracy = -1.0
    best_model_path = output_dir / "approx_sigmoid_best.pt"

    total_train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
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
            torch.save(model.state_dict(), best_model_path)

        print(
            f"[Epoch {epoch:02d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"train_acc={train_metrics['accuracy']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"time={epoch_seconds:.2f}s"
        )

    total_train_seconds = time.perf_counter() - total_train_start

    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    inference_metrics = measure_inference_time(model, test_loader, device)

    result = {
        "experiment_name": "train_approx_sigmoid",
        "model_name": "ApproxSigmoid CNN",
        "activation": {
            "hidden_activation": "sigmoid",
            "activation_mode": "approx",
            "degree": args.degree,
            "interval": [args.interval_left, args.interval_right],
            "method": args.method,
        },
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
        "history": history,
        "summary": {
            "best_val_accuracy": best_val_accuracy,
            "final_val_loss": val_metrics["loss"],
            "final_val_accuracy": val_metrics["accuracy"],
            "final_test_loss": test_metrics["loss"],
            "final_test_accuracy": test_metrics["accuracy"],
            "total_training_seconds": total_train_seconds,
            "average_epoch_seconds": sum(history["epoch_seconds"]) / len(history["epoch_seconds"]),
        },
        "inference": inference_metrics,
        "artifacts": {
            "best_model_path": str(best_model_path),
        },
    }

    metrics_path = output_dir / "train_approx_sigmoid_metrics.json"
    save_json(result, metrics_path)

    print("\nTraining finished.")
    print(f"Best val accuracy:   {best_val_accuracy:.4f}")
    print(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Total train time:    {total_train_seconds:.2f}s")
    print(f"Inference / batch:   {inference_metrics['seconds_per_batch']:.6f}s")
    print(f"Inference / sample:  {inference_metrics['seconds_per_sample']:.8f}s")
    print(f"Saved metrics to:    {metrics_path}")
    print(f"Saved best model to: {best_model_path}")


if __name__ == "__main__":
    main()