from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

from safe_activations.approx import approx_gelu
from safe_activations.exact import exact_gelu


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_degrees(text: str) -> List[int]:
    degrees = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not degrees:
        raise ValueError("No valid degrees provided.")
    for degree in degrees:
        if degree <= 0:
            raise ValueError(f"Invalid degree: {degree}")
    return degrees


def build_x_grid(
    *,
    x_min: float,
    x_max: float,
    num_points: int,
    device: str = "cpu",
) -> torch.Tensor:
    if x_min >= x_max:
        raise ValueError("x_min must be smaller than x_max.")
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")
    return torch.linspace(x_min, x_max, steps=num_points, device=device, dtype=torch.float32)


def compute_error_metrics(
    exact_y: torch.Tensor,
    approx_y: torch.Tensor,
) -> Dict[str, float]:
    abs_error = torch.abs(exact_y - approx_y)
    sq_error = (exact_y - approx_y) ** 2

    return {
        "mae": abs_error.mean().item(),
        "mse": sq_error.mean().item(),
        "max_error": abs_error.max().item(),
    }


def plot_curve_figure(
    x: torch.Tensor,
    exact_y: torch.Tensor,
    approx_results: Dict[int, torch.Tensor],
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(x.cpu().numpy(), exact_y.cpu().numpy(), label="Exact GELU", linewidth=2)

    for degree, approx_y in approx_results.items():
        plt.plot(
            x.cpu().numpy(),
            approx_y.cpu().numpy(),
            label=f"Approx degree={degree}",
            linewidth=1.8,
        )

    plt.title("GELU: Exact vs Polynomial Approximation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_error_figure(
    x: torch.Tensor,
    exact_y: torch.Tensor,
    approx_results: Dict[int, torch.Tensor],
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    for degree, approx_y in approx_results.items():
        abs_error = torch.abs(exact_y - approx_y)
        plt.plot(
            x.cpu().numpy(),
            abs_error.cpu().numpy(),
            label=f"Degree={degree}",
            linewidth=1.8,
        )

    plt.title("GELU Approximation Absolute Error")
    plt.xlabel("x")
    plt.ylabel("|exact - approx|")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot GELU approximation curves and errors.")
    parser.add_argument("--degrees", type=str, default="3,5,7")
    parser.add_argument("--method", type=str, default="chebyshev")
    parser.add_argument("--interval-left", type=float, default=-3.0)
    parser.add_argument("--interval-right", type=float, default=3.0)

    parser.add_argument("--x-min", type=float, default=-5.0)
    parser.add_argument("--x-max", type=float, default=5.0)
    parser.add_argument("--num-points", type=int, default=2000)

    parser.add_argument("--figure-dir", type=str, default="outputs/figures")
    parser.add_argument("--table-dir", type=str, default="outputs/tables")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.interval_left >= args.interval_right:
        raise ValueError("interval-left must be smaller than interval-right.")

    degrees = parse_degrees(args.degrees)
    interval = (args.interval_left, args.interval_right)

    figure_dir = ensure_dir(args.figure_dir)
    table_dir = ensure_dir(args.table_dir)

    x = build_x_grid(
        x_min=args.x_min,
        x_max=args.x_max,
        num_points=args.num_points,
        device="cpu",
    )

    exact_y = exact_gelu(x, approximate="none")

    approx_results: Dict[int, torch.Tensor] = {}
    metrics: List[dict] = []

    for degree in degrees:
        approx_y = approx_gelu(
            x,
            degree=degree,
            interval=interval,
            method=args.method,
            clip_input=True,
        )
        approx_results[degree] = approx_y

        error_metrics = compute_error_metrics(exact_y, approx_y)
        metrics.append(
            {
                "function": "gelu",
                "degree": degree,
                "method": args.method,
                "interval": [interval[0], interval[1]],
                "x_range": [args.x_min, args.x_max],
                "num_points": args.num_points,
                "mae": error_metrics["mae"],
                "mse": error_metrics["mse"],
                "max_error": error_metrics["max_error"],
            }
        )

    curve_path = figure_dir / "gelu_curves.png"
    error_path = figure_dir / "gelu_errors.png"
    metrics_path = table_dir / "gelu_metrics.json"

    plot_curve_figure(x, exact_y, approx_results, curve_path)
    plot_error_figure(x, exact_y, approx_results, error_path)

    output = {
        "function": "gelu",
        "method": args.method,
        "interval": [interval[0], interval[1]],
        "x_range": [args.x_min, args.x_max],
        "num_points": args.num_points,
        "results": metrics,
        "artifacts": {
            "curve_figure": str(curve_path),
            "error_figure": str(error_path),
            "metrics_json": str(metrics_path),
        },
    }
    save_json(output, metrics_path)

    print("GELU plotting finished.")
    print(f"Saved curve figure: {curve_path}")
    print(f"Saved error figure: {error_path}")
    print(f"Saved metrics json: {metrics_path}")

    for item in metrics:
        print(
            f"degree={item['degree']}, "
            f"mae={item['mae']:.6f}, "
            f"mse={item['mse']:.6f}, "
            f"max_error={item['max_error']:.6f}"
        )


if __name__ == "__main__":
    main()