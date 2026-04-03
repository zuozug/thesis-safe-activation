from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

from safe_activations.approx import approx_softmax
from safe_activations.exact import exact_softmax


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_rest_logits(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("rest-logits must contain at least one value.")
    return values


def build_logits_grid(
    *,
    x_min: float,
    x_max: float,
    num_points: int,
    rest_logits: List[float],
) -> torch.Tensor:
    if x_min >= x_max:
        raise ValueError("x_min must be smaller than x_max.")
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    x = torch.linspace(x_min, x_max, steps=num_points, dtype=torch.float32)
    rest = torch.tensor(rest_logits, dtype=torch.float32).unsqueeze(0).repeat(num_points, 1)
    logits = torch.cat([x.unsqueeze(1), rest], dim=1)
    return logits


def compute_distribution_metrics(
    exact_probs: torch.Tensor,
    approx_probs: torch.Tensor,
) -> Dict[str, float]:
    abs_diff = torch.abs(exact_probs - approx_probs)
    sq_diff = (exact_probs - approx_probs) ** 2

    approx_sum = approx_probs.sum(dim=1)
    sum_abs_error = torch.abs(approx_sum - 1.0)
    sum_sq_error = (approx_sum - 1.0) ** 2

    return {
        "mae": abs_diff.mean().item(),
        "mse": sq_diff.mean().item(),
        "max_error": abs_diff.max().item(),
        "prob_sum_mae": sum_abs_error.mean().item(),
        "prob_sum_mse": sum_sq_error.mean().item(),
        "prob_sum_max_error": sum_abs_error.max().item(),
        "invalid_value_count": int((~torch.isfinite(approx_probs)).sum().item()),
    }


def plot_softmax_effect_figure(
    x_axis: torch.Tensor,
    exact_probs: torch.Tensor,
    approx_probs: torch.Tensor,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    num_classes = exact_probs.shape[1]
    x_np = x_axis.cpu().numpy()

    for class_idx in range(num_classes):
        plt.plot(
            x_np,
            exact_probs[:, class_idx].cpu().numpy(),
            label=f"Exact class {class_idx}",
            linewidth=2,
        )
        plt.plot(
            x_np,
            approx_probs[:, class_idx].cpu().numpy(),
            label=f"Approx class {class_idx}",
            linewidth=1.8,
            linestyle="--",
        )

    plt.title("Softmax: Exact vs Approx Probability Curves")
    plt.xlabel("varying logit x")
    plt.ylabel("probability")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_softmax_difference_figure(
    x_axis: torch.Tensor,
    exact_probs: torch.Tensor,
    approx_probs: torch.Tensor,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    x_np = x_axis.cpu().numpy()
    l1_error = torch.abs(exact_probs - approx_probs).sum(dim=1)
    sum_error = torch.abs(approx_probs.sum(dim=1) - 1.0)

    plt.plot(x_np, l1_error.cpu().numpy(), label="L1 distribution error", linewidth=2)
    plt.plot(x_np, sum_error.cpu().numpy(), label="Probability-sum error", linewidth=2)

    plt.title("Softmax Approximation Distribution Difference")
    plt.xlabel("varying logit x")
    plt.ylabel("error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot softmax approximation behavior.")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--method", type=str, default="exp_poly_norm")
    parser.add_argument("--interval-left", type=float, default=-4.0)
    parser.add_argument("--interval-right", type=float, default=4.0)

    parser.add_argument("--x-min", type=float, default=-6.0)
    parser.add_argument("--x-max", type=float, default=6.0)
    parser.add_argument("--num-points", type=int, default=2000)

    parser.add_argument(
        "--rest-logits",
        type=str,
        default="0.0,-1.0",
        help='固定其余 logits，例如 "0.0,-1.0" 表示总 logits 为 [x, 0.0, -1.0]。',
    )

    parser.add_argument("--figure-dir", type=str, default="outputs/figures")
    parser.add_argument("--table-dir", type=str, default="outputs/tables")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.degree <= 0:
        raise ValueError("degree must be positive.")
    if args.interval_left >= args.interval_right:
        raise ValueError("interval-left must be smaller than interval-right.")

    figure_dir = ensure_dir(args.figure_dir)
    table_dir = ensure_dir(args.table_dir)

    rest_logits = parse_rest_logits(args.rest_logits)
    interval = (args.interval_left, args.interval_right)

    logits = build_logits_grid(
        x_min=args.x_min,
        x_max=args.x_max,
        num_points=args.num_points,
        rest_logits=rest_logits,
    )

    x_axis = logits[:, 0]

    exact_probs = exact_softmax(logits, dim=-1)
    approx_probs = approx_softmax(
        logits,
        dim=-1,
        degree=args.degree,
        interval=interval,
        method=args.method,
        clip_input=True,
    )

    metrics = compute_distribution_metrics(exact_probs, approx_probs)

    effect_path = figure_dir / "softmax_effect.png"
    diff_path = figure_dir / "softmax_distribution_difference.png"
    metrics_path = table_dir / "softmax_metrics.json"

    plot_softmax_effect_figure(x_axis, exact_probs, approx_probs, effect_path)
    plot_softmax_difference_figure(x_axis, exact_probs, approx_probs, diff_path)

    output = {
        "function": "softmax",
        "degree": args.degree,
        "method": args.method,
        "interval": [interval[0], interval[1]],
        "x_range": [args.x_min, args.x_max],
        "num_points": args.num_points,
        "rest_logits": rest_logits,
        "num_classes": 1 + len(rest_logits),
        "metrics": metrics,
        "artifacts": {
            "effect_figure": str(effect_path),
            "difference_figure": str(diff_path),
            "metrics_json": str(metrics_path),
        },
    }
    save_json(output, metrics_path)

    print("Softmax plotting finished.")
    print(f"Saved effect figure:     {effect_path}")
    print(f"Saved difference figure: {diff_path}")
    print(f"Saved metrics json:      {metrics_path}")
    print(
        f"mae={metrics['mae']:.6f}, "
        f"mse={metrics['mse']:.6f}, "
        f"max_error={metrics['max_error']:.6f}, "
        f"prob_sum_mae={metrics['prob_sum_mae']:.6f}"
    )


if __name__ == "__main__":
    main()