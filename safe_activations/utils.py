from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch


Tensor = torch.Tensor


def ensure_dir(path: str | Path) -> Path:
    """
    确保目录存在，不存在则递归创建。
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: str | Path) -> None:
    """
    将字典保存为 JSON 文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict:
    """
    读取 JSON 文件并返回字典。
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_python_float(value: Any) -> float:
    """
    将张量标量或数值转为 Python float。
    """
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def tensor_to_list(x: Tensor) -> list:
    """
    将张量转为 Python list。
    """
    return x.detach().cpu().tolist()


def build_x_grid(
    *,
    x_min: float,
    x_max: float,
    num_points: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    构造等距采样点。
    """
    if x_min >= x_max:
        raise ValueError("x_min must be smaller than x_max.")
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    return torch.linspace(
        x_min,
        x_max,
        steps=num_points,
        device=device,
        dtype=dtype,
    )


def compute_error_metrics(
    exact_y: Tensor,
    approx_y: Tensor,
) -> Dict[str, float]:
    """
    计算 MAE / MSE / Max Error。
    """
    abs_error = torch.abs(exact_y - approx_y)
    sq_error = (exact_y - approx_y) ** 2

    return {
        "mae": float(abs_error.mean().item()),
        "mse": float(sq_error.mean().item()),
        "max_error": float(abs_error.max().item()),
    }


def summarize_history(history: Dict[str, Sequence[float]]) -> Dict[str, float]:
    """
    对训练历史做一个简短摘要。
    """
    summary: Dict[str, float] = {}

    for key, values in history.items():
        if not values:
            continue
        summary[f"{key}_last"] = float(values[-1])
        summary[f"{key}_best"] = float(max(values)) if "accuracy" in key else float(min(values))

    return summary


def validate_interval(interval: Tuple[float, float]) -> Tuple[float, float]:
    """
    校验区间合法性。
    """
    left, right = interval
    if left >= right:
        raise ValueError(f"Invalid interval: {interval}")
    return float(left), float(right)


def parse_degrees(text: str) -> List[int]:
    """
    解析 '2,4,6' 这样的阶数字符串。
    """
    degrees = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not degrees:
        raise ValueError("No valid degrees provided.")
    for degree in degrees:
        if degree <= 0:
            raise ValueError(f"Invalid degree: {degree}")
    return degrees


def parse_intervals(text: str) -> List[Tuple[float, float]]:
    """
    解析 '-2,2;-3,3;-4,4' 这样的区间字符串。
    """
    results: List[Tuple[float, float]] = []

    for item in text.split(";"):
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


def is_probability_distribution(
    x: Tensor,
    *,
    dim: int = -1,
    atol: float = 1e-4,
) -> bool:
    """
    判断张量在指定维度上是否近似概率分布。
    """
    if not torch.isfinite(x).all():
        return False
    if (x < 0).any():
        return False

    sums = x.sum(dim=dim)
    return torch.allclose(sums, torch.ones_like(sums), atol=atol)


if __name__ == "__main__":
    x = build_x_grid(x_min=-3, x_max=3, num_points=7)
    print("x:", x)
    print("tensor_to_list:", tensor_to_list(x))
    print("degrees:", parse_degrees("2,4,6"))
    print("intervals:", parse_intervals("-2,2;-3,3"))