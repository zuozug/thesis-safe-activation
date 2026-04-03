from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def exact_relu(x: Tensor) -> Tensor:
    """
    ReLU 原函数。
    """
    return F.relu(x)


def exact_sigmoid(x: Tensor) -> Tensor:
    """
    Sigmoid 原函数。
    """
    return torch.sigmoid(x)


def exact_gelu(x: Tensor, approximate: str = "none") -> Tensor:
    """
    GELU 原函数。

    Args:
        x: 输入张量
        approximate: 传给 torch.nn.functional.gelu 的模式。
            - "none": 精确 GELU
            - "tanh": PyTorch 内置近似 GELU

    Returns:
        输出张量
    """
    if approximate not in {"none", "tanh"}:
        raise ValueError(
            f"Unsupported GELU approximate mode: {approximate}. "
            f"Expected 'none' or 'tanh'."
        )
    return F.gelu(x, approximate=approximate)


def exact_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Softmax 原函数。

    Args:
        x: 输入张量
        dim: 做 softmax 的维度，默认最后一维

    Returns:
        输出张量
    """
    return F.softmax(x, dim=dim)


EXACT_FUNCTIONS: Dict[str, Callable[..., Tensor]] = {
    "relu": exact_relu,
    "sigmoid": exact_sigmoid,
    "gelu": exact_gelu,
    "softmax": exact_softmax,
}


def list_supported_exact_functions() -> list[str]:
    """
    返回当前支持的原函数名称列表。
    """
    return list(EXACT_FUNCTIONS.keys())


def has_exact_function(name: str) -> bool:
    """
    判断是否支持某个原函数。
    """
    return name.lower() in EXACT_FUNCTIONS


def get_exact_function(name: str) -> Callable[..., Tensor]:
    """
    根据函数名获取原函数对象。

    Args:
        name: 函数名，如 relu / sigmoid / gelu / softmax

    Returns:
        对应的可调用函数

    Raises:
        KeyError: 当函数名不受支持时抛出
    """
    key = name.lower()
    if key not in EXACT_FUNCTIONS:
        supported = ", ".join(list_supported_exact_functions())
        raise KeyError(f"Unknown exact function: {name}. Supported: {supported}")
    return EXACT_FUNCTIONS[key]


def apply_exact_function(
    name: str,
    x: Tensor,
    *,
    softmax_dim: int = -1,
    gelu_approximate: str = "none",
) -> Tensor:
    """
    统一入口：根据函数名对输入张量应用原函数。

    Args:
        name: 函数名
        x: 输入张量
        softmax_dim: 当 name == "softmax" 时使用的维度
        gelu_approximate: 当 name == "gelu" 时使用的 GELU 模式

    Returns:
        输出张量
    """
    key = name.lower()

    if key == "relu":
        return exact_relu(x)
    if key == "sigmoid":
        return exact_sigmoid(x)
    if key == "gelu":
        return exact_gelu(x, approximate=gelu_approximate)
    if key == "softmax":
        return exact_softmax(x, dim=softmax_dim)

    supported = ", ".join(list_supported_exact_functions())
    raise KeyError(f"Unknown exact function: {name}. Supported: {supported}")


def describe_exact_function(name: str) -> dict:
    """
    返回某个原函数的描述信息，便于日志、调试或论文截图时使用。
    """
    key = name.lower()

    descriptions = {
        "relu": {
            "name": "relu",
            "display_name": "ReLU",
            "formula": "max(0, x)",
            "notes": "在 x=0 处不可导，是本课题的核心主实验对象。",
        },
        "sigmoid": {
            "name": "sigmoid",
            "display_name": "Sigmoid",
            "formula": "1 / (1 + exp(-x))",
            "notes": "平滑且饱和，适合作为平滑激活函数的补充实验对象。",
        },
        "gelu": {
            "name": "gelu",
            "display_name": "GELU",
            "formula": "x * Phi(x)",
            "notes": "现代模型常用激活函数，是第二主实验对象。",
        },
        "softmax": {
            "name": "softmax",
            "display_name": "Softmax",
            "formula": "exp(x_i) / sum_j exp(x_j)",
            "notes": "包含指数与归一化，近似难度最高，作为补充实验对象。",
        },
    }

    if key not in descriptions:
        supported = ", ".join(list_supported_exact_functions())
        raise KeyError(f"Unknown exact function: {name}. Supported: {supported}")

    return descriptions[key]


if __name__ == "__main__":
    x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 2.0])

    print("Supported exact functions:", list_supported_exact_functions())
    print("Input:", x)

    print("ReLU:", apply_exact_function("relu", x))
    print("Sigmoid:", apply_exact_function("sigmoid", x))
    print("GELU:", apply_exact_function("gelu", x))
    print("Softmax:", apply_exact_function("softmax", x))