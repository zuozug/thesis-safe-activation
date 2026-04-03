from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, Tuple

import torch

from safe_activations.fit_config import (
    build_runtime_config,
    get_default_degree,
    get_default_interval,
    get_default_method,
    list_supported_functions,
)

Tensor = torch.Tensor


# =========================
# 基础工具
# =========================

def _validate_interval(interval: Tuple[float, float]) -> Tuple[float, float]:
    left, right = interval
    if left >= right:
        raise ValueError(f"Invalid interval: {interval}")
    return float(left), float(right)


def _build_sample_points(
    interval: Tuple[float, float],
    num_samples: int,
    strategy: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """
    构造用于多项式拟合的采样点。

    strategy:
        - "least_squares": 等距采样
        - "chebyshev": 使用 Chebyshev 节点
        - 其他值：退化为等距采样
    """
    left, right = _validate_interval(interval)

    if num_samples < 2:
        raise ValueError(f"num_samples must be >= 2, got {num_samples}")

    if strategy == "chebyshev":
        k = torch.arange(num_samples, device=device, dtype=dtype)
        # Chebyshev nodes on [-1, 1]
        nodes = torch.cos((2 * k + 1) * torch.pi / (2 * num_samples))
        # map to [left, right]
        x = 0.5 * (left + right) + 0.5 * (right - left) * nodes
        return torch.sort(x).values

    return torch.linspace(left, right, num_samples, device=device, dtype=dtype)


def _build_vandermonde(x: Tensor, degree: int) -> Tensor:
    """
    构造 Vandermonde 矩阵：
        [1, x, x^2, ..., x^degree]
    """
    cols = [torch.ones_like(x)]
    for power in range(1, degree + 1):
        cols.append(x ** power)
    return torch.stack(cols, dim=-1)


def _evaluate_polynomial(x: Tensor, coeffs: Tensor) -> Tensor:
    """
    使用 Horner 法评估多项式。

    coeffs 约定为：
        coeffs[0] + coeffs[1] * x + ... + coeffs[n] * x^n
    """
    y = torch.zeros_like(x, dtype=x.dtype)
    for c in reversed(coeffs):
        y = y * x + c.to(dtype=x.dtype, device=x.device)
    return y


def _fit_polynomial_coeffs(
    x: Tensor,
    y: Tensor,
    degree: int,
) -> Tensor:
    """
    使用最小二乘拟合多项式系数。
    """
    if degree <= 0:
        raise ValueError(f"degree must be positive, got {degree}")

    vander = _build_vandermonde(x, degree)
    result = torch.linalg.lstsq(vander, y.unsqueeze(-1))
    coeffs = result.solution.squeeze(-1)
    return coeffs


# =========================
# 目标函数采样
# =========================

def _target_relu(x: Tensor) -> Tensor:
    return torch.clamp_min(x, 0.0)


def _target_sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


def _target_gelu(x: Tensor) -> Tensor:
    return torch.nn.functional.gelu(x, approximate="none")


def _target_exp(x: Tensor) -> Tensor:
    return torch.exp(x)


def _get_scalar_target(name: str) -> Callable[[Tensor], Tensor]:
    key = name.lower()
    if key == "relu":
        return _target_relu
    if key == "sigmoid":
        return _target_sigmoid
    if key == "gelu":
        return _target_gelu
    if key == "softmax":
        # softmax 的核心困难来自 exp 和归一化；
        # 这里对 exp 做标量多项式近似，再在向量级完成归一化。
        return _target_exp

    supported = ", ".join(list_supported_functions())
    raise KeyError(f"Unknown target function: {name}. Supported: {supported}")


# =========================
# 系数缓存
# =========================

@lru_cache(maxsize=128)
def get_polynomial_coefficients(
    name: str,
    degree: int,
    interval: Tuple[float, float],
    method: str,
    num_samples: int = 2048,
) -> Tensor:
    """
    获取某个函数在给定配置下的多项式系数。

    返回：
        CPU 上的 float64 Tensor，按 [c0, c1, ..., cn] 排列。
    """
    key = name.lower()
    left, right = _validate_interval(interval)

    if degree <= 0:
        raise ValueError(f"degree must be positive, got {degree}")

    fit_strategy = method
    if key == "softmax" and method == "exp_poly_norm":
        # 对 softmax，先拟合 exp(x)
        fit_strategy = "least_squares"

    x = _build_sample_points(
        (left, right),
        num_samples=num_samples,
        strategy=fit_strategy,
        device="cpu",
        dtype=torch.float64,
    )
    target_fn = _get_scalar_target(key)
    y = target_fn(x)

    coeffs = _fit_polynomial_coeffs(x, y, degree)
    return coeffs.detach().cpu().to(torch.float64)


def get_runtime_polynomial_coefficients(
    name: str,
    degree: int | None = None,
    interval: Tuple[float, float] | None = None,
    method: str | None = None,
) -> Tensor:
    """
    按运行时配置获取多项式系数。
    """
    runtime_degree = degree if degree is not None else get_default_degree(name)
    runtime_interval = interval if interval is not None else get_default_interval(name)
    runtime_method = method if method is not None else get_default_method(name)

    return get_polynomial_coefficients(
        name=name,
        degree=runtime_degree,
        interval=runtime_interval,
        method=runtime_method,
    )


# =========================
# 各函数近似实现
# =========================

def approx_relu(
    x: Tensor,
    *,
    degree: int | None = None,
    interval: Tuple[float, float] | None = None,
    method: str | None = None,
    clip_input: bool = True,
) -> Tensor:
    """
    ReLU 多项式近似。
    """
    cfg = build_runtime_config("relu", degree=degree, interval=interval, method=method)
    coeffs = get_runtime_polynomial_coefficients(
        "relu",
        degree=cfg["degree"],
        interval=cfg["interval"],
        method=cfg["method"],
    )

    x_eval = x
    if clip_input:
        left, right = cfg["interval"]
        x_eval = torch.clamp(x_eval, min=left, max=right)

    return _evaluate_polynomial(x_eval, coeffs.to(device=x.device, dtype=x.dtype))


def approx_sigmoid(
    x: Tensor,
    *,
    degree: int | None = None,
    interval: Tuple[float, float] | None = None,
    method: str | None = None,
    clip_input: bool = True,
) -> Tensor:
    """
    Sigmoid 多项式近似。
    """
    cfg = build_runtime_config(
        "sigmoid",
        degree=degree,
        interval=interval,
        method=method,
    )
    coeffs = get_runtime_polynomial_coefficients(
        "sigmoid",
        degree=cfg["degree"],
        interval=cfg["interval"],
        method=cfg["method"],
    )

    x_eval = x
    if clip_input:
        left, right = cfg["interval"]
        x_eval = torch.clamp(x_eval, min=left, max=right)

    y = _evaluate_polynomial(x_eval, coeffs.to(device=x.device, dtype=x.dtype))
    # 作为激活近似，限制到概率范围更稳一些
    return torch.clamp(y, min=0.0, max=1.0)


def approx_gelu(
    x: Tensor,
    *,
    degree: int | None = None,
    interval: Tuple[float, float] | None = None,
    method: str | None = None,
    clip_input: bool = True,
) -> Tensor:
    """
    GELU 多项式近似。
    """
    cfg = build_runtime_config("gelu", degree=degree, interval=interval, method=method)
    coeffs = get_runtime_polynomial_coefficients(
        "gelu",
        degree=cfg["degree"],
        interval=cfg["interval"],
        method=cfg["method"],
    )

    x_eval = x
    if clip_input:
        left, right = cfg["interval"]
        x_eval = torch.clamp(x_eval, min=left, max=right)

    return _evaluate_polynomial(x_eval, coeffs.to(device=x.device, dtype=x.dtype))


def approx_softmax(
    x: Tensor,
    *,
    dim: int = -1,
    degree: int | None = None,
    interval: Tuple[float, float] | None = None,
    method: str | None = None,
    clip_input: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """
    Softmax 近似。

    设计思路：
    1. 先做数值稳定化：x - max(x)
    2. 对 exp(z) 做多项式近似
    3. 再做归一化

    说明：
    - 这是明文环境下的轻量近似实现，用于函数级和输出层实验。
    - 本文不在这里强行追求完整 HE/FHE 可执行性，只先验证模块与效果。
    """
    cfg = build_runtime_config(
        "softmax",
        degree=degree,
        interval=interval,
        method=method,
    )
    coeffs = get_runtime_polynomial_coefficients(
        "softmax",
        degree=cfg["degree"],
        interval=cfg["interval"],
        method=cfg["method"],
    )

    # 数值稳定化
    shifted = x - torch.amax(x, dim=dim, keepdim=True)

    if clip_input:
        left, right = cfg["interval"]
        shifted = torch.clamp(shifted, min=left, max=right)

    approx_exp = _evaluate_polynomial(
        shifted,
        coeffs.to(device=x.device, dtype=x.dtype),
    )

    # 避免负值和极小值带来的归一化问题
    approx_exp = torch.clamp_min(approx_exp, eps)

    denom = approx_exp.sum(dim=dim, keepdim=True).clamp_min(eps)
    return approx_exp / denom


# =========================
# 统一接口
# =========================

APPROX_FUNCTIONS: Dict[str, Callable[..., Tensor]] = {
    "relu": approx_relu,
    "sigmoid": approx_sigmoid,
    "gelu": approx_gelu,
    "softmax": approx_softmax,
}


def list_supported_approx_functions() -> list[str]:
    return list(APPROX_FUNCTIONS.keys())


def has_approx_function(name: str) -> bool:
    return name.lower() in APPROX_FUNCTIONS


def get_approx_function(name: str) -> Callable[..., Tensor]:
    key = name.lower()
    if key not in APPROX_FUNCTIONS:
        supported = ", ".join(list_supported_approx_functions())
        raise KeyError(f"Unknown approx function: {name}. Supported: {supported}")
    return APPROX_FUNCTIONS[key]


def apply_approx_function(
    name: str,
    x: Tensor,
    *,
    degree: int | None = None,
    interval: Tuple[float, float] | None = None,
    method: str | None = None,
    softmax_dim: int = -1,
    clip_input: bool = True,
) -> Tensor:
    """
    统一入口：按函数名应用近似函数。
    """
    key = name.lower()

    if key == "relu":
        return approx_relu(
            x,
            degree=degree,
            interval=interval,
            method=method,
            clip_input=clip_input,
        )
    if key == "sigmoid":
        return approx_sigmoid(
            x,
            degree=degree,
            interval=interval,
            method=method,
            clip_input=clip_input,
        )
    if key == "gelu":
        return approx_gelu(
            x,
            degree=degree,
            interval=interval,
            method=method,
            clip_input=clip_input,
        )
    if key == "softmax":
        return approx_softmax(
            x,
            dim=softmax_dim,
            degree=degree,
            interval=interval,
            method=method,
            clip_input=clip_input,
        )

    supported = ", ".join(list_supported_approx_functions())
    raise KeyError(f"Unknown approx function: {name}. Supported: {supported}")


def describe_approximation(name: str) -> dict:
    """
    返回某个函数当前默认近似配置的摘要信息。
    """
    cfg = build_runtime_config(name)
    return {
        "name": cfg["name"],
        "method": cfg["method"],
        "degree": cfg["degree"],
        "interval": cfg["interval"],
        "is_main_experiment": cfg["is_main_experiment"],
        "notes": cfg["notes"],
    }


if __name__ == "__main__":
    x = torch.linspace(-3, 3, steps=9)

    print("Supported approx functions:", list_supported_approx_functions())
    print("Input:", x)

    print("Approx ReLU:", apply_approx_function("relu", x))
    print("Approx Sigmoid:", apply_approx_function("sigmoid", x))
    print("Approx GELU:", apply_approx_function("gelu", x))

    logits = torch.tensor([[1.2, 0.1, -0.5], [2.0, 1.8, 0.3]], dtype=torch.float32)
    print("Approx Softmax:", apply_approx_function("softmax", logits))