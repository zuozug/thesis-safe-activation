from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from safe_activations.approx import apply_approx_function
from safe_activations.exact import apply_exact_function
from safe_activations.fit_config import build_runtime_config, list_supported_functions


Tensor = torch.Tensor


class BaseApproxActivation(nn.Module):
    """
    可切换 exact / approx 模式的激活模块基类。

    设计目标：
    1. 对外提供统一的 nn.Module 接口
    2. 内部可在原函数与近似函数之间切换
    3. 统一管理 degree / interval / method / clip_input 等参数
    4. 便于后续 replace.py 自动替换模型中的激活层
    """

    SUPPORTED_MODES = {"exact", "approx"}

    def __init__(
        self,
        function_name: str,
        *,
        mode: str = "approx",
        degree: Optional[int] = None,
        interval: Optional[Tuple[float, float]] = None,
        method: Optional[str] = None,
        clip_input: bool = True,
        softmax_dim: int = -1,
        gelu_approximate: str = "none",
    ) -> None:
        super().__init__()

        self.function_name = function_name.lower()
        self._validate_function_name(self.function_name)

        self.mode = mode.lower()
        self._validate_mode(self.mode)

        self.degree = degree
        self.interval = interval
        self.method = method
        self.clip_input = clip_input

        # 仅对 softmax / gelu 生效的辅助参数
        self.softmax_dim = softmax_dim
        self.gelu_approximate = gelu_approximate

    @staticmethod
    def _validate_function_name(function_name: str) -> None:
        if function_name not in list_supported_functions():
            supported = ", ".join(list_supported_functions())
            raise KeyError(
                f"Unknown activation function: {function_name}. Supported: {supported}"
            )

    @classmethod
    def _validate_mode(cls, mode: str) -> None:
        if mode not in cls.SUPPORTED_MODES:
            supported = ", ".join(sorted(cls.SUPPORTED_MODES))
            raise ValueError(f"Unsupported mode: {mode}. Supported: {supported}")

    def set_mode(self, mode: str) -> "BaseApproxActivation":
        """
        切换模块模式：
            - exact: 使用原函数
            - approx: 使用近似函数
        """
        mode = mode.lower()
        self._validate_mode(mode)
        self.mode = mode
        return self

    def use_exact(self) -> "BaseApproxActivation":
        return self.set_mode("exact")

    def use_approx(self) -> "BaseApproxActivation":
        return self.set_mode("approx")

    def set_degree(self, degree: int) -> "BaseApproxActivation":
        if degree <= 0:
            raise ValueError(f"degree must be positive, got {degree}")
        self.degree = degree
        return self

    def set_interval(self, interval: Tuple[float, float]) -> "BaseApproxActivation":
        left, right = interval
        if left >= right:
            raise ValueError(f"Invalid interval: {interval}")
        self.interval = interval
        return self

    def set_method(self, method: str) -> "BaseApproxActivation":
        self.method = method
        return self

    def set_clip_input(self, clip_input: bool) -> "BaseApproxActivation":
        self.clip_input = bool(clip_input)
        return self

    def get_runtime_config(self) -> Dict[str, Any]:
        """
        获取当前模块的运行时配置。
        """
        cfg = build_runtime_config(
            self.function_name,
            degree=self.degree,
            interval=self.interval,
            method=self.method,
        )
        cfg.update(
            {
                "mode": self.mode,
                "clip_input": self.clip_input,
                "softmax_dim": self.softmax_dim,
                "gelu_approximate": self.gelu_approximate,
            }
        )
        return cfg

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "exact":
            return apply_exact_function(
                self.function_name,
                x,
                softmax_dim=self.softmax_dim,
                gelu_approximate=self.gelu_approximate,
            )

        if self.mode == "approx":
            return apply_approx_function(
                self.function_name,
                x,
                degree=self.degree,
                interval=self.interval,
                method=self.method,
                softmax_dim=self.softmax_dim,
                clip_input=self.clip_input,
            )

        raise RuntimeError(f"Unexpected mode: {self.mode}")

    def extra_repr(self) -> str:
        cfg = self.get_runtime_config()
        return (
            f"function_name={cfg['name']}, "
            f"mode={cfg['mode']}, "
            f"method={cfg['method']}, "
            f"degree={cfg['degree']}, "
            f"interval={cfg['interval']}, "
            f"clip_input={cfg['clip_input']}"
        )


class ApproxReLU(BaseApproxActivation):
    """
    ReLU 模块封装。
    """

    def __init__(
        self,
        *,
        mode: str = "approx",
        degree: Optional[int] = None,
        interval: Optional[Tuple[float, float]] = None,
        method: Optional[str] = None,
        clip_input: bool = True,
    ) -> None:
        super().__init__(
            "relu",
            mode=mode,
            degree=degree,
            interval=interval,
            method=method,
            clip_input=clip_input,
        )


class ApproxSigmoid(BaseApproxActivation):
    """
    Sigmoid 模块封装。
    """

    def __init__(
        self,
        *,
        mode: str = "approx",
        degree: Optional[int] = None,
        interval: Optional[Tuple[float, float]] = None,
        method: Optional[str] = None,
        clip_input: bool = True,
    ) -> None:
        super().__init__(
            "sigmoid",
            mode=mode,
            degree=degree,
            interval=interval,
            method=method,
            clip_input=clip_input,
        )


class ApproxGELU(BaseApproxActivation):
    """
    GELU 模块封装。
    """

    def __init__(
        self,
        *,
        mode: str = "approx",
        degree: Optional[int] = None,
        interval: Optional[Tuple[float, float]] = None,
        method: Optional[str] = None,
        clip_input: bool = True,
        gelu_approximate: str = "none",
    ) -> None:
        super().__init__(
            "gelu",
            mode=mode,
            degree=degree,
            interval=interval,
            method=method,
            clip_input=clip_input,
            gelu_approximate=gelu_approximate,
        )


class ApproxSoftmax(BaseApproxActivation):
    """
    Softmax 模块封装。

    说明：
    - softmax 通常用于输出层，因此额外暴露 dim 参数。
    - approx 模式下会走 exp 的多项式近似 + 归一化。
    """

    def __init__(
        self,
        *,
        mode: str = "approx",
        degree: Optional[int] = None,
        interval: Optional[Tuple[float, float]] = None,
        method: Optional[str] = None,
        clip_input: bool = True,
        dim: int = -1,
    ) -> None:
        super().__init__(
            "softmax",
            mode=mode,
            degree=degree,
            interval=interval,
            method=method,
            clip_input=clip_input,
            softmax_dim=dim,
        )

    def extra_repr(self) -> str:
        cfg = self.get_runtime_config()
        return (
            f"function_name={cfg['name']}, "
            f"mode={cfg['mode']}, "
            f"method={cfg['method']}, "
            f"degree={cfg['degree']}, "
            f"interval={cfg['interval']}, "
            f"dim={cfg['softmax_dim']}, "
            f"clip_input={cfg['clip_input']}"
        )


def build_approx_activation(
    function_name: str,
    *,
    mode: str = "approx",
    degree: Optional[int] = None,
    interval: Optional[Tuple[float, float]] = None,
    method: Optional[str] = None,
    clip_input: bool = True,
    softmax_dim: int = -1,
    gelu_approximate: str = "none",
) -> BaseApproxActivation:
    """
    工厂函数：根据函数名构造对应的 Approx 模块。
    """
    key = function_name.lower()

    if key == "relu":
        return ApproxReLU(
            mode=mode,
            degree=degree,
            interval=interval,
            method=method,
            clip_input=clip_input,
        )

    if key == "sigmoid":
        return ApproxSigmoid(
            mode=mode,
            degree=degree,
            interval=interval,
            method=method,
            clip_input=clip_input,
        )

    if key == "gelu":
        return ApproxGELU(
            mode=mode,
            degree=degree,
            interval=interval,
            method=method,
            clip_input=clip_input,
            gelu_approximate=gelu_approximate,
        )

    if key == "softmax":
        return ApproxSoftmax(
            mode=mode,
            degree=degree,
            interval=interval,
            method=method,
            clip_input=clip_input,
            dim=softmax_dim,
        )

    supported = ", ".join(list_supported_functions())
    raise KeyError(
        f"Unknown activation function for build_approx_activation: "
        f"{function_name}. Supported: {supported}"
    )


def is_approx_activation_module(module: nn.Module) -> bool:
    """
    判断某个模块是否属于本项目定义的 Approx 激活模块。
    """
    return isinstance(module, BaseApproxActivation)


if __name__ == "__main__":
    x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 2.0], dtype=torch.float32)

    relu_module = ApproxReLU()
    print(relu_module)
    print("ApproxReLU:", relu_module(x))

    relu_module.use_exact()
    print("ApproxReLU exact mode:", relu_module(x))

    gelu_module = ApproxGELU(mode="approx", degree=5)
    print(gelu_module)
    print("ApproxGELU:", gelu_module(x))

    logits = torch.tensor([[1.2, 0.1, -0.5], [2.0, 1.8, 0.3]], dtype=torch.float32)
    softmax_module = ApproxSoftmax(mode="approx", degree=3, dim=-1)
    print(softmax_module)
    print("ApproxSoftmax:", softmax_module(logits))

    built = build_approx_activation("sigmoid", mode="approx")
    print(built)
    print("Built ApproxSigmoid:", built(x))