from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from safe_activations.modules import build_approx_activation
from safe_activations.replace import replace_activation_modules


Tensor = torch.Tensor


def build_native_activation(name: str) -> nn.Module:
    """
    构造 PyTorch 原生激活层。

    这里用于 baseline 模型，或作为自动替换前的初始模型结构。
    """
    key = name.lower()

    if key == "relu":
        return nn.ReLU()
    if key == "sigmoid":
        return nn.Sigmoid()
    if key == "gelu":
        return nn.GELU()
    if key == "softmax":
        return nn.Softmax(dim=-1)

    raise KeyError(
        f"Unsupported native activation: {name}. "
        f"Supported: relu, sigmoid, gelu, softmax"
    )


def build_hidden_activation(
    name: str,
    *,
    activation_mode: str = "native",
    degree: int | None = None,
    interval: tuple[float, float] | None = None,
    method: str | None = None,
    clip_input: bool = True,
) -> nn.Module:
    """
    构造隐藏层激活。

    activation_mode:
        - "native": 返回 PyTorch 原生激活
        - "exact": 返回 Approx 模块，但内部走原函数
        - "approx": 返回 Approx 模块，但内部走近似函数
    """
    mode = activation_mode.lower()

    if mode == "native":
        return build_native_activation(name)

    if mode in {"exact", "approx"}:
        return build_approx_activation(
            name,
            mode=mode,
            degree=degree,
            interval=interval,
            method=method,
            clip_input=clip_input,
        )

    raise ValueError(
        f"Unsupported activation_mode: {activation_mode}. "
        f"Expected: native / exact / approx"
    )


class SmallCNN(nn.Module):
    """
    用于 MNIST 的小型 CNN。

    设计目标：
    1. 作为 baseline 模型直接训练
    2. 支持不同隐藏层激活函数（ReLU / GELU / Sigmoid）
    3. 支持 native / exact / approx 三种模式
    4. 默认输出 logits，便于直接配合 CrossEntropyLoss 使用

    默认结构：
        Conv(1 -> 16, 3x3, padding=1)
        Activation
        MaxPool(2)

        Conv(16 -> 32, 3x3, padding=1)
        Activation
        MaxPool(2)

        Flatten
        Linear(32*7*7 -> 128)
        Activation
        Linear(128 -> 10)
    """

    def __init__(
        self,
        *,
        hidden_activation: str = "relu",
        activation_mode: str = "native",
        degree: int | None = None,
        interval: tuple[float, float] | None = None,
        method: str | None = None,
        clip_input: bool = True,
        num_classes: int = 10,
        apply_output_softmax: bool = False,
        output_activation_mode: str = "native",
        output_degree: int | None = None,
        output_interval: tuple[float, float] | None = None,
        output_method: str | None = None,
    ) -> None:
        super().__init__()

        self.hidden_activation_name = hidden_activation.lower()
        self.activation_mode = activation_mode.lower()
        self.apply_output_softmax = apply_output_softmax
        self.output_activation_mode = output_activation_mode.lower()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.act1 = build_hidden_activation(
            self.hidden_activation_name,
            activation_mode=self.activation_mode,
            degree=degree,
            interval=interval,
            method=method,
            clip_input=clip_input,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = build_hidden_activation(
            self.hidden_activation_name,
            activation_mode=self.activation_mode,
            degree=degree,
            interval=interval,
            method=method,
            clip_input=clip_input,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.act3 = build_hidden_activation(
            self.hidden_activation_name,
            activation_mode=self.activation_mode,
            degree=degree,
            interval=interval,
            method=method,
            clip_input=clip_input,
        )

        self.fc2 = nn.Linear(128, num_classes)

        # 默认不加 softmax，方便直接训练 logits
        # 只有在专门做 softmax 输出层实验时才启用
        if self.apply_output_softmax:
            self.output_activation = build_hidden_activation(
                "softmax",
                activation_mode=self.output_activation_mode,
                degree=output_degree,
                interval=output_interval,
                method=output_method,
                clip_input=clip_input,
            )
        else:
            self.output_activation = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.act3(x)

        x = self.fc2(x)

        if self.output_activation is not None:
            x = self.output_activation(x)

        return x

    def get_model_config(self) -> dict:
        return {
            "hidden_activation": self.hidden_activation_name,
            "activation_mode": self.activation_mode,
            "apply_output_softmax": self.apply_output_softmax,
            "output_activation_mode": self.output_activation_mode,
        }


def build_baseline_cnn(hidden_activation: str = "relu") -> SmallCNN:
    """
    构造 baseline CNN。
    默认返回原生激活层版本。
    """
    return SmallCNN(
        hidden_activation=hidden_activation,
        activation_mode="native",
        apply_output_softmax=False,
    )


def build_exact_cnn(
    hidden_activation: str = "relu",
    *,
    degree: int | None = None,
    interval: tuple[float, float] | None = None,
    method: str | None = None,
) -> SmallCNN:
    """
    构造 exact 模式版本。
    这类模型使用 Approx 模块封装，但内部仍走原函数。
    主要用于验证模块封装本身是否影响模型行为。
    """
    return SmallCNN(
        hidden_activation=hidden_activation,
        activation_mode="exact",
        degree=degree,
        interval=interval,
        method=method,
        apply_output_softmax=False,
    )


def build_approx_cnn(
    hidden_activation: str = "relu",
    *,
    degree: int | None = None,
    interval: tuple[float, float] | None = None,
    method: str | None = None,
) -> SmallCNN:
    """
    构造 approx 模式版本。
    这是后面主实验使用的模型构造函数。
    """
    return SmallCNN(
        hidden_activation=hidden_activation,
        activation_mode="approx",
        degree=degree,
        interval=interval,
        method=method,
        apply_output_softmax=False,
    )


def build_softmax_eval_cnn(
    hidden_activation: str = "relu",
    *,
    hidden_activation_mode: str = "native",
    hidden_degree: int | None = None,
    hidden_interval: tuple[float, float] | None = None,
    hidden_method: str | None = None,
    output_activation_mode: str = "approx",
    output_degree: int | None = None,
    output_interval: tuple[float, float] | None = None,
    output_method: str | None = None,
) -> SmallCNN:
    """
    构造用于 softmax 输出层轻量实验的模型。

    注意：
    - 训练主线里通常不建议在模型末尾直接加 softmax，因为 CrossEntropyLoss 直接吃 logits。
    - 这个函数主要给 softmax 近似的独立评估脚本使用。
    """
    return SmallCNN(
        hidden_activation=hidden_activation,
        activation_mode=hidden_activation_mode,
        degree=hidden_degree,
        interval=hidden_interval,
        method=hidden_method,
        apply_output_softmax=True,
        output_activation_mode=output_activation_mode,
        output_degree=output_degree,
        output_interval=output_interval,
        output_method=output_method,
    )


def build_replaced_cnn(
    hidden_activation: str = "relu",
    *,
    targets: Optional[list[str]] = None,
    mode: str = "approx",
    degree_map: Optional[dict[str, int]] = None,
    interval_map: Optional[dict[str, tuple[float, float]]] = None,
    method_map: Optional[dict[str, str]] = None,
    clip_input: bool = True,
) -> tuple[SmallCNN, list]:
    """
    先构造原生激活层 CNN，再通过 replace.py 自动替换。
    这个函数用于验证自动替换链路是否完整。
    """
    model = build_baseline_cnn(hidden_activation=hidden_activation)
    records = replace_activation_modules(
        model,
        targets=targets,
        mode=mode,
        degree_map=degree_map,
        interval_map=interval_map,
        method_map=method_map,
        clip_input=clip_input,
    )
    return model, records


if __name__ == "__main__":
    x = torch.randn(4, 1, 28, 28)

    baseline = build_baseline_cnn(hidden_activation="relu")
    y_baseline = baseline(x)
    print("Baseline output shape:", y_baseline.shape)
    print("Baseline config:", baseline.get_model_config())

    approx_relu_model = build_approx_cnn(
        hidden_activation="relu",
        degree=4,
        interval=(-3.0, 3.0),
        method="chebyshev",
    )
    y_approx_relu = approx_relu_model(x)
    print("ApproxReLU output shape:", y_approx_relu.shape)
    print("ApproxReLU config:", approx_relu_model.get_model_config())

    approx_gelu_model = build_approx_cnn(
        hidden_activation="gelu",
        degree=5,
        interval=(-3.0, 3.0),
        method="chebyshev",
    )
    y_approx_gelu = approx_gelu_model(x)
    print("ApproxGELU output shape:", y_approx_gelu.shape)
    print("ApproxGELU config:", approx_gelu_model.get_model_config())

    replaced_model, records = build_replaced_cnn(
        hidden_activation="relu",
        targets=["relu"],
        mode="approx",
        degree_map={"relu": 4},
        interval_map={"relu": (-3.0, 3.0)},
        method_map={"relu": "chebyshev"},
    )
    y_replaced = replaced_model(x)
    print("Replaced model output shape:", y_replaced.shape)
    print("Replacement record count:", len(records))

    softmax_eval_model = build_softmax_eval_cnn(
        hidden_activation="relu",
        hidden_activation_mode="native",
        output_activation_mode="approx",
        output_degree=3,
        output_interval=(-4.0, 4.0),
        output_method="exp_poly_norm",
    )
    y_softmax = softmax_eval_model(x)
    print("Softmax eval output shape:", y_softmax.shape)