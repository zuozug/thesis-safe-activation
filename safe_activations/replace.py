from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch.nn as nn

from safe_activations.modules import (
    ApproxGELU,
    ApproxReLU,
    ApproxSigmoid,
    ApproxSoftmax,
    BaseApproxActivation,
    build_approx_activation,
    is_approx_activation_module,
)


@dataclass
class ReplacementRecord:
    """
    记录一次模块替换的结果。
    """
    module_path: str
    old_type: str
    new_type: str
    function_name: str
    mode: str
    degree: Optional[int]
    interval: Optional[Tuple[float, float]]
    method: Optional[str]

    def to_dict(self) -> dict:
        return {
            "module_path": self.module_path,
            "old_type": self.old_type,
            "new_type": self.new_type,
            "function_name": self.function_name,
            "mode": self.mode,
            "degree": self.degree,
            "interval": self.interval,
            "method": self.method,
        }


SUPPORTED_TARGETS: Set[str] = {"relu", "sigmoid", "gelu", "softmax"}


def _normalize_targets(targets: Optional[Iterable[str]]) -> Set[str]:
    if targets is None:
        return set(SUPPORTED_TARGETS)

    normalized = {t.lower() for t in targets}
    unknown = normalized - SUPPORTED_TARGETS
    if unknown:
        raise ValueError(
            f"Unsupported targets: {sorted(unknown)}. "
            f"Supported: {sorted(SUPPORTED_TARGETS)}"
        )
    return normalized


def _get_function_name_from_module(module: nn.Module) -> Optional[str]:
    """
    将 PyTorch 原生激活层映射到函数名。
    """
    if isinstance(module, nn.ReLU):
        return "relu"
    if isinstance(module, nn.Sigmoid):
        return "sigmoid"
    if isinstance(module, nn.GELU):
        return "gelu"
    if isinstance(module, nn.Softmax):
        return "softmax"
    return None


def _build_replacement_module(
    old_module: nn.Module,
    *,
    function_name: str,
    mode: str,
    degree_map: Optional[Dict[str, int]],
    interval_map: Optional[Dict[str, Tuple[float, float]]],
    method_map: Optional[Dict[str, str]],
    clip_input: bool,
) -> BaseApproxActivation:
    """
    根据旧模块和全局配置构造新的 Approx 模块。
    """
    degree = degree_map.get(function_name) if degree_map else None
    interval = interval_map.get(function_name) if interval_map else None
    method = method_map.get(function_name) if method_map else None

    softmax_dim = -1
    gelu_approximate = "none"

    if isinstance(old_module, nn.Softmax):
        softmax_dim = old_module.dim

    if isinstance(old_module, nn.GELU):
        # PyTorch GELU 模块上有 approximate 属性
        gelu_approximate = getattr(old_module, "approximate", "none")

    new_module = build_approx_activation(
        function_name,
        mode=mode,
        degree=degree,
        interval=interval,
        method=method,
        clip_input=clip_input,
        softmax_dim=softmax_dim,
        gelu_approximate=gelu_approximate,
    )
    return new_module


def _replace_in_module(
    module: nn.Module,
    *,
    parent_path: str,
    targets: Set[str],
    mode: str,
    degree_map: Optional[Dict[str, int]],
    interval_map: Optional[Dict[str, Tuple[float, float]]],
    method_map: Optional[Dict[str, str]],
    clip_input: bool,
    records: List[ReplacementRecord],
) -> None:
    """
    递归遍历模块树并执行替换。
    """
    for child_name, child_module in module.named_children():
        child_path = f"{parent_path}.{child_name}" if parent_path else child_name

        # 已经是 Approx 模块就跳过，避免重复替换
        if is_approx_activation_module(child_module):
            continue

        function_name = _get_function_name_from_module(child_module)

        if function_name is not None and function_name in targets:
            new_module = _build_replacement_module(
                child_module,
                function_name=function_name,
                mode=mode,
                degree_map=degree_map,
                interval_map=interval_map,
                method_map=method_map,
                clip_input=clip_input,
            )

            setattr(module, child_name, new_module)

            records.append(
                ReplacementRecord(
                    module_path=child_path,
                    old_type=type(child_module).__name__,
                    new_type=type(new_module).__name__,
                    function_name=function_name,
                    mode=mode,
                    degree=new_module.degree,
                    interval=new_module.interval,
                    method=new_module.method,
                )
            )
        else:
            _replace_in_module(
                child_module,
                parent_path=child_path,
                targets=targets,
                mode=mode,
                degree_map=degree_map,
                interval_map=interval_map,
                method_map=method_map,
                clip_input=clip_input,
                records=records,
            )


def replace_activation_modules(
    model: nn.Module,
    *,
    targets: Optional[Iterable[str]] = None,
    mode: str = "approx",
    degree_map: Optional[Dict[str, int]] = None,
    interval_map: Optional[Dict[str, Tuple[float, float]]] = None,
    method_map: Optional[Dict[str, str]] = None,
    clip_input: bool = True,
) -> List[ReplacementRecord]:
    """
    自动替换模型中的激活层。

    Args:
        model: 需要替换的 PyTorch 模型
        targets: 指定要替换的函数名集合，如 ["relu", "gelu"]
                 默认替换所有受支持的激活层
        mode: 新模块的初始模式，"exact" 或 "approx"
        degree_map: 不同函数对应的阶数字典，例如 {"relu": 4, "gelu": 5}
        interval_map: 不同函数对应的区间字典，例如 {"relu": (-3, 3)}
        method_map: 不同函数对应的方法字典，例如 {"relu": "chebyshev"}
        clip_input: 是否在近似前裁剪输入到配置区间

    Returns:
        替换记录列表
    """
    normalized_targets = _normalize_targets(targets)

    if mode not in {"exact", "approx"}:
        raise ValueError(f"Unsupported mode: {mode}. Expected 'exact' or 'approx'.")

    records: List[ReplacementRecord] = []
    _replace_in_module(
        model,
        parent_path="",
        targets=normalized_targets,
        mode=mode,
        degree_map=degree_map,
        interval_map=interval_map,
        method_map=method_map,
        clip_input=clip_input,
        records=records,
    )
    return records


def count_replaced_modules(records: List[ReplacementRecord]) -> Dict[str, int]:
    """
    统计每类函数被替换的数量。
    """
    counts: Dict[str, int] = {}
    for record in records:
        counts[record.function_name] = counts.get(record.function_name, 0) + 1
    return counts


def summarize_replacements(records: List[ReplacementRecord]) -> str:
    """
    生成简短的替换摘要，便于日志打印或实验记录。
    """
    if not records:
        return "No activation modules were replaced."

    counts = count_replaced_modules(records)
    parts = [f"{name}: {count}" for name, count in sorted(counts.items())]
    return "Replaced activation modules -> " + ", ".join(parts)


def print_replacement_details(records: List[ReplacementRecord]) -> None:
    """
    打印详细替换信息。
    """
    if not records:
        print("No activation modules were replaced.")
        return

    print("Replacement details:")
    for record in records:
        print(
            f"- {record.module_path}: "
            f"{record.old_type} -> {record.new_type} "
            f"(function={record.function_name}, "
            f"mode={record.mode}, "
            f"degree={record.degree}, "
            f"interval={record.interval}, "
            f"method={record.method})"
        )


def switch_all_approx_modules(model: nn.Module, mode: str) -> nn.Module:
    """
    将模型中所有 Approx 模块统一切换到 exact 或 approx 模式。
    """
    if mode not in {"exact", "approx"}:
        raise ValueError(f"Unsupported mode: {mode}. Expected 'exact' or 'approx'.")

    for module in model.modules():
        if isinstance(module, BaseApproxActivation):
            module.set_mode(mode)
    return model


def get_model_activation_summary(model: nn.Module) -> Dict[str, int]:
    """
    统计模型中各类激活层数量，包括原生层与 Approx 层。
    """
    summary = {
        "nn.ReLU": 0,
        "nn.Sigmoid": 0,
        "nn.GELU": 0,
        "nn.Softmax": 0,
        "ApproxReLU": 0,
        "ApproxSigmoid": 0,
        "ApproxGELU": 0,
        "ApproxSoftmax": 0,
    }

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            summary["nn.ReLU"] += 1
        elif isinstance(module, nn.Sigmoid):
            summary["nn.Sigmoid"] += 1
        elif isinstance(module, nn.GELU):
            summary["nn.GELU"] += 1
        elif isinstance(module, nn.Softmax):
            summary["nn.Softmax"] += 1
        elif isinstance(module, ApproxReLU):
            summary["ApproxReLU"] += 1
        elif isinstance(module, ApproxSigmoid):
            summary["ApproxSigmoid"] += 1
        elif isinstance(module, ApproxGELU):
            summary["ApproxGELU"] += 1
        elif isinstance(module, ApproxSoftmax):
            summary["ApproxSoftmax"] += 1

    return summary


if __name__ == "__main__":
    class DemoNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.GELU(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(32, 10),
                nn.Softmax(dim=-1),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    model = DemoNet()

    print("Before replacement:")
    print(model)
    print(get_model_activation_summary(model))

    records = replace_activation_modules(
        model,
        targets=["relu", "gelu", "softmax"],
        mode="approx",
        degree_map={"relu": 4, "gelu": 5, "softmax": 3},
        interval_map={"relu": (-3.0, 3.0), "gelu": (-3.0, 3.0), "softmax": (-4.0, 4.0)},
        method_map={"relu": "chebyshev", "gelu": "chebyshev", "softmax": "exp_poly_norm"},
        clip_input=True,
    )

    print()
    print("After replacement:")
    print(model)
    print(get_model_activation_summary(model))
    print(summarize_replacements(records))
    print_replacement_details(records)

    switch_all_approx_modules(model, "exact")
    print()
    print("Switched all Approx modules to exact mode.")
    print(model)