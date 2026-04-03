from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class FunctionConfig:
    """
    单个函数的近似配置。

    Attributes:
        name: 函数名，如 relu / sigmoid / gelu / softmax
        method: 近似方法名称
        degrees: 候选多项式阶数列表
        interval: 默认逼近区间
        is_main_experiment: 是否属于主实验主线
        notes: 备注说明
    """
    name: str
    method: str
    degrees: List[int]
    interval: Tuple[float, float]
    is_main_experiment: bool
    notes: str = ""


DEFAULT_FUNCTION_CONFIGS: Dict[str, FunctionConfig] = {
    "relu": FunctionConfig(
        name="relu",
        method="chebyshev",
        degrees=[2, 4, 6],
        interval=(-3.0, 3.0),
        is_main_experiment=True,
        notes="核心主实验函数，优先完成函数级与模型级完整验证。",
    ),
    "sigmoid": FunctionConfig(
        name="sigmoid",
        method="least_squares",
        degrees=[3, 5, 7],
        interval=(-4.0, 4.0),
        is_main_experiment=False,
        notes="补充函数，重点关注平滑性与饱和区误差。",
    ),
    "gelu": FunctionConfig(
        name="gelu",
        method="chebyshev",
        degrees=[3, 5, 7],
        interval=(-3.0, 3.0),
        is_main_experiment=True,
        notes="第二主实验函数，用于体现对现代激活函数的支持。",
    ),
    "softmax": FunctionConfig(
        name="softmax",
        method="exp_poly_norm",
        degrees=[3, 5],
        interval=(-4.0, 4.0),
        is_main_experiment=False,
        notes="补充实验函数，优先做函数级分析与输出层轻量验证。",
    ),
}


def list_supported_functions() -> List[str]:
    """返回当前支持的函数名列表。"""
    return list(DEFAULT_FUNCTION_CONFIGS.keys())


def has_function_config(name: str) -> bool:
    """判断某个函数是否存在配置。"""
    return name.lower() in DEFAULT_FUNCTION_CONFIGS


def get_function_config(name: str) -> FunctionConfig:
    """
    获取单个函数配置的深拷贝，避免外部误修改全局默认配置。
    """
    key = name.lower()
    if key not in DEFAULT_FUNCTION_CONFIGS:
        supported = ", ".join(list_supported_functions())
        raise KeyError(f"Unknown function config: {name}. Supported: {supported}")
    return deepcopy(DEFAULT_FUNCTION_CONFIGS[key])


def get_all_function_configs() -> Dict[str, FunctionConfig]:
    """获取全部函数配置的深拷贝。"""
    return deepcopy(DEFAULT_FUNCTION_CONFIGS)


def get_default_degree(name: str) -> int:
    """
    获取某个函数默认使用的阶数。
    这里约定默认取候选列表中的中间值。
    """
    config = get_function_config(name)
    if not config.degrees:
        raise ValueError(f"No degree candidates configured for function: {name}")
    return config.degrees[len(config.degrees) // 2]


def get_default_interval(name: str) -> Tuple[float, float]:
    """获取某个函数默认区间。"""
    return get_function_config(name).interval


def get_default_method(name: str) -> str:
    """获取某个函数默认近似方法。"""
    return get_function_config(name).method


def build_runtime_config(
    name: str,
    degree: int | None = None,
    interval: Tuple[float, float] | None = None,
    method: str | None = None,
) -> dict:
    """
    构造运行时配置字典，供 approx.py / plots / experiments 使用。

    返回字段:
        - name
        - method
        - degree
        - interval
        - is_main_experiment
        - notes
    """
    config = get_function_config(name)

    runtime_degree = degree if degree is not None else get_default_degree(name)
    runtime_interval = interval if interval is not None else config.interval
    runtime_method = method if method is not None else config.method

    if runtime_degree <= 0:
        raise ValueError(f"degree must be positive, got {runtime_degree}")

    left, right = runtime_interval
    if left >= right:
        raise ValueError(
            f"Invalid interval for {name}: left must be smaller than right, "
            f"got {runtime_interval}"
        )

    return {
        "name": config.name,
        "method": runtime_method,
        "degree": runtime_degree,
        "interval": runtime_interval,
        "is_main_experiment": config.is_main_experiment,
        "notes": config.notes,
    }


if __name__ == "__main__":
    print("Supported functions:", list_supported_functions())
    for fn_name in list_supported_functions():
        print(build_runtime_config(fn_name))