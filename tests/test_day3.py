from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from safe_activations.modules import (
    ApproxGELU,
    ApproxReLU,
    ApproxSigmoid,
    ApproxSoftmax,
    BaseApproxActivation,
    build_approx_activation,
)
from safe_activations.replace import replace_activation_modules, switch_all_approx_modules


class DemoNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, 4),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class TestDay3ModuleAndReplace(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.randn(4, 8)

    def test_module_building(self) -> None:
        relu_module = build_approx_activation("relu", mode="approx")
        sigmoid_module = build_approx_activation("sigmoid", mode="approx")
        gelu_module = build_approx_activation("gelu", mode="approx")
        softmax_module = build_approx_activation("softmax", mode="approx")

        self.assertIsInstance(relu_module, BaseApproxActivation)
        self.assertIsInstance(sigmoid_module, BaseApproxActivation)
        self.assertIsInstance(gelu_module, BaseApproxActivation)
        self.assertIsInstance(softmax_module, BaseApproxActivation)

    def test_module_forward(self) -> None:
        relu_module = ApproxReLU(mode="approx", degree=4, interval=(-3, 3))
        sigmoid_module = ApproxSigmoid(mode="approx", degree=5, interval=(-4, 4))
        gelu_module = ApproxGELU(mode="approx", degree=5, interval=(-3, 3))
        softmax_module = ApproxSoftmax(mode="approx", degree=3, interval=(-4, 4), dim=-1)

        relu_y = relu_module(self.x)
        sigmoid_y = sigmoid_module(self.x)
        gelu_y = gelu_module(self.x)
        softmax_y = softmax_module(torch.randn(4, 4))

        self.assertEqual(relu_y.shape, self.x.shape)
        self.assertEqual(sigmoid_y.shape, self.x.shape)
        self.assertEqual(gelu_y.shape, self.x.shape)
        self.assertEqual(softmax_y.shape, (4, 4))

    def test_replace_activation_modules(self) -> None:
        model = DemoNet()
        records = replace_activation_modules(
            model,
            targets=["relu", "gelu", "softmax"],
            mode="approx",
            degree_map={"relu": 4, "gelu": 5, "softmax": 3},
        )

        self.assertGreaterEqual(len(records), 3)

        y = model(self.x)
        self.assertEqual(y.shape, (4, 4))

    def test_switch_modes(self) -> None:
        model = DemoNet()
        replace_activation_modules(
            model,
            targets=["relu", "gelu", "softmax"],
            mode="approx",
            degree_map={"relu": 4, "gelu": 5, "softmax": 3},
        )

        switch_all_approx_modules(model, "exact")
        for module in model.modules():
            if isinstance(module, BaseApproxActivation):
                self.assertEqual(module.mode, "exact")


if __name__ == "__main__":
    unittest.main()