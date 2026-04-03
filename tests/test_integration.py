from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from models.cnn import (
    SmallCNN,
    build_approx_cnn,
    build_baseline_cnn,
    build_exact_cnn,
    build_replaced_cnn,
    build_softmax_eval_cnn,
)
from safe_activations.approx import (
    apply_approx_function,
    approx_gelu,
    approx_relu,
    approx_sigmoid,
    approx_softmax,
)
from safe_activations.exact import (
    apply_exact_function,
    exact_gelu,
    exact_relu,
    exact_sigmoid,
    exact_softmax,
)
from safe_activations.fit_config import (
    build_runtime_config,
    get_all_function_configs,
    get_default_degree,
    get_default_interval,
    has_function_config,
    list_supported_functions,
)
from safe_activations.modules import (
    ApproxGELU,
    ApproxReLU,
    ApproxSigmoid,
    ApproxSoftmax,
    BaseApproxActivation,
    build_approx_activation,
)
from safe_activations.replace import (
    get_model_activation_summary,
    replace_activation_modules,
    summarize_replacements,
    switch_all_approx_modules,
)


class TestFitConfigIntegration(unittest.TestCase):
    def test_supported_functions_exist(self) -> None:
        supported = list_supported_functions()
        self.assertIn("relu", supported)
        self.assertIn("sigmoid", supported)
        self.assertIn("gelu", supported)
        self.assertIn("softmax", supported)

    def test_default_configs_are_accessible(self) -> None:
        self.assertTrue(has_function_config("relu"))
        self.assertTrue(has_function_config("sigmoid"))
        self.assertTrue(has_function_config("gelu"))
        self.assertTrue(has_function_config("softmax"))

        configs = get_all_function_configs()
        self.assertIn("relu", configs)
        self.assertIn("gelu", configs)

    def test_runtime_config_builds_successfully(self) -> None:
        cfg = build_runtime_config("relu")
        self.assertEqual(cfg["name"], "relu")
        self.assertGreater(cfg["degree"], 0)

        degree = get_default_degree("gelu")
        interval = get_default_interval("gelu")
        self.assertGreater(degree, 0)
        self.assertLess(interval[0], interval[1])


class TestExactApproxFunctionIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.linspace(-3, 3, steps=17, dtype=torch.float32)
        self.logits = torch.tensor(
            [[1.2, 0.1, -0.5], [2.0, 1.8, 0.3]],
            dtype=torch.float32,
        )

    def test_exact_functions_keep_shape(self) -> None:
        self.assertEqual(exact_relu(self.x).shape, self.x.shape)
        self.assertEqual(exact_sigmoid(self.x).shape, self.x.shape)
        self.assertEqual(exact_gelu(self.x).shape, self.x.shape)
        self.assertEqual(exact_softmax(self.logits).shape, self.logits.shape)

    def test_approx_functions_keep_shape(self) -> None:
        self.assertEqual(approx_relu(self.x).shape, self.x.shape)
        self.assertEqual(approx_sigmoid(self.x).shape, self.x.shape)
        self.assertEqual(approx_gelu(self.x).shape, self.x.shape)
        self.assertEqual(approx_softmax(self.logits).shape, self.logits.shape)

    def test_apply_exact_function_dispatch(self) -> None:
        relu_y = apply_exact_function("relu", self.x)
        gelu_y = apply_exact_function("gelu", self.x)
        softmax_y = apply_exact_function("softmax", self.logits)

        self.assertEqual(relu_y.shape, self.x.shape)
        self.assertEqual(gelu_y.shape, self.x.shape)
        self.assertEqual(softmax_y.shape, self.logits.shape)

    def test_apply_approx_function_dispatch(self) -> None:
        relu_y = apply_approx_function("relu", self.x, degree=4, interval=(-3, 3))
        gelu_y = apply_approx_function("gelu", self.x, degree=5, interval=(-3, 3))
        softmax_y = apply_approx_function(
            "softmax",
            self.logits,
            degree=3,
            interval=(-4, 4),
        )

        self.assertEqual(relu_y.shape, self.x.shape)
        self.assertEqual(gelu_y.shape, self.x.shape)
        self.assertEqual(softmax_y.shape, self.logits.shape)

    def test_sigmoid_output_range_is_reasonable(self) -> None:
        y = approx_sigmoid(self.x, degree=5, interval=(-4, 4))
        self.assertTrue(torch.all(y >= 0.0).item())
        self.assertTrue(torch.all(y <= 1.0).item())

    def test_softmax_probability_sum_is_close_to_one(self) -> None:
        y = approx_softmax(self.logits, degree=3, interval=(-4, 4))
        sums = y.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))

    def test_exact_and_approx_are_finite(self) -> None:
        outputs = [
            exact_relu(self.x),
            exact_sigmoid(self.x),
            exact_gelu(self.x),
            exact_softmax(self.logits),
            approx_relu(self.x),
            approx_sigmoid(self.x),
            approx_gelu(self.x),
            approx_softmax(self.logits),
        ]
        for y in outputs:
            self.assertTrue(torch.isfinite(y).all().item())


class TestModuleIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.linspace(-3, 3, steps=17, dtype=torch.float32)
        self.logits = torch.tensor(
            [[1.2, 0.1, -0.5], [2.0, 1.8, 0.3]],
            dtype=torch.float32,
        )

    def test_build_approx_activation(self) -> None:
        relu_module = build_approx_activation("relu", mode="approx")
        gelu_module = build_approx_activation("gelu", mode="approx")
        self.assertIsInstance(relu_module, BaseApproxActivation)
        self.assertIsInstance(gelu_module, BaseApproxActivation)

    def test_exact_and_approx_modes_switch(self) -> None:
        module = ApproxReLU(mode="approx", degree=4, interval=(-3, 3))
        y1 = module(self.x)

        module.use_exact()
        y2 = module(self.x)

        self.assertEqual(y1.shape, self.x.shape)
        self.assertEqual(y2.shape, self.x.shape)
        self.assertEqual(module.mode, "exact")

    def test_softmax_module_forward(self) -> None:
        module = ApproxSoftmax(mode="approx", degree=3, interval=(-4, 4), dim=-1)
        y = module(self.logits)
        self.assertEqual(y.shape, self.logits.shape)

        sums = y.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))

    def test_module_repr_and_config(self) -> None:
        module = ApproxGELU(mode="approx", degree=5, interval=(-3, 3))
        repr_text = repr(module)
        self.assertIn("function_name=gelu", repr_text)

        cfg = module.get_runtime_config()
        self.assertEqual(cfg["name"], "gelu")
        self.assertEqual(cfg["degree"], 5)

    def test_sigmoid_module_forward(self) -> None:
        module = ApproxSigmoid(mode="approx", degree=5, interval=(-4, 4))
        y = module(self.x)
        self.assertEqual(y.shape, self.x.shape)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class TestReplaceIntegration(unittest.TestCase):
    def test_replace_activation_modules(self) -> None:
        model = DemoNet()
        records = replace_activation_modules(
            model,
            targets=["relu", "gelu", "softmax"],
            mode="approx",
            degree_map={"relu": 4, "gelu": 5, "softmax": 3},
            interval_map={
                "relu": (-3.0, 3.0),
                "gelu": (-3.0, 3.0),
                "softmax": (-4.0, 4.0),
            },
            method_map={
                "relu": "chebyshev",
                "gelu": "chebyshev",
                "softmax": "exp_poly_norm",
            },
        )

        self.assertGreaterEqual(len(records), 3)

        summary = summarize_replacements(records)
        self.assertIn("relu", summary)
        self.assertIn("gelu", summary)
        self.assertIn("softmax", summary)

        activation_summary = get_model_activation_summary(model)
        self.assertEqual(activation_summary["nn.ReLU"], 0)
        self.assertEqual(activation_summary["nn.GELU"], 0)
        self.assertEqual(activation_summary["nn.Softmax"], 0)
        self.assertGreaterEqual(activation_summary["ApproxReLU"], 1)
        self.assertGreaterEqual(activation_summary["ApproxGELU"], 1)
        self.assertGreaterEqual(activation_summary["ApproxSoftmax"], 1)

    def test_switch_all_approx_modules(self) -> None:
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


class TestCNNIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.images = torch.randn(4, 1, 28, 28)

    def test_baseline_cnn_forward(self) -> None:
        model = build_baseline_cnn(hidden_activation="relu")
        y = model(self.images)
        self.assertEqual(y.shape, (4, 10))

    def test_exact_cnn_forward(self) -> None:
        model = build_exact_cnn(
            hidden_activation="relu",
            degree=4,
            interval=(-3, 3),
            method="chebyshev",
        )
        y = model(self.images)
        self.assertEqual(y.shape, (4, 10))

    def test_approx_cnn_forward_relu(self) -> None:
        model = build_approx_cnn(
            hidden_activation="relu",
            degree=4,
            interval=(-3, 3),
            method="chebyshev",
        )
        y = model(self.images)
        self.assertEqual(y.shape, (4, 10))

    def test_approx_cnn_forward_gelu(self) -> None:
        model = build_approx_cnn(
            hidden_activation="gelu",
            degree=5,
            interval=(-3, 3),
            method="chebyshev",
        )
        y = model(self.images)
        self.assertEqual(y.shape, (4, 10))

    def test_softmax_eval_cnn_forward(self) -> None:
        model = build_softmax_eval_cnn(
            hidden_activation="relu",
            hidden_activation_mode="native",
            output_activation_mode="approx",
            output_degree=3,
            output_interval=(-4, 4),
            output_method="exp_poly_norm",
        )
        y = model(self.images)
        self.assertEqual(y.shape, (4, 10))

        sums = y.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))

    def test_replaced_cnn_chain(self) -> None:
        model, records = build_replaced_cnn(
            hidden_activation="relu",
            targets=["relu"],
            mode="approx",
            degree_map={"relu": 4},
            interval_map={"relu": (-3.0, 3.0)},
            method_map={"relu": "chebyshev"},
        )
        y = model(self.images)

        self.assertIsInstance(model, SmallCNN)
        self.assertEqual(y.shape, (4, 10))
        self.assertGreaterEqual(len(records), 3)


if __name__ == "__main__":
    unittest.main()