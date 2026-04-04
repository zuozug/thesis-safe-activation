from __future__ import annotations

import unittest

import torch

from safe_activations.approx import approx_gelu, approx_relu, approx_sigmoid, approx_softmax
from safe_activations.exact import exact_gelu, exact_relu, exact_sigmoid, exact_softmax
from safe_activations.utils import compute_error_metrics, is_probability_distribution


class TestDay2FunctionLevel(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.linspace(-3, 3, steps=101, dtype=torch.float32)
        self.logits = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 1.5, 0.2]],
            dtype=torch.float32,
        )

    def test_relu_curve_and_error(self) -> None:
        exact_y = exact_relu(self.x)
        approx_y = approx_relu(self.x, degree=4, interval=(-3, 3), method="chebyshev")
        metrics = compute_error_metrics(exact_y, approx_y)

        self.assertEqual(exact_y.shape, approx_y.shape)
        self.assertGreaterEqual(metrics["mae"], 0.0)
        self.assertGreaterEqual(metrics["mse"], 0.0)
        self.assertGreaterEqual(metrics["max_error"], 0.0)

    def test_sigmoid_curve_and_error(self) -> None:
        exact_y = exact_sigmoid(self.x)
        approx_y = approx_sigmoid(self.x, degree=5, interval=(-4, 4), method="least_squares")
        metrics = compute_error_metrics(exact_y, approx_y)

        self.assertEqual(exact_y.shape, approx_y.shape)
        self.assertTrue(torch.all(approx_y >= 0.0).item())
        self.assertTrue(torch.all(approx_y <= 1.0).item())
        self.assertGreaterEqual(metrics["mae"], 0.0)

    def test_gelu_curve_and_error(self) -> None:
        exact_y = exact_gelu(self.x)
        approx_y = approx_gelu(self.x, degree=5, interval=(-3, 3), method="chebyshev")
        metrics = compute_error_metrics(exact_y, approx_y)

        self.assertEqual(exact_y.shape, approx_y.shape)
        self.assertGreaterEqual(metrics["max_error"], 0.0)

    def test_softmax_curve_and_distribution(self) -> None:
        exact_y = exact_softmax(self.logits, dim=-1)
        approx_y = approx_softmax(self.logits, dim=-1, degree=3, interval=(-4, 4), method="exp_poly_norm")
        metrics = compute_error_metrics(exact_y, approx_y)

        self.assertEqual(exact_y.shape, approx_y.shape)
        self.assertTrue(is_probability_distribution(approx_y, dim=-1))
        self.assertGreaterEqual(metrics["mae"], 0.0)


if __name__ == "__main__":
    unittest.main()