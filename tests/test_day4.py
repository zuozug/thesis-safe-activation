from __future__ import annotations

import unittest

import torch

from models.cnn import (
    build_approx_cnn,
    build_baseline_cnn,
    build_exact_cnn,
    build_replaced_cnn,
    build_softmax_eval_cnn,
)


class TestDay4ModelLevel(unittest.TestCase):
    def setUp(self) -> None:
        self.images = torch.randn(4, 1, 28, 28)

    def test_baseline_cnn(self) -> None:
        model = build_baseline_cnn(hidden_activation="relu")
        y = model(self.images)
        self.assertEqual(y.shape, (4, 10))

    def test_exact_cnn(self) -> None:
        model = build_exact_cnn(
            hidden_activation="relu",
            degree=4,
            interval=(-3, 3),
            method="chebyshev",
        )
        y = model(self.images)
        self.assertEqual(y.shape, (4, 10))

    def test_approx_relu_cnn(self) -> None:
        model = build_approx_cnn(
            hidden_activation="relu",
            degree=4,
            interval=(-3, 3),
            method="chebyshev",
        )
        y = model(self.images)
        self.assertEqual(y.shape, (4, 10))

    def test_approx_gelu_cnn(self) -> None:
        model = build_approx_cnn(
            hidden_activation="gelu",
            degree=5,
            interval=(-3, 3),
            method="chebyshev",
        )
        y = model(self.images)
        self.assertEqual(y.shape, (4, 10))

    def test_softmax_eval_cnn(self) -> None:
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

    def test_replaced_cnn(self) -> None:
        model, records = build_replaced_cnn(
            hidden_activation="relu",
            targets=["relu"],
            mode="approx",
            degree_map={"relu": 4},
            interval_map={"relu": (-3.0, 3.0)},
            method_map={"relu": "chebyshev"},
        )
        y = model(self.images)

        self.assertEqual(y.shape, (4, 10))
        self.assertGreaterEqual(len(records), 3)


if __name__ == "__main__":
    unittest.main()