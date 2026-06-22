import math
import unittest

import torch

from kanprey.basis_layers import GroupedBasisActivation
from kanprey.config import ModelConfig
from kanprey.kan_layers import GroupRational
from kanprey.model import BasisKANpreyLM
from scripts.benchmark_basis_kernels import (
    centers_and_width,
    gaussian_exact,
    gaussian_exp2,
    gaussian_lut_linear,
    run_benchmarks,
)


class BasisLayerTests(unittest.TestCase):
    def test_all_grouped_basis_families_are_finite_and_differentiable(self):
        families = [
            "chebyshev",
            "legendre",
            "gaussian",
            "inverse_quadratic",
            "wendland",
            "triangular_hat",
            "quadratic_hat",
            "relu_power",
            "soft_tree",
        ]
        for family in families:
            with self.subTest(family=family):
                torch.manual_seed(0)
                layer = GroupedBasisActivation(
                    d_in=8,
                    num_groups=4,
                    family=family,
                    degree=5,
                    centers=6,
                    width_scale=1.5,
                    input_norm="tanh",
                    relu_power=2,
                )
                x = torch.randn(3, 2, 8, requires_grad=True)
                y = layer(x)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.isfinite(y).all().item())
                y.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all().item())
                self.assertTrue(torch.isfinite(layer.coeff.grad).all().item())
                diag = layer.diagnostics(x.detach())
                self.assertEqual(diag.family, family)
                self.assertEqual(len(diag.basis_mean), layer.n_basis)
                self.assertEqual(len(diag.basis_std), layer.n_basis)

    def test_soft_tree_membership_is_partition_of_unity(self):
        layer = GroupedBasisActivation(
            d_in=8, num_groups=4, family="soft_tree", depth=3,
            steepness=4.0, input_norm="tanh",
        )
        self.assertEqual(layer.n_basis, 8)
        x = torch.randn(6, 8) * 2.0
        mu = layer.basis_values(x.reshape(-1, layer.g, layer.d_g))
        self.assertEqual(mu.shape[-1], 8)
        sums = mu.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))
        self.assertTrue((mu >= 0).all().item())

    def test_soft_tree_output_is_convex_combination_of_leaves(self):
        torch.manual_seed(0)
        layer = GroupedBasisActivation(
            d_in=8, num_groups=4, family="soft_tree", depth=3, init="random",
        )
        x = torch.randn(20, 8) * 3.0
        y = layer(x)
        # Each output is a convex combination of its group's leaf values, so it
        # must lie within the global leaf-value range regardless of input.
        self.assertLessEqual(y.max().item(), layer.coeff.max().item() + 1e-5)
        self.assertGreaterEqual(y.min().item(), layer.coeff.min().item() - 1e-5)

    def test_soft_tree_thresholds_and_steepness_are_learnable(self):
        layer = GroupedBasisActivation(
            d_in=8, num_groups=4, family="soft_tree", depth=3,
        )
        x = torch.randn(5, 8, requires_grad=True)
        layer(x).square().mean().backward()
        self.assertIsNotNone(layer.tree_thresholds.grad)
        self.assertIsNotNone(layer.tree_log_steepness.grad)
        self.assertTrue(torch.isfinite(layer.tree_thresholds.grad).all().item())
        self.assertTrue(torch.isfinite(layer.tree_log_steepness.grad).all().item())
        self.assertGreater(layer.tree_thresholds.grad.abs().sum().item(), 0.0)

    def test_soft_tree_lm_forward_and_backward(self):
        cfg = ModelConfig(
            vocab_size=32, d_model=16, n_heads=4, n_layers=2, max_seq_len=8,
            dropout=0.0, basis_family="soft_tree", basis_tree_depth=3, basis_groups=4,
        )
        model = BasisKANpreyLM(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
        logits = model(idx)
        self.assertEqual(logits.shape, (2, cfg.max_seq_len, cfg.vocab_size))
        self.assertTrue(torch.isfinite(logits).all().item())
        logits.square().mean().backward()
        grad_norm = sum(
            p.grad.detach().abs().sum().item()
            for p in model.parameters() if p.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

    def test_chebyshev_identity_init_is_exact_without_normalization(self):
        layer = GroupedBasisActivation(
            d_in=8,
            num_groups=4,
            family="chebyshev",
            degree=5,
            input_norm="none",
            init="identity",
        )
        x = torch.linspace(-0.9, 0.9, 40).reshape(5, 8)
        max_err = (layer(x) - x).abs().max().item()
        self.assertLess(max_err, 1e-4)

    def test_rational_denominator_modes_preserve_identity_init(self):
        x = torch.randn(4, 8, requires_grad=True)
        for denom in ("abs", "softplus", "square"):
            with self.subTest(denom=denom):
                layer = GroupRational(8, num_groups=4, m=3, n=2, init="identity", denominator=denom)
                y = layer(x)
                self.assertLess((y - x).abs().max().item(), 1e-5)
                y.square().mean().backward(retain_graph=True)
                self.assertTrue(torch.isfinite(x.grad).all().item())
                x.grad.zero_()

    def test_basis_lm_forward_shape(self):
        cfg = ModelConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=2,
            max_seq_len=8,
            dropout=0.0,
            basis_family="legendre",
            basis_degree=3,
            basis_groups=4,
            basis_centers=6,
        )
        model = BasisKANpreyLM(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
        logits = model(idx)
        self.assertEqual(logits.shape, (2, cfg.max_seq_len, cfg.vocab_size))
        self.assertTrue(torch.isfinite(logits).all().item())
        loss = logits.square().mean()
        loss.backward()
        grad_norm = sum(p.grad.detach().abs().sum().item() for p in model.parameters() if p.grad is not None)
        self.assertGreater(grad_norm, 0.0)

    def test_gaussian_microbench_reference_functions(self):
        x = torch.linspace(-1.0, 1.0, 12).reshape(3, 4).requires_grad_(True)
        centers, width = centers_and_width(5, 1.5, device=x.device, dtype=x.dtype)
        exact = gaussian_exact(x, centers, width)
        exp2 = gaussian_exp2(x, centers, width)
        lut = gaussian_lut_linear(x, centers, width, table_size=512)
        self.assertLess((exact - exp2).abs().max().item(), 1e-6)
        self.assertLess((exact - lut).abs().max().item(), 1e-3)

    def test_microbench_harness_runs_tiny_cpu_case(self):
        results = run_benchmarks(
            shape=(4, 3),
            device=torch.device("cpu"),
            dtype=torch.float32,
            centers_count=4,
            width_scale=1.5,
            repeats=1,
            seed=123,
        )
        self.assertEqual({r.name for r in results}, {
            "exact_torch_exp",
            "exp2_rewrite",
            "clamped_exp_z16",
            "lut_linear_z16_2048",
            "inverse_quadratic",
            "wendland_c2",
        })
        for result in results:
            self.assertTrue(math.isfinite(result.forward_ms))
            self.assertTrue(math.isfinite(result.backward_ms))


if __name__ == "__main__":
    unittest.main()
