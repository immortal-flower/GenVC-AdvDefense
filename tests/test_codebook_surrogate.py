import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class SoftCodebookCorrectionTest(unittest.TestCase):
    def test_input_gradient_is_finite(self):
        from sde_rf_wan.codebook_surrogate import (
            soft_codebook_correction, uncovered_residual_ratio,
        )

        residual = torch.tensor([0.4, -0.3, 0.2, 0.1], requires_grad=True)
        atoms = torch.tensor([
            [1., 0., 0., 0.], [0., 1., 0., 0.],
            [0., 0., 1., 0.], [0., 0., 0., 1.],
        ])
        correction = soft_codebook_correction(residual, atoms, 2)
        objective = uncovered_residual_ratio(residual, correction, 0.3)
        objective.backward()
        self.assertTrue(torch.isfinite(objective))
        self.assertTrue(torch.isfinite(residual.grad).all())
        self.assertGreater(residual.grad.abs().sum().item(), 0)

    def test_rejects_invalid_geometry(self):
        from sde_rf_wan.codebook_surrogate import soft_codebook_correction

        with self.assertRaises(ValueError):
            soft_codebook_correction(torch.zeros(4), torch.zeros(2, 3), 1)
