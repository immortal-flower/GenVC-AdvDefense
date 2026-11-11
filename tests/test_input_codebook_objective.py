import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class InputCodebookObjectiveTest(unittest.TestCase):
    def test_gradient_reaches_video(self):
        from exp_flf2v.input_codebook_objective import input_codebook_loss

        class FakeVAE:
            def encode(self, videos):
                return [videos[0].mean(dim=0, keepdim=True)]

        x = torch.full((5, 3, 2, 2), 0.5, requires_grad=True)
        clean = torch.zeros(1, 5, 2, 2)
        residual = torch.zeros_like(clean)
        residual[:, 1] = torch.tensor([[0.4, -0.3], [0.2, 0.1]])
        atoms = torch.eye(4)
        objective = input_codebook_loss(
            FakeVAE(), x, clean, residual, atoms,
            latent_position=1, m=2, gain=0.3,
        )
        objective.backward()
        self.assertTrue(torch.isfinite(objective))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_quantization_forward(self):
        from exp_flf2v.input_codebook_objective import straight_through_8bit

        x = torch.tensor([0.501], requires_grad=True)
        y = straight_through_8bit(x)
        self.assertAlmostEqual(y.item(), (x.detach() * 255).round().item() / 255)
        y.backward()
        self.assertEqual(x.grad.item(), 1.0)
