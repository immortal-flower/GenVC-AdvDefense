import unittest
import torch
from exp_flf2v.trajectory_input_objective import (
    correction, full_hard_forward, hard_forward_soft_backward, rollout_loss,
)
from sde_rf_wan.turbo_codebook import TurboPerFrameCodebook


class TrajectoryObjectiveTest(unittest.TestCase):
    def test_hard_forward_soft_gradient(self):
        x = torch.tensor([2.], requires_grad=True)
        y = hard_forward_soft_backward(torch.tensor([7.]), x.square())
        self.assertEqual(y.item(), 7.)
        y.sum().backward()
        self.assertEqual(x.grad.item(), 4.)

    def test_updates_and_repredicts(self):
        target = torch.ones(1, 2, 3, 2, 2, requires_grad=True)
        calls = []
        def predict(x, t):
            calls.append(t)
            return x * .1
        loss, terms = rollout_loss(target, torch.zeros_like(target), torch.zeros_like(target),
                                  [.9, .8, .7], predict, lambda r,s,f: r * .2)
        grad, = torch.autograd.grad(loss, target)
        self.assertEqual(calls, [.9, .8, .7])
        self.assertEqual(len(terms), 2)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_hard_forward_matches_production_selection(self):
        book = TurboPerFrameCodebook(K=32, M=4, frame_shape=(2, 3, 4),
                                    seed=42, device=torch.device('cpu'), gen_batch=8)
        r = torch.randn(2, 3, 4, requires_grad=True)
        _, _, expected = book.select_atoms(r, 1, 2)
        actual = correction(book, r, 1, 2, pool_size=8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        g, = torch.autograd.grad((actual * torch.arange(24).reshape_as(actual)).sum(), r)
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(g.abs().sum().item(), 0)

    def test_rejects_invalid_schedule(self):
        x = torch.zeros(1, 2, 3, 2, 2)
        with self.assertRaises(ValueError):
            rollout_loss(x, x, x, [.8, .9], lambda x,t:x, lambda r,s,f:r)

    def test_full_forward_same_early_scores_and_no_terminal_prediction(self):
        target = torch.ones(1, 2, 3, 2, 2, requires_grad=True)
        reference = torch.zeros_like(target)
        initial = torch.zeros_like(target)
        times = [1., .9, .8, .7, 0.]
        _, early = rollout_loss(target, reference, initial, times[:3],
                                lambda x, t: x * .1, lambda r, s, f: r * .2)
        calls, picks = [], []

        def predict(x, t):
            calls.append(t)
            return x * .1

        def correct(r, step, pos):
            picks.append((step, pos))
            return r * .2

        state, scores = full_hard_forward(target, reference, initial, times, 2,
                                          predict, correct)
        self.assertEqual(calls, times[:-1])
        self.assertEqual(picks, [(step, pos) for step in range(2) for pos in range(3)])
        torch.testing.assert_close(torch.tensor(scores), early.detach())
        self.assertFalse(state.requires_grad)

    def test_full_forward_final_step_is_ode(self):
        initial = torch.ones(1, 1, 1, 1, 1)
        state, _ = full_hard_forward(initial, initial, initial, [1., .5, 0.], 0,
                                     lambda x, t: x, None)
        torch.testing.assert_close(state, torch.full_like(initial, .25))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_gpu_hard_selection_and_backward(self):
        book = TurboPerFrameCodebook(K=32, M=4, frame_shape=(2, 3, 4),
                                    seed=42, device=torch.device('cuda:0'), gen_batch=8)
        r = torch.randn(2, 3, 4, device='cuda:0', requires_grad=True)
        _, _, expected = book.select_atoms(r, 0, 0)
        actual = correction(book, r, 0, 0, pool_size=8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        grad, = torch.autograd.grad((actual * r.detach()).sum(), r)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
