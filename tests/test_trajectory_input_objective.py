import unittest
import torch
from exp_flf2v.trajectory_input_objective import (
    correction, full_hard_forward, full_trajectory, hard_forward_soft_backward,
    match_grid_amplitude, rgb_distortion, rollout_loss, uniform_grid_direction,
    stepwise_trajectory_vjp,
)
from sde_rf_wan.turbo_codebook import TurboPerFrameCodebook


class TrajectoryObjectiveTest(unittest.TestCase):
    def test_stepwise_matches_retained_full_and_suffix_vjp(self):
        torch.manual_seed(7)
        target = torch.randn(1, 2, 3, 2, 2, dtype=torch.float64, requires_grad=True)
        initial = torch.randn_like(target)
        reference = torch.zeros_like(target)
        times = [1., .9, .8, .7, 0.]
        predict = lambda x, t: x.sin() * .1 + t * .03
        hard = lambda r, s, f: torch.tanh(r) * .2
        soft = lambda r, s, f: hard_forward_soft_backward(hard(r, s, f), r * .13)
        terminal_grad = torch.randn_like(target)
        for horizon in (3, 4):
            with self.subTest(horizon=horizon):
                end, _ = full_trajectory(
                    target, reference, initial, times, 2, predict, hard,
                    backward_steps=horizon, predict_grad=predict, correct_grad=soft,
                )
                expected, = torch.autograd.grad(end, target, grad_outputs=terminal_grad)
                calls = []
                actual = stepwise_trajectory_vjp(
                    target, reference, initial, times, 2, predict, hard, predict, soft,
                    terminal_grad, backward_steps=horizon, expected_end=end.detach(),
                    progress=calls.append,
                )
                torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
                self.assertEqual(calls, list(range(4, 4 - horizon, -1)))
                self.assertFalse(actual.requires_grad)

    def test_stepwise_rejects_changed_replay_and_invalid_horizon(self):
        x = torch.ones(1, 1, 1, 1, 1)
        predict = lambda x, t: x * .1
        hard = lambda r, s, f: r * .2
        for horizon, replay in ((1, predict), (3, lambda x, t: x * .2)):
            with self.subTest(horizon=horizon), self.assertRaises((ValueError, RuntimeError)):
                stepwise_trajectory_vjp(
                    x, x, x, [1., .8, .5, 0.], 2, predict, hard, replay, hard,
                    x, backward_steps=horizon,
                )

    def test_stepwise_matches_retained_codebook_vjp(self):
        torch.manual_seed(12)
        book = TurboPerFrameCodebook(K=16, M=2, frame_shape=(1, 2, 2),
                                    seed=42, device=torch.device('cpu'), gen_batch=8)
        target = torch.randn(1, 1, 2, 2, 2, requires_grad=True)
        initial = torch.randn_like(target)
        times = [1., .8, .5, 0.]
        predict = lambda x, t: x.sin() * .1
        hard = lambda r, s, f: book.select_atoms(r, s, f)[2]
        soft = lambda r, s, f: correction(book, r, s, f, pool_size=4)
        end, _ = full_trajectory(
            target, initial, initial, times, 2, predict, hard, backward_steps=3,
            predict_grad=predict, correct_grad=soft,
        )
        expected, = torch.autograd.grad(end.square().mean(), target)
        actual = stepwise_trajectory_vjp(
            target, initial, initial, times, 2, predict, hard, predict, soft,
            2 * end.detach() / end.numel(), backward_steps=3, expected_end=end.detach(),
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

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

    def test_truncated_backward_keeps_forward_and_cuts_prefix(self):
        target = torch.ones(1, 1, 1, 1, 1, requires_grad=True)
        initial = torch.zeros_like(target)
        times = [1., .9, .8, .7, 0.]
        predict = lambda state, t: state * .1
        # Only the first SDE correction has a target dependency in its forward.
        # Its gradient must be cut, leaving solely our chosen soft derivative.
        hard = lambda r, s, f: r if s == 0 else torch.zeros_like(r)
        calls = []

        def soft(r, s, f):
            calls.append(s)
            return hard_forward_soft_backward(hard(r, s, f), r)

        expected, _ = full_hard_forward(target, initial, initial, times, 2, predict, hard)
        actual, _ = full_trajectory(
            target, initial, initial, times, 2, predict, hard, backward_steps=3,
            predict_grad=predict, correct_grad=soft,
        )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(calls, [1])
        (grad,) = torch.autograd.grad(actual.sum(), target)
        # Single tracked SDE injection followed by two ODE updates.
        expected_grad = 3 * .9**2 * (.9 - .8)**.5 * (1 - .1*(.8-.7)) * (1 - .1*.7)
        torch.testing.assert_close(grad, torch.full_like(grad, expected_grad))
        with self.assertRaises(ValueError):
            full_trajectory(target, initial, initial, times, 2, predict, hard,
                            backward_steps=2, predict_grad=predict, correct_grad=soft)

    def test_rgb_objective_is_negative_mean_frame_psnr(self):
        rgb = torch.tensor([.1, .2]).reshape(2, 1, 1, 1).requires_grad_(True)
        score = rgb_distortion(rgb, torch.zeros_like(rgb))
        self.assertAlmostEqual(score.item(), -16.9897, places=4)
        (grad,) = torch.autograd.grad(score, rgb)
        self.assertTrue((grad > 0).all())

    def test_full_backward_tracks_first_and_last_sde_selection(self):
        target = torch.ones(1, 1, 1, 1, 1, requires_grad=True)
        initial = torch.zeros_like(target)
        times = [1., .9, .8, .7, 0.]
        calls, selected = [], []

        def predict(state, t):
            calls.append(t)
            return state * .1

        def correct(residual, step, pos):
            selected.append(step)
            return residual * .2

        state, _ = full_trajectory(
            target, initial, initial, times, 2, None, None, backward_steps=4,
            predict_grad=predict, correct_grad=correct,
        )
        self.assertEqual(calls, times[:-1])
        self.assertEqual(selected, [0, 1])
        (grad,) = torch.autograd.grad(state.sum(), target)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad), 0.)

    def test_grid_amplitude_with_clipped_nonmonotonic_energy(self):
        amp, mse = match_grid_amplitude(lambda a: min((a-1)**2 + 1, 5), 2.25, 4)
        self.assertAlmostEqual(amp, 1 + 1.25**.5, delta=.002)
        self.assertLess(abs(mse - 2.25), .00225)
        with self.assertRaises(ValueError):
            match_grid_amplitude(lambda a: 1., 2., 4)

    def test_continuous_grid_breaks_quantization_ties(self):
        shape = (3, 3, 20, 20)
        grid = uniform_grid_direction(shape, 42)
        torch.testing.assert_close(grid, uniform_grid_direction(shape, 42))
        self.assertLessEqual(float(grid.abs().max()), 1.)

        def energy(direction, amplitude):
            return float((direction * amplitude).round().clamp(-4, 4).square().mean())

        with self.assertRaises(ValueError):
            match_grid_amplitude(lambda a: energy(grid.sign(), a), 1.7, 8)
        _, matched = match_grid_amplitude(lambda a: energy(grid, a), 1.7, 8)
        self.assertLess(abs(matched - 1.7), .017)

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
