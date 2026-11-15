"""Short-rollout surrogate: exact hard codebook forward, local soft backward."""
import torch
from sde_rf_wan.codebook_surrogate import soft_codebook_correction
from sde_rf_wan.sde_convert import velocity_to_score, diffusion_coeff, sde_drift


def hard_forward_soft_backward(hard, soft):
    return hard.detach() + (soft - soft.detach())


def correction(book, residual, step, position, pool_size=80, backward=True):
    if not book.M <= pool_size <= book.K:
        raise ValueError('Expected M <= pool_size <= K')
    flat = residual.flatten()
    with torch.no_grad():
        gen = torch.Generator(device=book.device).manual_seed(book._sf_seed(step, position))
        scores = torch.empty(book.K, device=book.device)
        for start in range(0, book.K, book.gen_batch):
            atoms = book._generate_atoms_batch(gen, min(book.gen_batch, book.K-start))
            scores[start:start+len(atoms)] = atoms @ flat.detach()
        selected = scores.abs().topk(book.M).indices.sort().values
        signs = scores[selected].sign()
        signs[signs == 0] = 1
        hard = book._regenerate_and_combine(book._sf_seed(step, position), selected.tolist(), signs)
        if not backward:
            return hard
        pool = scores.abs().topk(pool_size).indices.sort().values
        atoms = book.regenerate_selected_atoms(pool.tolist(), step, position).detach()
        temperature = max(float(scores[pool].abs().mean()) * 0.1, 1e-5)
    soft = soft_codebook_correction(flat, atoms, book.M,
                                   selection_temperature=temperature,
                                   sign_temperature=temperature).reshape_as(hard)
    return hard_forward_soft_backward(hard, soft)


def rollout_loss(target, reference, initial_state, times, predict, correct, g_scale=3.0):
    """Re-predict AFTER every update; reference is fixed, not an optimized target.

    times contains N+1 timestamps for N correction updates. Average error across
    latent positions and checkpoints. This is not full decoded RGB distortion.
    """
    if len(times) < 2 or any(not 0 < t <= 1 for t in times):
        raise ValueError('Positive rollout times required')
    if any(times[i] <= times[i+1] for i in range(len(times)-1)):
        raise ValueError('Times must strictly decrease')
    if target.shape != reference.shape or target.shape != initial_state.shape or target.ndim != 5 or target.shape[0] != 1:
        raise ValueError('Expected matching (1,C,F,H,W) tensors')
    state = initial_state
    velocity = predict(state, times[0])
    losses = []
    for step, (t, next_t) in enumerate(zip(times[:-1], times[1:])):
        residual = target - (state - t * velocity)
        noise = torch.stack([correct(residual[0, :, f], step, f)
                             for f in range(target.shape[2])], dim=1).unsqueeze(0)
        g = diffusion_coeff(t, g_scale)
        drift = sde_drift(velocity, velocity_to_score(velocity, state, t), g)
        state = state - drift * (t-next_t) + g * (t-next_t)**0.5 * noise
        velocity = predict(state, next_t)
        estimate = state - next_t * velocity
        losses.append((estimate-reference.detach()).square().mean())
    return torch.stack(losses).mean(), torch.stack(losses)


@torch.no_grad()
def full_hard_forward(target, reference, initial_state, times, num_sde_steps,
                      predict, correct, g_scale=3.0, progress=None):
    """Run the native hard-forward recurrence through t=0 in the given geometry."""
    return full_trajectory(target, reference, initial_state, times, num_sde_steps,
                           predict, correct, g_scale=g_scale, progress=progress)


def full_trajectory(target, reference, initial_state, times, num_sde_steps,
                    predict, correct, *, backward_steps=0, predict_grad=None,
                    correct_grad=None, g_scale=3.0, progress=None):
    """Same recurrence; optionally retain gradients only over a final suffix.

    Callbacks for the suffix must preserve hard-forward values. Absolute
    step indices are kept for the codebook, even after the graph cut.
    """
    if (len(times) < 2 or times[-1] != 0 or not 0 < times[0] <= 1 or
            any(a <= b for a, b in zip(times[:-1], times[1:])) or
            not 0 <= num_sde_steps <= len(times) - 2):
        raise ValueError('Expected a descending schedule to zero with a deterministic tail')
    if (target.shape != reference.shape or target.shape != initial_state.shape or
            target.ndim != 5 or target.shape[0] != 1):
        raise ValueError('Expected matching (1,C,F,H,W) tensors')
    n_steps = len(times) - 1
    if not 0 <= backward_steps <= n_steps:
        raise ValueError('Invalid backward horizon')
    cut = n_steps - backward_steps
    if backward_steps and (cut >= num_sde_steps or predict_grad is None or correct_grad is None):
        raise ValueError('Backward suffix needs a codebook selection and both callbacks')
    state = initial_state
    terms = []
    for step, (t, next_t) in enumerate(zip(times[:-1], times[1:])):
        track = step >= cut
        if step == cut:
            state = state.detach()
        with torch.set_grad_enabled(track):
            velocity = (predict_grad if track else predict)(state, t)
            estimate = state - t * velocity
            if step in (1, 2):
                terms.append(float((estimate.detach() - reference.detach()).square().mean()))
            dt = t - next_t
            if step < num_sde_steps:
                residual = target - estimate
                select = correct_grad if track else correct
                noise = torch.stack([select(residual[0, :, pos], step, pos)
                                     for pos in range(target.shape[2])], dim=1).unsqueeze(0)
                g = diffusion_coeff(t, g_scale)
                drift = sde_drift(velocity, velocity_to_score(velocity, state, t), g)
                state = state - drift * dt + g * dt**0.5 * noise
            else:
                state = state - velocity * dt
        if progress is not None:
            progress(step + 1)
    return state, terms


def rgb_distortion(reconstructed, reference):
    """Negative mean frame PSNR; a larger score means worse RGB fidelity."""
    if reconstructed.shape != reference.shape or reconstructed.ndim != 4:
        raise ValueError('Expected matching (F,C,H,W) RGB tensors')
    frame_mse = (reconstructed.float() - reference.detach().float()).square().mean((1, 2, 3))
    return 10 * torch.log10(frame_mse + 1e-10).mean()
