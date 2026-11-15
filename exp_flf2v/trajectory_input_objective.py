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
