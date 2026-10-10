"""Controlled corruption attacks on transmitted Turbo-DDCM codebook data."""
import math
import struct
from pathlib import Path

import numpy as np


STREAM_ATTACK_CHOICES = ['none', 'index-bitflip', 'sign-bitflip',
                         'sign-impact-bitflip', 'sign-trajectory-bitflip']
STREAM_DEFENSE_CHOICES = ['none', 'sign-repetition3']
STREAM_TRANSPORT_CHOICES = ['logical', 'serialized']


def _parse_tdcm_payload_layout(raw):
    """Locate every index and packed sign bit without touching TDCM headers.

    The returned offsets point into the actual serialized byte array.  Keeping
    this parser here (rather than estimating fixed offsets) also supports a
    variable ``M_actual`` in future tail steps.
    """
    if raw[:4] != b'TDCM':
        raise ValueError(f'Invalid TDCM magic: {bytes(raw[:4])!r}')
    if len(raw) < 48:
        raise ValueError('Truncated TDCM header')
    K, M, n_sde, n_ddim, n_lat, seed, ndim, n_fr, height, width = \
        struct.unpack_from('<10I', raw, 4)
    off = 44
    prompt_len = struct.unpack_from('<I', raw, off)[0]
    off += 4 + 4 * ndim + prompt_len
    idx_size = 2 if K <= 65536 else 4
    idx_fmt = '<H' if idx_size == 2 else '<I'
    slots = []
    frame_indices = {}
    for step in range(n_sde):
        for frame in range(n_lat):
            if off + 2 > len(raw):
                raise ValueError('Truncated TDCM payload before M_actual')
            m_actual = struct.unpack_from('<H', raw, off)[0]
            off += 2
            indices_offset = off
            indices_bytes = m_actual * idx_size
            signs_offset = indices_offset + indices_bytes
            signs_bytes = (m_actual + 7) // 8
            if signs_offset + signs_bytes > len(raw):
                raise ValueError('Truncated TDCM index/sign payload')
            key = (step, frame)
            values = []
            for atom in range(m_actual):
                index_offset = indices_offset + atom * idx_size
                index = struct.unpack_from(idx_fmt, raw, index_offset)[0]
                if index >= K:
                    raise ValueError(f'Invalid codebook index {index} >= K={K}')
                values.append(index)
                slots.append(dict(
                    step=step, frame=frame, atom=atom, index=index,
                    index_offset=index_offset, index_size=idx_size,
                    sign_offset=signs_offset + atom // 8,
                    sign_mask=1 << (atom % 8), frame_key=key,
                ))
            frame_indices[key] = set(values)
            off = signs_offset + signs_bytes
    if off != len(raw):
        raise ValueError(f'Unexpected {len(raw) - off} trailing TDCM bytes')
    return dict(K=K, M=M, n_sde=n_sde, n_ddim=n_ddim, n_lat=n_lat,
                seed=seed, num_frames=n_fr, height=height, width=width,
                slots=slots, frame_indices=frame_indices,
                bits_per_index=max(1, math.ceil(math.log2(K))))


def select_impact_sign_slots(step_data, codebook, count, step_idx, frame_indices):
    """Greedily choose signs that maximally rotate normalized codebook noise.

    The objective is the MSE between the original unit-variance combined noise
    and the combined noise after a candidate set of sign flips.  It uses only
    public bitstream indices/signs and the shared codebook seed—not the source
    video, reconstruction metrics, or Wan gradients.
    """
    import torch

    frame_indices = sorted({int(frame) for frame in frame_indices})
    if not frame_indices:
        raise ValueError('At least one impact target frame is required')
    capacity = sum(len(step_data[step_idx][frame][1]) for frame in frame_indices)
    if count < 1 or count > capacity:
        raise ValueError(f'Impact count must be in [1,{capacity}]')
    with torch.no_grad():
        states = {}
        for frame_idx in frame_indices:
            indices, signs = step_data[step_idx][frame_idx]
            atoms = codebook.regenerate_selected_atoms(
                indices, step_idx, frame_idx).float()
            signs_t = torch.tensor(signs, device=atoms.device, dtype=torch.float32)
            current_sum = (signs_t.unsqueeze(1) * atoms).sum(0)
            original = current_sum / current_sum.std().clamp_min(1e-8)
            states[frame_idx] = dict(
                indices=indices, atoms=atoms, signs=signs_t,
                current=current_sum, original=original,
                remaining=list(range(len(signs))), current_mse=0.0,
            )
        chosen = []
        trajectory = []
        for _ in range(count):
            total_before = sum(state['current_mse'] for state in states.values())
            best = None
            for frame_idx, state in states.items():
                if not state['remaining']:
                    continue
                rem_t = torch.tensor(state['remaining'], device=state['atoms'].device,
                                     dtype=torch.long)
                candidate_sums = (state['current'].unsqueeze(0)
                                  - 2.0 * state['signs'][rem_t].unsqueeze(1)
                                  * state['atoms'][rem_t])
                candidate_std = candidate_sums.std(
                    dim=1, keepdim=True).clamp_min(1e-8)
                candidate_noise = candidate_sums / candidate_std
                mse = ((candidate_noise - state['original'].unsqueeze(0)) ** 2).mean(dim=1)
                local = int(mse.argmax().item())
                candidate_total = total_before - state['current_mse'] + float(mse[local].item())
                if best is None or candidate_total > best['total']:
                    best = dict(frame=frame_idx, local=local,
                                mse=float(mse[local].item()), total=candidate_total)
                del rem_t, candidate_sums, candidate_std, candidate_noise, mse
            state = states[best['frame']]
            atom_position = state['remaining'].pop(best['local'])
            state['current'] = (state['current']
                                - 2.0 * state['signs'][atom_position]
                                * state['atoms'][atom_position])
            state['signs'][atom_position] *= -1.0
            state['current_mse'] = best['mse']
            chosen.append((step_idx, best['frame'], atom_position))
            trajectory.append(best['total'] / len(states))
        cosines = []
        for state in states.values():
            final_noise = state['current'] / state['current'].std().clamp_min(1e-8)
            cosines.append(float(torch.nn.functional.cosine_similarity(
                state['original'], final_noise, dim=0).item()))
        final_mse = sum(state['current_mse'] for state in states.values()) / len(states)
        final_cosine = sum(cosines) / len(cosines)
        selected_indices = [int(states[frame]['indices'][position])
                            for _, frame, position in chosen]
        allocation = {str(frame): sum(1 for _, selected_frame, _ in chosen
                                      if selected_frame == frame)
                      for frame in frame_indices}
        del states
    return chosen, dict(
        objective='greedy mean normalized-noise MSE across target frames',
        target_frames=frame_indices,
        allocation_by_frame=allocation,
        selected_atom_positions=[position for _, _, position in chosen],
        selected_frame_positions=[frame for _, frame, _ in chosen],
        selected_codebook_indices=selected_indices,
        cumulative_noise_mse=trajectory,
        final_noise_mse=final_mse,
        final_noise_cosine=final_cosine,
    )


def select_trajectory_sign_slots(step_data, codebook, count, step_idx,
                                 frame_indices, pipe, model, i2v_cond,
                                 trials=8, pool_factor=2.0, seed=42):
    """Choose sign flips by probing their effect on the next Wan prediction.

    A cheap geometric pass first constructs a candidate pool.  Several exact
    ``count``-bit subsets from that pool are then injected into the first SDE
    update.  The winner maximizes the MSE between the clean and corrupted Wan
    velocity at the *next* timestep.  This is a forward-only white-box probe:
    it uses the public model, bitstream, conditions and shared seed, but no
    source-video pixels, reconstruction labels or backward pass.

    Only the first SDE step is supported for now.  Probing a later step would
    require replaying every preceding attacked trajectory for each candidate.
    """
    import torch
    from sde_rf_wan.sde_convert import velocity_to_score, diffusion_coeff, sde_drift

    if step_idx != 0:
        raise ValueError('Trajectory sign ranking currently supports SDE step 1 only')
    if trials < 1:
        raise ValueError('Trajectory trials must be positive')
    frame_indices = sorted({int(frame) for frame in frame_indices})
    capacity = sum(len(step_data[step_idx][frame][1]) for frame in frame_indices)
    if count < 1 or count > capacity:
        raise ValueError(f'Trajectory count must be in [1,{capacity}]')
    pool_size = min(capacity, max(count, int(math.ceil(count * pool_factor))))

    # The returned order is the greedy geometry ranking trajectory.  Its first
    # ``count`` entries form a deterministic baseline candidate.
    pool_slots, pool_metadata = select_impact_sign_slots(
        step_data, codebook, pool_size, step_idx, frame_indices)
    pool_slots = [tuple(slot) for slot in pool_slots]
    candidates = [tuple(range(count))]
    seen = {candidates[0]}
    rng = np.random.default_rng(seed)
    attempts = 0
    while len(candidates) < trials and attempts < max(100, trials * 50):
        attempts += 1
        proposal = tuple(sorted(int(v) for v in
                                rng.choice(pool_size, size=count, replace=False)))
        if proposal not in seen:
            seen.add(proposal)
            candidates.append(proposal)

    with torch.no_grad():
        embeds = model.encode_prompt('')
        model_fn = pipe._model_fn(embeds, i2v_cond)
        gen = torch.Generator(device='cpu').manual_seed(pipe.seed)
        x_t = torch.randn(1, *pipe.latent_shape, generator=gen).to(pipe.device)
        t_curr = pipe.timesteps[0].item()
        t_next = pipe.timesteps[1].item()
        delta_t = t_curr - t_next
        u_t = model_fn(x_t, t_curr)
        score = velocity_to_score(u_t, x_t, t_curr)
        g_t = diffusion_coeff(t_curr, pipe.g_scale)
        f_t = sde_drift(u_t, score, g_t)
        noise_coeff = g_t * (delta_t ** 0.5)
        drift_state = x_t - f_t * delta_t

        clean_frames = []
        for frame_idx in range(pipe.num_latent_frames):
            indices, signs = step_data[0][frame_idx]
            clean_frames.append(codebook.reconstruct(indices, signs, 0, frame_idx))
        clean_noise = torch.stack(clean_frames, dim=1).unsqueeze(0)
        clean_next = drift_state + noise_coeff * clean_noise
        clean_velocity = model_fn(clean_next, t_next)

        scores = []
        best_score = -1.0
        best_slots = None
        best_state_mse = None
        for candidate_id, pool_positions in enumerate(candidates):
            selected = [pool_slots[position] for position in pool_positions]
            by_frame = {}
            for _, frame_idx, atom_position in selected:
                by_frame.setdefault(frame_idx, []).append(atom_position)
            attacked_noise = clean_noise.clone()
            for frame_idx, atom_positions in by_frame.items():
                indices, signs = step_data[0][frame_idx]
                attacked_signs = list(signs)
                for atom_position in atom_positions:
                    attacked_signs[atom_position] *= -1
                attacked_noise[0, :, frame_idx] = codebook.reconstruct(
                    indices, attacked_signs, 0, frame_idx)
            attacked_next = drift_state + noise_coeff * attacked_noise
            attacked_velocity = model_fn(attacked_next, t_next)
            velocity_mse = float(((attacked_velocity - clean_velocity) ** 2).mean().item())
            state_mse = float(((attacked_next - clean_next) ** 2).mean().item())
            scores.append(dict(candidate=candidate_id,
                               pool_positions=list(pool_positions),
                               velocity_mse=velocity_mse,
                               next_state_mse=state_mse))
            if velocity_mse > best_score:
                best_score = velocity_mse
                best_state_mse = state_mse
                best_slots = selected
            del attacked_noise, attacked_next, attacked_velocity

        del embeds, model_fn, x_t, u_t, score, f_t, drift_state
        del clean_frames, clean_noise, clean_next, clean_velocity
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    allocation = {str(frame): sum(1 for _, selected_frame, _ in best_slots
                                   if selected_frame == frame)
                  for frame in frame_indices}
    selected_indices = [int(step_data[step][frame][0][position])
                        for step, frame, position in best_slots]
    return best_slots, dict(
        objective='maximum next-timestep Wan velocity MSE',
        target_frames=frame_indices,
        allocation_by_frame=allocation,
        selected_atom_positions=[position for _, _, position in best_slots],
        selected_frame_positions=[frame for _, frame, _ in best_slots],
        selected_codebook_indices=selected_indices,
        candidate_pool_size=pool_size,
        requested_trials=int(trials),
        evaluated_trials=len(candidates),
        winning_velocity_mse=best_score,
        winning_next_state_mse=best_state_mse,
        candidate_scores=scores,
        geometry_pool=pool_metadata,
    )


def apply_serialized_codebook_attack(input_path, output_path, attack, rate=0.0,
                                     seed=42, early_steps=0, count=0,
                                     target_steps=None, target_frames=None,
                                     selected_slots=None, selection_metadata=None):
    """Flip real bits in a serialized ``.tdcm`` payload and write a new file.

    Only index bytes or packed sign bytes are eligible.  Header fields,
    ``M_actual`` and prompt/shape metadata are deliberately excluded so the
    experiment measures codebook-trajectory sensitivity rather than parser
    failure.  Index flips are constrained to valid, unique codebook indices.
    """
    if attack not in ('index-bitflip', 'sign-bitflip', 'sign-impact-bitflip',
                      'sign-trajectory-bitflip'):
        raise ValueError('Serialized attack requires index-bitflip or sign-bitflip')
    if count < 0:
        raise ValueError('Serialized stream attack count cannot be negative')
    if count == 0 and not 0 < rate <= 1:
        raise ValueError('Serialized stream attack requires count > 0 or rate in (0,1]')
    raw = bytearray(Path(input_path).read_bytes())
    layout = _parse_tdcm_payload_layout(raw)
    slots = layout['slots']
    total_symbols = len(slots)
    total_payload_bits = total_symbols * (layout['bits_per_index'] + 1)
    if early_steps > 0 and target_steps:
        raise ValueError('early_steps and target_steps are mutually exclusive')
    if target_steps:
        step_ids = {int(v) for v in target_steps}
        invalid = sorted(v for v in step_ids if v < 0 or v >= layout['n_sde'])
        if invalid:
            raise ValueError(f'Invalid zero-based SDE steps: {invalid}')
    else:
        max_step = min(early_steps, layout['n_sde']) if early_steps > 0 else layout['n_sde']
        step_ids = set(range(max_step))
    if target_frames:
        frame_ids = {int(v) for v in target_frames}
        invalid = sorted(v for v in frame_ids if v < 0 or v >= layout['n_lat'])
        if invalid:
            raise ValueError(f'Invalid latent frame positions: {invalid}')
    else:
        frame_ids = set(range(layout['n_lat']))
    eligible = [slot for slot in slots
                if slot['step'] in step_ids and slot['frame'] in frame_ids]
    requested = int(count) if count > 0 else max(1, int(round(total_symbols * rate)))
    if requested > len(eligible):
        raise ValueError(f'Requested {requested} flips but only {len(eligible)} symbols are eligible')
    rng = np.random.default_rng(seed)
    if selected_slots is not None:
        if attack not in ('sign-impact-bitflip', 'sign-trajectory-bitflip'):
            raise ValueError('Explicit selected_slots require a ranked sign attack')
        selected_set = {tuple(slot) for slot in selected_slots}
        chosen = [i for i, slot in enumerate(eligible)
                  if (slot['step'], slot['frame'], slot['atom']) in selected_set]
        if len(chosen) != len(selected_set) or len(chosen) != requested:
            raise ValueError('Impact-selected slots do not match eligible serialized slots/count')
    else:
        chosen = rng.choice(len(eligible), size=requested, replace=False)
    examples = []
    occupied = {key: set(values) for key, values in layout['frame_indices'].items()}
    for slot_id in chosen:
        slot = eligible[int(slot_id)]
        detail = dict(step=slot['step'], frame=slot['frame'], atom=slot['atom'])
        if attack in ('sign-bitflip', 'sign-impact-bitflip',
                      'sign-trajectory-bitflip'):
            old_positive = bool(raw[slot['sign_offset']] & slot['sign_mask'])
            raw[slot['sign_offset']] ^= slot['sign_mask']
            detail.update(old_sign=1 if old_positive else -1,
                          new_sign=-1 if old_positive else 1,
                          byte_offset=slot['sign_offset'],
                          bit_in_byte=slot['atom'] % 8)
        else:
            old_index = slot['index']
            used = occupied[slot['frame_key']]
            replacement = None
            flipped_bit = None
            for bit in rng.permutation(layout['bits_per_index']):
                candidate = old_index ^ (1 << int(bit))
                if candidate < layout['K'] and candidate not in used:
                    replacement = candidate
                    flipped_bit = int(bit)
                    break
            if replacement is None:
                raise RuntimeError('Could not find a unique valid serialized index bit flip')
            byte_offset = slot['index_offset'] + flipped_bit // 8
            raw[byte_offset] ^= 1 << (flipped_bit % 8)
            used.remove(old_index)
            used.add(replacement)
            slot['index'] = replacement
            detail.update(old_index=old_index, new_index=replacement,
                          flipped_bit=flipped_bit, byte_offset=byte_offset,
                          bit_in_byte=flipped_bit % 8)
        if len(examples) < 20:
            examples.append(detail)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(bytes(raw))
    return dict(
        attack=attack, rate=rate, requested_count=count, seed=seed,
        early_steps=early_steps,
        target_steps_zero_based=sorted(step_ids), target_frames=sorted(frame_ids),
        defense='none', defense_steps=0, transport='serialized-tdcm',
        total_symbols=total_symbols, total_payload_bits=total_payload_bits,
        protection_overhead_bits=0, protected_payload_bits=total_payload_bits,
        channel_flipped_bits=requested, changed_symbols=requested,
        changed_bits=requested, corrected_logical_symbols=0,
        payload_BER=requested / total_payload_bits,
        requested_symbol_rate=requested / total_symbols,
        eligible_symbols=len(eligible), examples=examples,
        input_path=str(input_path), output_path=str(output_path),
        defense_model='none', selection_metadata=selection_metadata,
    )


def apply_codebook_stream_attack(step_data, attack, K, rate=0.0, seed=42,
                                 early_steps=0, defense='none', defense_steps=1,
                                 count=0, target_steps=None, target_frames=None):
    """Return a copied codebook trajectory with a small number of bit flips.

    ``rate`` is the fraction of all transmitted index/sign symbols selected
    for corruption. ``early_steps`` restricts candidate slots while retaining
    the all-stream denominator, so results remain comparable across settings.
    """
    copied=[[(list(indices),list(signs)) for indices,signs in step]
            for step in step_data]
    total_symbols=sum(len(indices) for step in copied for indices,_ in step)
    bits_per_index=max(1,math.ceil(math.log2(K)))
    total_payload_bits=total_symbols*(bits_per_index+1)
    if defense not in STREAM_DEFENSE_CHOICES:raise ValueError(f'Unknown stream defense: {defense}')
    if defense_steps<1:raise ValueError('defense_steps must be positive')
    protected_steps=min(defense_steps,len(copied)) if defense!='none' else 0
    signs_per_step=(sum(len(signs) for _,signs in copied[0]) if copied else 0)
    protection_overhead_bits=(2*protected_steps*signs_per_step if defense=='sign-repetition3' else 0)
    metadata=dict(attack=attack,rate=rate,requested_count=count,seed=seed,
                  early_steps=early_steps,
                  defense=defense,defense_steps=protected_steps,
                  total_symbols=total_symbols,total_payload_bits=total_payload_bits,
                  protection_overhead_bits=protection_overhead_bits,
                  protected_payload_bits=total_payload_bits+protection_overhead_bits,
                  channel_flipped_bits=0,changed_symbols=0,changed_bits=0,
                  corrected_logical_symbols=0,payload_BER=0.0,examples=[],
                  defense_model=('idealized physical-channel repetition simulation; '
                                 'serialized .tdcm stores the majority-decoded logical payload'
                                 if defense!='none' else 'none'))
    if attack=='none':return copied,metadata
    if attack not in STREAM_ATTACK_CHOICES:raise ValueError(f'Unknown stream attack: {attack}')
    if count<0:raise ValueError('Stream attack count cannot be negative')
    if count==0 and not 0<rate<=1:raise ValueError('Stream attack requires count > 0 or rate in (0,1]')
    if early_steps>0 and target_steps:raise ValueError('early_steps and target_steps are mutually exclusive')
    if target_steps:
        step_ids={int(v) for v in target_steps}
        invalid=sorted(v for v in step_ids if v<0 or v>=len(copied))
        if invalid:raise ValueError(f'Invalid zero-based SDE steps: {invalid}')
    else:
        max_step=min(early_steps,len(copied)) if early_steps>0 else len(copied)
        step_ids=set(range(max_step))
    n_frames=len(copied[0]) if copied else 0
    if target_frames:
        frame_ids={int(v) for v in target_frames}
        invalid=sorted(v for v in frame_ids if v<0 or v>=n_frames)
        if invalid:raise ValueError(f'Invalid latent frame positions: {invalid}')
    else:frame_ids=set(range(n_frames))
    slots=[(s,f,m) for s in sorted(step_ids)
           for f,(indices,_) in enumerate(copied[s]) if f in frame_ids
           for m in range(len(indices))]
    metadata['target_steps_zero_based']=sorted(step_ids)
    metadata['target_frames']=sorted(frame_ids)
    requested=int(count) if count>0 else max(1,int(round(total_symbols*rate)))
    rng=np.random.default_rng(seed)
    if defense=='sign-repetition3':
        if attack!='sign-bitflip':raise ValueError('sign-repetition3 currently protects sign-bitflip attacks only')
        if any(s>=protected_steps for s in step_ids):
            raise ValueError('For this pilot, attacked steps must all be protected by sign-repetition3')
        physical_slots=[(s,f,m,copy) for s,f,m in slots for copy in range(3)]
        if requested>len(physical_slots):
            raise ValueError(f'Requested {requested} flips but only {len(physical_slots)} protected bits are eligible')
        chosen=rng.choice(len(physical_slots),size=requested,replace=False)
        flip_counts={}
        for physical_id in chosen:
            s,f,m,copy=physical_slots[int(physical_id)]
            flip_counts[(s,f,m)]=flip_counts.get((s,f,m),0)+1
            if len(metadata['examples'])<20:
                metadata['examples'].append(dict(step=s,frame=f,atom=m,copy=copy))
        residual=0
        touched=len(flip_counts)
        for (s,f,m),count in flip_counts.items():
            if count>=2:
                copied[s][f][1][m]=-int(copied[s][f][1][m]);residual+=1
        metadata['channel_flipped_bits']=requested
        metadata['changed_bits']=requested
        metadata['changed_symbols']=residual
        metadata['corrected_logical_symbols']=touched-residual
        metadata['eligible_symbols']=len(slots)
        metadata['eligible_physical_bits']=len(physical_slots)
        metadata['payload_BER']=requested/(total_payload_bits+protection_overhead_bits)
        metadata['requested_symbol_rate']=requested/total_symbols
        return copied,metadata
    if requested>len(slots):raise ValueError(f'Requested {requested} corruptions but only {len(slots)} eligible slots')
    chosen=rng.choice(len(slots),size=requested,replace=False)
    for slot_id in chosen:
        s,f,m=slots[int(slot_id)]
        indices,signs=copied[s][f]
        old_index=int(indices[m]);old_sign=int(signs[m])
        if attack=='sign-bitflip':
            signs[m]=-old_sign
            detail=dict(step=s,frame=f,atom=m,old_sign=old_sign,new_sign=int(signs[m]))
        else:
            occupied=set(indices);occupied.discard(old_index)
            bit_order=rng.permutation(bits_per_index)
            replacement=None;flipped_bit=None
            for bit in bit_order:
                candidate=old_index^(1<<int(bit))
                if candidate<K and candidate not in occupied:
                    replacement=candidate;flipped_bit=int(bit);break
            if replacement is None:
                raise RuntimeError('Could not find a unique valid one-bit index replacement')
            indices[m]=replacement
            detail=dict(step=s,frame=f,atom=m,old_index=old_index,
                        new_index=int(replacement),flipped_bit=flipped_bit)
        if len(metadata['examples'])<20:metadata['examples'].append(detail)
    metadata['changed_symbols']=requested
    metadata['changed_bits']=requested
    metadata['channel_flipped_bits']=requested
    metadata['payload_BER']=requested/total_payload_bits
    metadata['requested_symbol_rate']=requested/total_symbols
    metadata['eligible_symbols']=len(slots)
    return copied,metadata
