"""Controlled corruption attacks on transmitted Turbo-DDCM codebook data."""
import math
import struct
from pathlib import Path

import numpy as np


STREAM_ATTACK_CHOICES = ['none', 'index-bitflip', 'sign-bitflip',
                         'sign-impact-bitflip']
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


def select_impact_sign_slots(step_data, codebook, count, step_idx, frame_idx):
    """Greedily choose signs that maximally rotate normalized codebook noise.

    The objective is the MSE between the original unit-variance combined noise
    and the combined noise after a candidate set of sign flips.  It uses only
    public bitstream indices/signs and the shared codebook seed—not the source
    video, reconstruction metrics, or Wan gradients.
    """
    import torch

    indices, signs = step_data[step_idx][frame_idx]
    if count < 1 or count > len(signs):
        raise ValueError(f'Impact count must be in [1,{len(signs)}]')
    with torch.no_grad():
        atoms = codebook.regenerate_selected_atoms(
            indices, step_idx, frame_idx).float()
        signs_t = torch.tensor(signs, device=atoms.device, dtype=torch.float32)
        current_sum = (signs_t.unsqueeze(1) * atoms).sum(0)
        original = current_sum / current_sum.std().clamp_min(1e-8)
        remaining = list(range(len(signs)))
        chosen = []
        trajectory = []
        for _ in range(count):
            rem_t = torch.tensor(remaining, device=atoms.device, dtype=torch.long)
            candidate_sums = (current_sum.unsqueeze(0)
                              - 2.0 * signs_t[rem_t].unsqueeze(1) * atoms[rem_t])
            candidate_std = candidate_sums.std(dim=1, keepdim=True).clamp_min(1e-8)
            candidate_noise = candidate_sums / candidate_std
            mse = ((candidate_noise - original.unsqueeze(0)) ** 2).mean(dim=1)
            best_local = int(mse.argmax().item())
            atom_position = remaining.pop(best_local)
            current_sum = current_sum - 2.0 * signs_t[atom_position] * atoms[atom_position]
            signs_t[atom_position] *= -1.0
            chosen.append((step_idx, frame_idx, atom_position))
            trajectory.append(float(mse[best_local].item()))
        final_noise = current_sum / current_sum.std().clamp_min(1e-8)
        final_mse = float(((final_noise - original) ** 2).mean().item())
        final_cosine = float(torch.nn.functional.cosine_similarity(
            original, final_noise, dim=0).item())
        selected_indices = [int(indices[position]) for _, _, position in chosen]
        del atoms, signs_t, current_sum, original, final_noise
    return chosen, dict(
        objective='greedy normalized-noise MSE',
        selected_atom_positions=[position for _, _, position in chosen],
        selected_codebook_indices=selected_indices,
        cumulative_noise_mse=trajectory,
        final_noise_mse=final_mse,
        final_noise_cosine=final_cosine,
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
    if attack not in ('index-bitflip', 'sign-bitflip', 'sign-impact-bitflip'):
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
        if attack != 'sign-impact-bitflip':
            raise ValueError('Explicit selected_slots are reserved for sign-impact-bitflip')
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
        if attack in ('sign-bitflip', 'sign-impact-bitflip'):
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
