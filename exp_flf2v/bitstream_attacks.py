"""Controlled corruption attacks on transmitted Turbo-DDCM codebook data."""
import math
import numpy as np


STREAM_ATTACK_CHOICES = ['none', 'index-bitflip', 'sign-bitflip']


def apply_codebook_stream_attack(step_data, attack, K, rate, seed=42, early_steps=0):
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
    metadata=dict(attack=attack,rate=rate,seed=seed,early_steps=early_steps,
                  total_symbols=total_symbols,total_payload_bits=total_payload_bits,
                  changed_symbols=0,changed_bits=0,payload_BER=0.0,examples=[])
    if attack=='none':return copied,metadata
    if attack not in STREAM_ATTACK_CHOICES:raise ValueError(f'Unknown stream attack: {attack}')
    if not 0<rate<=1:raise ValueError('Stream attack rate must be in (0,1]')
    max_step=min(early_steps,len(copied)) if early_steps>0 else len(copied)
    slots=[(s,f,m) for s in range(max_step) for f,(indices,_) in enumerate(copied[s])
           for m in range(len(indices))]
    requested=max(1,int(round(total_symbols*rate)))
    if requested>len(slots):raise ValueError(f'Requested {requested} corruptions but only {len(slots)} eligible slots')
    rng=np.random.default_rng(seed)
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
    metadata['payload_BER']=requested/total_payload_bits
    metadata['eligible_symbols']=len(slots)
    return copied,metadata
