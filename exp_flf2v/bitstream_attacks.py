"""Controlled corruption attacks on transmitted Turbo-DDCM codebook data."""
import math
import numpy as np


STREAM_ATTACK_CHOICES = ['none', 'index-bitflip', 'sign-bitflip']
STREAM_DEFENSE_CHOICES = ['none', 'sign-repetition3']


def apply_codebook_stream_attack(step_data, attack, K, rate, seed=42, early_steps=0,
                                 defense='none', defense_steps=1):
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
    metadata=dict(attack=attack,rate=rate,seed=seed,early_steps=early_steps,
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
    if not 0<rate<=1:raise ValueError('Stream attack rate must be in (0,1]')
    max_step=min(early_steps,len(copied)) if early_steps>0 else len(copied)
    slots=[(s,f,m) for s in range(max_step) for f,(indices,_) in enumerate(copied[s])
           for m in range(len(indices))]
    requested=max(1,int(round(total_symbols*rate)))
    rng=np.random.default_rng(seed)
    if defense=='sign-repetition3':
        if attack!='sign-bitflip':raise ValueError('sign-repetition3 currently protects sign-bitflip attacks only')
        if early_steps==0 or max_step>protected_steps:
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
    metadata['eligible_symbols']=len(slots)
    return copied,metadata
