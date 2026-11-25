"""Temporal crop geometry for the 33-frame VAE experiment."""


def crop_start(tile_id, round_id, window):
    if window not in (5,9,17) or tile_id < 0 or round_id < 0:
        raise ValueError('Expected nonnegative indices and window 5,9,17')
    # Same offset/stride rule as the original five-frame experiment.
    return (tile_id*7+(8+round_id)*5) % (34-window)
