import importlib.util
from pathlib import Path


MODULE_PATH = (Path(__file__).parents[1] / "exp_flf2v" /
               "analyze_validation_frame_propagation.py")
SPEC = importlib.util.spec_from_file_location("validation_frame_propagation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_represented_output_frames():
    assert MODULE.represented_output_frames(0, 33) == [0]
    assert MODULE.represented_output_frames(1, 33) == [1, 2, 3, 4]
    assert MODULE.represented_output_frames(3, 33) == [9, 10, 11, 12]
    assert MODULE.represented_output_frames(8, 33) == [29, 30, 31, 32]


def test_aggregate_and_cell_summary():
    detail = []
    for seed, losses in [(42, [1.0, 2.0, 3.0]), (123, [3.0, 4.0, 5.0])]:
        for frame, loss in enumerate(losses):
            detail.append({
                "role": "sensitive", "sde_step": 1, "latent_position": 0,
                "requested_bits": 16, "attack_seed": seed,
                "output_frame": frame, "is_direct_segment": frame == 0,
                "clean_psnr_db": 30.0, "attacked_psnr_db": 30.0 - loss,
                "psnr_loss_db": loss, "run_name": "synthetic",
            })
    frames = MODULE.aggregate_frames(detail)
    assert len(frames) == 3
    assert frames[0]["psnr_loss_mean_db"] == 2.0
    assert frames[0]["psnr_loss_std_db"] == 1.0
    cells = MODULE.summarize_cells(detail, 3)
    assert len(cells) == 1
    assert cells[0]["gop_psnr_loss_mean_db"] == 3.0
    assert cells[0]["direct_segment_loss_mean_db"] == 2.0
    assert cells[0]["outside_segment_loss_mean_db"] == 3.5
    assert cells[0]["worst_output_frame"] == 2
