import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "exp_flf2v" / "summarize_codebook_sensitivity_grid.py"
SPEC = importlib.util.spec_from_file_location("sensitivity_grid_summary", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_run_name_parser():
    match = MODULE.RUN_RE.fullmatch(
        "sensitivity_grid_sign16_step17_frame8_aseed123_jockey"
    )
    assert match is not None
    assert match.groups() == ("sign", "16", "17", "8", "123", None, "jockey")


def test_run_name_parser_with_gop():
    match = MODULE.RUN_RE.fullmatch(
        "sensitivity_grid_sign16_step1_frame3_aseed42_gop2_jockey"
    )
    assert match is not None
    assert match.groups() == ("sign", "16", "1", "3", "42", "2", "jockey")


def test_run_name_parser_rejects_old_one_dimensional_sweep():
    assert MODULE.RUN_RE.fullmatch("sensitivity_step1_frame3_sign16") is None


def test_group_rows():
    rows = [{"field": "sign", "seed": 7}, {"field": "sign", "seed": 42}]
    grouped = dict(MODULE._group_rows(rows, lambda row: row["field"]))
    assert grouped == {"sign": rows}

