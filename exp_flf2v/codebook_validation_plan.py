"""Pure-Python helpers for targeted codebook validation plans."""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValidationJob:
    role: str
    step: int
    frame: int
    count: int
    attack_seed: int

    def run_name(self, sequence: str, gop_index: int = 0) -> str:
        tag = "".join(ch.lower() if ch.isalnum() or ch in "_-" else "_"
                      for ch in sequence)
        gop_tag = "" if gop_index == 0 else f"_gop{gop_index}"
        return (f"sensitivity_grid_sign{self.count}_step{self.step}_"
                f"frame{self.frame}_aseed{self.attack_seed}{gop_tag}_{tag}")


@dataclass(frozen=True)
class SparseValidationJob:
    cells: tuple
    count: int
    attack_seed: int
    role: str = "sparse-field-control"

    @property
    def step(self):
        return "|".join(str(step + 1) for step, _ in self.cells)

    @property
    def frame(self):
        return "|".join(str(frame) for _, frame in self.cells)

    def run_name(self, sequence: str, gop_index: int = 0) -> str:
        tag = "".join(ch.lower() if ch.isalnum() or ch in "_-" else "_"
                      for ch in sequence)
        gop_tag = "" if gop_index == 0 else f"_gop{gop_index}"
        return (f"sign_uep_unprotected_count{self.count}_"
                f"aseed{self.attack_seed}{gop_tag}_{tag}")


def parse_sparse_cells(tokens, max_step=17, max_frame=8):
    cells = []
    for token in tokens or []:
        try:
            step_text, frame_text = token.split(":", 1)
            step, frame = int(step_text), int(frame_text)
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid sparse cell {token!r}; use STEP:FRAME")
        if not 1 <= step <= max_step:
            raise ValueError(f"Sparse SDE step must lie in [1,{max_step}], got {step}")
        if not 0 <= frame <= max_frame:
            raise ValueError(f"Sparse latent position must lie in [0,{max_frame}], got {frame}")
        cells.append((step - 1, frame))
    if not cells:
        raise ValueError("Sparse validation requires at least one cell")
    if len(set(cells)) != len(cells):
        raise ValueError("Sparse validation cells contain duplicates")
    return tuple(cells)


def expand_sparse_jobs(cells, counts, attack_seeds, symbols_per_cell=64):
    pool_size = len(cells) * symbols_per_cell
    if any(int(count) < 1 or int(count) > pool_size for count in counts):
        raise ValueError(f"Sparse bit counts must lie in [1,{pool_size}]")
    return [
        SparseValidationJob(tuple(cells), int(count), int(seed))
        for count in counts
        for seed in attack_seeds
    ]


def load_cells(path):
    cells = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            step = int(row["sde_step"])
            frame = int(row["latent_position"])
            if not 1 <= step <= 17:
                raise ValueError(f"SDE step must lie in [1,17], got {step}")
            if not 0 <= frame <= 8:
                raise ValueError(f"Latent position must lie in [0,8], got {frame}")
            cells.append((row.get("role", "validation"), step, frame))
    if not cells:
        raise ValueError("Validation plan contains no cells")
    if len({(step, frame) for _, step, frame in cells}) != len(cells):
        raise ValueError("Validation plan contains duplicate step/frame cells")
    return cells


def expand_jobs(cells, counts, attack_seeds):
    jobs = []
    for count in counts:
        if not 1 <= int(count) <= 64:
            raise ValueError(f"Bit count must lie in [1,64], got {count}")
        for seed in attack_seeds:
            for role, step, frame in cells:
                jobs.append(ValidationJob(
                    role=role, step=step, frame=frame,
                    count=int(count), attack_seed=int(seed)))
    return jobs
