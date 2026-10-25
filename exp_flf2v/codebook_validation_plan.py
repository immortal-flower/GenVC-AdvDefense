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
