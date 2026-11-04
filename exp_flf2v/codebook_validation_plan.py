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
    repetitions: int = 1
    disturbance_mode: str = "random"
    interleave_seed: int = 0
    burst_start: int = 0

    @property
    def role(self):
        if self.repetitions == 1:
            return "sparse-field-control"
        return f"sparse-repetition-{self.repetitions}"

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
        protection = ("unprotected" if self.repetitions == 1
                      else f"rep{self.repetitions}")
        if self.disturbance_mode == "contiguous-burst":
            return (f"sign_uep_{protection}_burst{self.count}_"
                    f"ileave{self.interleave_seed}_start{self.burst_start}"
                    f"{gop_tag}_{tag}")
        return (f"sign_uep_{protection}_count{self.count}_"
                f"aseed{self.attack_seed}{gop_tag}_{tag}")


@dataclass(frozen=True)
class PolicyValidationJob:
    policy: str
    protected_signs: int
    count: int
    attack_seed: int
    policy_seed: int = 42
    repetitions: int = 3
    cells: tuple = ()
    interleave_seed: int = 0

    @property
    def role(self):
        return f"same-budget-{self.policy}"

    @property
    def step(self):
        if not self.cells:
            return "policy"
        return "|".join(str(step + 1) for step, _ in self.cells)

    @property
    def frame(self):
        if not self.cells:
            return "policy"
        return "|".join(str(frame) for _, frame in self.cells)

    @property
    def disturbance_mode(self):
        return "uniform-sign-bsc"

    @property
    def burst_start(self):
        return 0

    def run_name(self, sequence: str, gop_index: int = 0) -> str:
        tag = "".join(ch.lower() if ch.isalnum() or ch in "_-" else "_"
                      for ch in sequence)
        gop_tag = "" if gop_index == 0 else f"_gop{gop_index}"
        return (f"sign_policy_{self.policy}_p{self.protected_signs}_"
                f"b{self.count}_pseed{self.policy_seed}_"
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


def expand_sparse_jobs(cells, counts, attack_seeds, symbols_per_cell=64,
                       repetitions=1, disturbance_mode="random",
                       interleave_seed=0, burst_starts=(0,)):
    pool_size = len(cells) * symbols_per_cell
    if any(int(count) < 1 or int(count) > pool_size for count in counts):
        raise ValueError(f"Sparse bit counts must lie in [1,{pool_size}]")
    if disturbance_mode not in ("random", "contiguous-burst"):
        raise ValueError(f"Unsupported sparse disturbance mode: {disturbance_mode}")
    if int(interleave_seed) < 0:
        raise ValueError("Sparse interleave seed cannot be negative")
    if disturbance_mode == "contiguous-burst":
        if repetitions < 3:
            raise ValueError("Contiguous burst validation requires repetition")
        starts = tuple(int(value) for value in burst_starts)
        if not starts or any(value < 0 for value in starts):
            raise ValueError("Sparse burst starts must be non-negative")
        redundancy_bits = len(cells) * symbols_per_cell * (repetitions - 1)
        for count in counts:
            for start in starts:
                if start + int(count) > redundancy_bits:
                    raise ValueError(
                        f"Burst [{start},{start + int(count)}) exceeds "
                        f"{redundancy_bits} redundancy bits")
        return [
            SparseValidationJob(
                tuple(cells), int(count), int(attack_seeds[0]),
                int(repetitions), disturbance_mode, int(interleave_seed),
                int(start))
            for count in counts
            for start in starts
        ]
    return [
        SparseValidationJob(
            tuple(cells), int(count), int(seed), int(repetitions),
            disturbance_mode, int(interleave_seed), 0)
        for count in counts
        for seed in attack_seeds
    ]


def expand_policy_jobs(policies, protected_signs, counts, attack_seeds,
                       policy_seeds=(42,), repetitions=3, cells=(),
                       interleave_seed=0):
    allowed = {"eep", "random", "early-step", "frozen-joint"}
    policies = tuple(str(policy) for policy in policies)
    invalid = sorted(set(policies) - allowed)
    if not policies or invalid:
        raise ValueError(f"Invalid protection policies: {invalid}")
    if int(protected_signs) < 1:
        raise ValueError("protected_signs must be positive")
    if repetitions < 3 or repetitions % 2 != 1:
        raise ValueError("Policy repetitions must be an odd integer >=3")
    if any(int(count) < 1 for count in counts):
        raise ValueError("Policy channel counts must be positive")
    cells = tuple(cells or ())
    if "frozen-joint" in policies and not cells:
        raise ValueError("frozen-joint policy requires frozen cells")
    jobs = []
    for count in counts:
        for attack_seed in attack_seeds:
            for policy in policies:
                seeds = policy_seeds if policy == "random" else (0,)
                for policy_seed in seeds:
                    jobs.append(PolicyValidationJob(
                        policy=policy, protected_signs=int(protected_signs),
                        count=int(count), attack_seed=int(attack_seed),
                        policy_seed=int(policy_seed), repetitions=int(repetitions),
                        cells=cells if policy == "frozen-joint" else (),
                        interleave_seed=int(interleave_seed)))
    return jobs


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
