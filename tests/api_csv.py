"""Small CSV files for the API tests (bytes, built in memory)."""

import numpy as np

N_ROWS = 120
SEED = 7


def make_csv(n_rows: int = N_ROWS, *, seed: int = SEED) -> bytes:
    """A mixed CSV: two quantities, a category whose levels differ by case, a code-like integer column."""
    rng = np.random.default_rng(seed)
    lines = ["age,weight,sex,city,visits"]
    for _ in range(n_rows):
        sex = rng.choice(["Male", "male", "Female"])
        city = rng.choice(["Rome", "Milan", "Turin"])
        lines.append(f"{rng.integers(18, 90)},{rng.normal(70, 12):.1f},{sex},{city},{rng.integers(0, 6)}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def all_cells(data: bytes) -> list[list[str]]:
    """Rows of ``data`` as lists of cells (header excluded)."""
    return [line.split(",") for line in data.decode("utf-8").splitlines()[1:]]
