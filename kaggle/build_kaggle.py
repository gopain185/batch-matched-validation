"""Builds the Kaggle dataset and notebook for this project.

    python kaggle/build_kaggle.py [--public] [--notebook-only]

writes kaggle/dataset/ (two simulated worlds with their training, validation and test batches, the result tables
of the three sweeps and the datasheet) and kaggle/notebook/ (a CPU notebook that embeds the package source and
reruns the sweeps).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bmv.experiments import SEED, TEST_ROWS, test_sizes  # noqa: E402
from bmv.sim import World, batch, new_means  # noqa: E402

OUT = ROOT / "kaggle"
DATASET_ID = "locdunhat/grouped-batches-unknown-group-size"
NOTEBOOK_ID = "locdunhat/validating-test-time-centring-for-the-test-batch"
TITLE = "Validating Test-Time Centring for the Test Batch"


def _frame(world: str, split: str, b) -> pd.DataFrame:
    df = pd.concat([pd.DataFrame(b.x.round(4), columns=[f"x_{i}" for i in range(b.x.shape[1])]),
                    pd.DataFrame(b.c.round(4), columns=[f"c_{i}" for i in range(b.c.shape[1])])], axis=1)
    df.insert(0, "group", b.group)
    df.insert(0, "label", b.y)
    df.insert(0, "split", split)
    df.insert(0, "world", world)
    return df


def build_dataset(public: bool) -> Path:
    out = OUT / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, scenario in enumerate(["all 2", "mixed 1-18"]):
        rng = np.random.default_rng([SEED, 99, i])
        w = World()
        mu = new_means(w, rng)
        name = scenario.replace(" ", "_")
        parts += [_frame(name, "train", batch(w, mu, np.full(w.train_groups, w.K), rng)),
                  _frame(name, "validation", batch(w, mu, np.full(100, w.K), rng)),
                  _frame(name, "test", batch(w, mu, test_sizes(scenario, TEST_ROWS, rng), rng))]
    pd.concat(parts).to_parquet(out / "worlds.parquet", index=False)
    for f in sorted((ROOT / "results").glob("exp*.csv")):
        (out / f.name).write_bytes(f.read_bytes())
    datasheet = (OUT / "DATASHEET.md").read_text(encoding="utf-8")
    (out / "DATASHEET.md").write_text(datasheet, encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "Grouped Batches, Unknown Group Size",
        "id": DATASET_ID,
        "subtitle": "Known-truth grouped items for choosing test-time centring settings",
        "description": datasheet,
        "isPrivate": not public,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["benchmark", "classification"],
    }, indent=2), encoding="utf-8")
    return out


def _sources() -> dict[str, str]:
    return {f.relative_to(ROOT).as_posix(): f.read_text(encoding="utf-8") for f in sorted((ROOT / "bmv").glob("*.py"))}


SETUP = """

import os, sys
ROOT = "/kaggle/working/batch-matched-validation" if os.path.isdir("/kaggle/working") else os.path.abspath("batch-matched-validation")
for path, text in SOURCES.items():
    os.makedirs(os.path.dirname(os.path.join(ROOT, path)), exist_ok=True)
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        f.write(text)
sys.path.insert(0, ROOT)
"""

IMPORTS = """import time
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from bmv import experiments as ex
pd.set_option("display.precision", 4); pd.set_option("display.width", 250)
RULES = ex.RULES
results = {}
def run(fn):
    t = time.time(); df = fn(); print(fn.__name__, round(time.time() - t), "s")
    results[fn.__name__] = df
    return df[["setting", "purity", "plain", "oracle_gain"] + [r + "_regret" for r in RULES] + ["embedded_m"]]
"""

PLOT = """df = results["exp1_scenarios"]
x = np.arange(len(df)); width = 0.13
fig, ax = plt.subplots(figsize=(10, 4))
for i, r in enumerate(RULES):
    ax.bar(x + (i - 2.5) * width, df[r + "_regret"], width, yerr=2 * df[r + "_regret_se"], capsize=2, label=r)
ax.set_xticks(x, df["setting"]); ax.set_ylabel("regret (oracle minus chosen, balanced accuracy)")
ax.legend(ncol=3); plt.tight_layout(); plt.show()
"""

SAMPLE = """import glob
from bmv.methods import Heads, Config, clusters, purity
from bmv.sim import Batch, balanced_accuracy
hits = glob.glob("/kaggle/input/**/worlds.parquet", recursive=True)
if hits:
    df = pd.read_parquet(hits[0])
    xc = [c for c in df.columns if c.startswith("x_")]; cc = [c for c in df.columns if c.startswith("c_")]
    as_batch = lambda g: Batch(g[xc].to_numpy(), g[cc].to_numpy(), g.label.to_numpy(), g.group.to_numpy())
    for world, g in df.groupby("world"):
        tr, te = as_batch(g[g.split == "train"]), as_batch(g[g.split == "test"])
        heads, est = Heads(tr, 18), clusters(te.c)
        p = heads.prepare(te, est)
        row = {str(c): round(balanced_accuracy(te.y, Heads.scores(p, c).argmax(1)), 4)
               for c in [Config("plain"), Config("twin", 0.5, 2), Config("twin", 2.0, 3)]}
        print(world, "cluster purity", round(purity(est, te.group), 3), row)
else:
    print("dataset not attached")
"""


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", (OUT / "notebook_intro.md").read_text(encoding="utf-8")),
        ("code", "SOURCES = " + json.dumps(_sources(), indent=0) + SETUP),
        ("code", IMPORTS),
        ("markdown", "## 1. Six test batches\n\nRegret of each selection rule (20 worlds; the `*_se` columns of "
                     "`results` hold the standard errors). `embedded_m` is the cut the embedded rule picked most often."),
        ("code", "run(ex.exp1_scenarios)"),
        ("code", PLOT),
        ("markdown", "## 2. How many validation groups\n\nMixed test batch."),
        ("code", "run(ex.exp2_validation_size)"),
        ("markdown", "## 3. Strength of the group effect\n\nMixed test batch."),
        ("code", "run(ex.exp3_group_effect)"),
        ("markdown", "## 4. The worlds in the dataset\n\nTwo saved worlds (train, validation and test batches); "
                     "three configurations on their test batches."),
        ("code", SAMPLE),
        ("markdown", (OUT / "notebook_findings.md").read_text(encoding="utf-8")),
    ]


def build_notebook(public: bool) -> Path:
    out = OUT / "notebook"
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for i, (kind, src) in enumerate(_cells()):
        c = {"cell_type": kind, "id": f"cell-{i}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        cells.append(c)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / "batch-matched-validation.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": NOTEBOOK_ID, "title": TITLE, "code_file": "batch-matched-validation.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": not public, "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [DATASET_ID], "competition_sources": [], "kernel_sources": [],
        "keywords": ["benchmark"],
    }, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    public = "--public" in sys.argv
    if "--notebook-only" not in sys.argv:
        print("wrote", build_dataset(public))
    print("wrote", build_notebook(public))
