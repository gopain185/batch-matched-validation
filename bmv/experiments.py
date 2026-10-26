"""The sweeps behind the README tables. Each returns one DataFrame; `python -m bmv` runs them all.

In every world a training batch fits the heads (true groups, as a labelled train has prompt ids), a validation batch of full groups (every author once per group)
chooses a configuration, and a test batch with a given group-size distribution scores the choice. Selection rules:

- `full`: validate on the full groups, as a prompt-grouped CV does.
- `small`: validate on groups cut to 2 and to 3 items (mean of the two).
- `grid_mean`: mean over groups cut to 18, 6, 3 and 2.
- `worst_case`: the configuration whose lowest score over those four cuts is highest.
- `matched`: cut the validation groups to sizes drawn from the test batch's own cluster-size histogram, which is
  visible without labels.

- `embedded`: mix the cut validation items into the unlabelled test batch, cluster them together and score only the
  validation items, so they meet the same crowding as the test; the cut m (2, 3, 6 or 18) is the one whose
  validation items' cluster-size histogram is closest to the test items'.

Every validation cut is clustered by content just as the test is, so the rules see the same kind of estimated
groups on both sides.

`oracle` picks on the test labels and is the ceiling; regret is oracle minus the rule's test score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .methods import Heads, clusters, grid, purity
from .sim import Batch, World, balanced_accuracy, batch, new_means, subsample

SEED = 20261027
WORLDS = 20
TEST_ROWS = 1800
DRAWS = 3
CUTS = [18, 6, 3, 2]

SCENARIOS = {
    "all 18": ([18], [1.0]),
    "all 6": ([6], [1.0]),
    "all 3": ([3], [1.0]),
    "all 2": ([2], [1.0]),
    "mixed 1-18": ([1, 2, 3, 6, 12, 18], [0.25, 0.25, 0.2, 0.15, 0.1, 0.05]),
    "mostly singletons": ([1, 2, 3, 4], [0.6, 0.25, 0.1, 0.05]),
}
RULES = ["full", "small", "grid_mean", "worst_case", "matched", "embedded"]
BINS = np.array([1, 2, 3, 4, 7, 13, 19, 10**9])  # cluster-size histogram bins: 1, 2, 3, 4-6, 7-12, 13-18, >18


def test_sizes(scenario: str, rows: int, rng: np.random.Generator) -> np.ndarray:
    values, probs = SCENARIOS[scenario]
    sizes = []
    while sum(sizes) < rows:
        sizes.append(int(rng.choice(values, p=probs)))
    return np.array(sizes)


def evaluate(heads: Heads, b, configs) -> np.ndarray:
    p = heads.prepare(b, clusters(b.c))
    return np.array([balanced_accuracy(b.y, Heads.scores(p, c).argmax(1)) for c in configs])


def size_hist(groups: np.ndarray, rows: np.ndarray) -> np.ndarray:
    size = np.bincount(groups)[groups[rows]]
    return np.histogram(size, BINS)[0] / len(rows)


def embedded(heads: Heads, val, test, configs, rng) -> tuple[np.ndarray, int]:
    """Validation items cut to m and clustered together with the test batch; scored on the validation items only."""
    best = None
    for m in CUTS:
        v = subsample(val, np.full(int(val.group.max()) + 1, m), rng)
        both = Batch(np.vstack([v.x, test.x]), np.vstack([v.c, test.c]), np.concatenate([v.y, test.y]),
                     np.concatenate([v.group, test.group + 10**6]))
        g = clusters(both.c)
        rows_v, rows_t = np.arange(len(v.y)), np.arange(len(v.y), len(both.y))
        gap = np.abs(size_hist(g, rows_v) - size_hist(g, rows_t)).sum()
        if best is None or gap < best[0]:
            p = heads.prepare(both, g)
            acc = np.array([balanced_accuracy(v.y, Heads.scores(p, c)[rows_v].argmax(1)) for c in configs])
            best = (gap, acc, m)
    return best[1], best[2]


def one_world(w: World, scenario: str, val_groups: int, rng: np.random.Generator, configs) -> dict:
    mu = new_means(w, rng)
    heads = Heads(batch(w, mu, np.full(w.train_groups, w.K), rng), w.K)
    val = batch(w, mu, np.full(val_groups, w.K), rng)
    sizes = test_sizes(scenario, TEST_ROWS, rng)
    test = batch(w, mu, sizes, rng)
    test_groups = clusters(test.c)
    observed = np.bincount(test_groups)

    def cut(m_or_sizes) -> np.ndarray:
        return np.mean([evaluate(heads, subsample(val, np.broadcast_to(m_or_sizes, val_groups) if np.isscalar(m_or_sizes)
                                                  else m_or_sizes(), rng), configs) for _ in range(DRAWS)], 0)

    by_cut = {m: evaluate(heads, val, configs) if m == w.K else cut(m) for m in CUTS}
    val_score = {
        "full": by_cut[18],
        "small": 0.5 * (by_cut[3] + by_cut[2]),
        "grid_mean": np.mean([by_cut[m] for m in CUTS], 0),
        "worst_case": np.min([by_cut[m] for m in CUTS], 0),
        "matched": cut(lambda: rng.choice(np.minimum(observed, w.K), val_groups)),
    }
    val_score["embedded"], m_emb = embedded(heads, val, test, configs, rng)
    t = evaluate(heads, test, configs)
    out = {"plain": t[0], "oracle": t.max(), "purity": purity(test_groups, test.group), "embedded_m": m_emb}
    for r in RULES:
        out[r] = t[int(np.argmax(val_score[r]))]
        out[r + "_choice"] = str(configs[int(np.argmax(val_score[r]))])
    return out


def run(scenario: str, w: World = World(), val_groups: int = 100, worlds: int = WORLDS, key: int = 0) -> dict:
    configs = grid()
    rows = [one_world(w, scenario, val_groups, np.random.default_rng([SEED, key, i]), configs) for i in range(worlds)]
    df = pd.DataFrame(rows)
    res = {"embedded_m": df["embedded_m"].value_counts().index[0], "purity": df["purity"].mean(), "plain": df["plain"].mean(), "oracle_gain": (df["oracle"] - df["plain"]).mean()}
    for r in RULES:
        regret = df["oracle"] - df[r]
        res[r + "_regret"] = regret.mean()
        res[r + "_regret_se"] = regret.std(ddof=1) / np.sqrt(len(df))
        res[r + "_top"] = df[r + "_choice"].value_counts().index[0]
    return res


def _table(name: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.insert(0, "experiment", name)
    return df


def exp1_scenarios() -> pd.DataFrame:
    """Regret of each selection rule for six test group-size distributions (tau = 1, 100 validation groups)."""
    return _table("scenarios", [{"setting": s, **run(s, key=i)} for i, s in enumerate(SCENARIOS)])


def exp2_validation_size() -> pd.DataFrame:
    """How many validation groups the rules need, on the mixed test."""
    return _table("validation_size", [{"setting": f"{n} groups", **run("mixed 1-18", val_groups=n, key=100 + n)}
                                      for n in (30, 100, 300)])


def exp3_group_effect() -> pd.DataFrame:
    """Strength of the shared group effect, on the mixed test."""
    return _table("group_effect", [{"setting": f"tau {tau:g}", **run("mixed 1-18", World(tau=tau), key=200 + i)}
                                   for i, tau in enumerate((0.0, 0.5, 1.0, 2.0))])


ALL = [exp1_scenarios, exp2_validation_size, exp3_group_effect]
