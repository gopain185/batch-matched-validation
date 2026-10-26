"""A known-truth world of grouped items whose features share a group effect.

K classes (authors) and many groups (prompts, topics). An item by author k in group g has features
x = mu[k] + u[g] + e, with u[g] ~ N(0, tau^2 I) shared by the group and e ~ N(0, I), and content features
c = v[g] + noise_c * n that say only which group it belongs to. Inside a group the authors are all different, as in a
balanced competition train. Group ids are hidden at test time and are estimated by clustering c, and a batch can mix
groups of any size from 1 to K.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class World:
    K: int = 18
    d: int = 16
    sep: float = 0.75
    tau: float = 1.0
    content_dim: int = 100
    noise_c: float = 1.5
    train_groups: int = 150


@dataclass
class Batch:
    x: np.ndarray
    c: np.ndarray
    y: np.ndarray
    group: np.ndarray


def new_means(w: World, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, w.sep, (w.K, w.d))


def batch(w: World, mu: np.ndarray, sizes: np.ndarray, rng: np.random.Generator) -> Batch:
    """One group per entry of sizes; the authors inside a group are all different."""
    y = np.concatenate([rng.choice(w.K, s, replace=False) for s in sizes])
    group = np.repeat(np.arange(len(sizes)), sizes)
    u = rng.normal(0.0, w.tau, (len(sizes), w.d))
    x = mu[y] + u[group] + rng.normal(size=(len(y), w.d))
    v = rng.normal(size=(len(sizes), w.content_dim))
    c = v[group] + w.noise_c * rng.normal(size=(len(y), w.content_dim))
    return Batch(x, c, y, group)


def subsample(b: Batch, sizes: np.ndarray, rng: np.random.Generator) -> Batch:
    """Cut group g of b down to sizes[g] random items (b must hold at least that many)."""
    keep = np.concatenate([rng.choice(np.flatnonzero(b.group == g), s, replace=False) for g, s in enumerate(sizes)])
    return Batch(b.x[keep], b.c[keep], b.y[keep], b.group[keep])


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([(pred[y == k] == k).mean() for k in np.unique(y)]))
