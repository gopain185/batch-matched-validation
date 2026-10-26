"""Linear discriminant heads and the transductive rules whose settings are being chosen.

Every rule reads a whole test batch with estimated group ids (content clusters, `clusters`). `score` subtracts a share w of each group's mean
log-probability; `twin` adds w times the log-probabilities of a second head fitted on group-centred training
features, read on test features minus their group mean. Both act only on clusters of `lo` to 18 items, so `lo` is a
guard against clusters too small (or too impure) for their mean to be mostly the shared effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from .sim import Batch

HI = 18


def clusters(c: np.ndarray, cut: float = 0.85) -> np.ndarray:
    """Average-linkage clusters of the content features under cosine distance, cut at `cut`."""
    if len(c) < 2:
        return np.zeros(len(c), int)
    return fcluster(linkage(c, method="average", metric="cosine"), cut, criterion="distance") - 1


def purity(groups: np.ndarray, truth: np.ndarray) -> float:
    """Share of item pairs put in the same cluster that really share a group."""
    same = groups[:, None] == groups[None, :]
    np.fill_diagonal(same, False)
    return float((same & (truth[:, None] == truth[None, :])).sum() / max(same.sum(), 1))


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def group_mean(v: np.ndarray, group: np.ndarray) -> np.ndarray:
    _, inv, counts = np.unique(group, return_inverse=True, return_counts=True)
    sums = np.zeros((counts.size, v.shape[1]))
    np.add.at(sums, inv, v)
    return (sums / counts[:, None])[inv]


def group_size(group: np.ndarray) -> np.ndarray:
    _, inv, counts = np.unique(group, return_inverse=True, return_counts=True)
    return counts[inv]


class LDA:
    def __init__(self, x: np.ndarray, y: np.ndarray, K: int):
        self.means = np.stack([x[y == k].mean(0) for k in range(K)])
        r = x - self.means[y]
        self.prec = np.linalg.inv(r.T @ r / len(x) + 1e-6 * np.eye(x.shape[1]))

    def log_proba(self, x: np.ndarray) -> np.ndarray:
        a = x @ self.prec @ self.means.T - 0.5 * np.einsum("kd,de,ke->k", self.means, self.prec, self.means)
        return log_softmax(a)


@dataclass(frozen=True)
class Config:
    rule: str  # "plain", "score" or "twin"
    w: float = 0.0
    lo: int = 2

    def __str__(self) -> str:
        return "plain" if self.rule == "plain" else f"{self.rule} w={self.w:g} lo={self.lo}"


def grid(weights=(0.25, 0.5, 1.0, 2.0), los=(2, 3, 4, 6)) -> list[Config]:
    return [Config("plain")] + [Config(r, w, lo) for r in ("score", "twin") for w in weights for lo in los]


class Heads:
    def __init__(self, train: Batch, K: int):
        self.plain = LDA(train.x, train.y, K)
        self.centred = LDA(train.x - group_mean(train.x, train.group), train.y, K)

    def prepare(self, b: Batch, groups: np.ndarray) -> dict:
        """Everything the rules need for one batch and its estimated groups, shared by every configuration."""
        lp = self.plain.log_proba(b.x)
        return {"lp": lp, "lp_mean": group_mean(lp, groups), "size": group_size(groups),
                "lc": self.centred.log_proba(b.x - group_mean(b.x, groups))}

    @staticmethod
    def scores(p: dict, c: Config) -> np.ndarray:
        if c.rule == "plain":
            return p["lp"]
        on = ((p["size"] >= c.lo) & (p["size"] <= HI))[:, None]
        if c.rule == "score":
            return p["lp"] - on * c.w * p["lp_mean"]
        if c.rule == "twin":
            return p["lp"] + on * c.w * p["lc"]
        raise ValueError(c.rule)
