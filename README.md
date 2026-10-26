# batch-matched-validation

Some test-time rules read the whole test batch: centring each item's scores or features inside its group (passages
answering the same prompt, images from the same site) removes a nuisance the group shares. Their settings (how much
to centre, from which group size on) are chosen on validation data, and the right setting depends on the group
structure of the test batch, which you do not control. In the Name That LLM competition, the setting my
prompt-grouped CV preferred (stronger centring) was the worst of four on the public LB, and the LB order followed my
CV recomputed with prompts cut to two or three answers. This repo asks, on a simulation where the truth is known:
which way of validating chooses well for a test batch whose group sizes are unknown?

## The simulation

`bmv/sim.py`. 18 authors and many groups. An item by author k in group g has style features
`x = mu[k] + u[g] + e` (16 dimensions, `mu[k] ~ N(0, 0.75^2)`, group effect `u[g] ~ N(0, tau^2)` with tau = 1 unless a
sweep changes it, unit noise) and content features `c = v[g] + 1.5 n` (100 dimensions) that say only which group the
item belongs to. Inside a group the authors are all different. Group ids are hidden at test time: groups are found by
clustering `c` (cosine distance, average linkage, cut 0.85), on the test and on every validation batch alike.

A training batch (150 full groups) fits two linear discriminant heads: a plain one, and a "twin" on group-centred
features. The configurations to choose from (`bmv/methods.py`): `plain`; `score` (subtract w times the cluster's mean
log-probability); `twin` (add w times the twin's log-probabilities on features minus their cluster mean); w in
{0.25, 0.5, 1, 2}; applied only to clusters of `lo` to 18 items, `lo` in {2, 3, 4, 6}. 33 configurations.

A validation batch of 100 full groups chooses one. The test batch has about 1,800 items with a group-size
distribution from the list below, and scores the choice. Regret is the test balanced accuracy of the best
configuration on the test (the oracle) minus that of the chosen one, averaged over 20 worlds (standard errors
0.0001–0.006, all in `results/`). `python -m bmv` reproduces every number in under half an hour on a laptop CPU.

Selection rules:

| rule | validates on |
| --- | --- |
| `full` | the full validation groups (what a prompt-grouped CV does) |
| `small` | groups cut to 2 and to 3 items (mean of the two) |
| `grid_mean` | mean over cuts to 18, 6, 3 and 2 items |
| `worst_case` | the configuration whose lowest score over those four cuts is highest |
| `matched` | groups cut to sizes drawn from the test's own cluster-size histogram (visible without labels) |
| `embedded` | groups cut to m items and **clustered together with the unlabelled test batch**, scored on the validation items only; m in {2, 3, 6, 18} is the one whose validation items' cluster-size histogram is closest to the test items' |

## Results

**1. Six test batches (tau = 1).** Regret of each rule; the last column is the cut the embedded rule picked most often.

| test groups | cluster purity | oracle gain over plain | full | small | grid_mean | worst_case | matched | embedded | embedded m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all 18 | 0.94 | +0.172 | 0.001 | 0.061 | 0.014 | 0.072 | 0.000 | **0.000** | 18 |
| all 6 | 0.72 | +0.129 | 0.008 | 0.029 | 0.006 | 0.045 | 0.002 | **0.002** | 6 |
| all 3 | 0.28 | +0.030 | 0.027 | 0.005 | 0.010 | 0.005 | 0.017 | **0.014** | 3 |
| all 2 | 0.13 | +0.009 | 0.037 | 0.015 | 0.030 | 0.009 | 0.037 | **0.008** | 2 |
| mixed 1–18 | 0.77 | +0.105 | 0.004 | 0.024 | 0.007 | 0.037 | 0.003 | **0.005** | 6 |
| mostly singletons | 0.15 | +0.008 | 0.029 | 0.012 | 0.030 | 0.009 | 0.034 | **0.009** | 2 |
| **worst row** | | | 0.037 | 0.061 | 0.030 | 0.072 | 0.037 | **0.014** | |

Every fixed rule wins in some rows and loses badly in others. Full-group validation picks strong centring
(`twin w=2`), which is right for full groups and costs 0.03–0.04 when the test groups are small, more than the
whole oracle gain there. Validating on small cuts is the mirror image. Matching the test's cluster-size histogram
does not fix it: in a crowded test of pairs, clusters merge, so the observed clusters look large and the rule picks
strong centring again. Only the embedded rule stays within 0.014 in every row, because its validation items are
clustered among the test items and meet the same crowding and the same merges.

**2. How many validation groups (mixed test).**

| validation groups | full | small | grid_mean | matched | embedded |
| --- | --- | --- | --- | --- | --- |
| 30 | 0.004 | 0.019 | 0.005 | 0.003 | 0.011 |
| 100 | 0.003 | 0.031 | 0.008 | 0.004 | 0.009 |
| 300 | 0.004 | 0.037 | 0.006 | 0.003 | 0.004 |

The embedded rule is the noisiest choice when it has few validation groups (it scores only the validation items,
diluted among the test); its regret shrinks with more of them.

**3. Strength of the group effect (mixed test).**

| tau | oracle gain over plain | full | small | grid_mean | matched | embedded |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | +0.005 | 0.012 | 0.005 | 0.004 | 0.003 | 0.004 |
| 0.5 | +0.028 | 0.009 | 0.012 | 0.005 | 0.007 | 0.008 |
| 1 | +0.107 | 0.004 | 0.025 | 0.007 | 0.005 | 0.005 |
| 2 | +0.237 | 0.010 | 0.010 | 0.001 | 0.001 | 0.006 |

With no group effect, full-group validation still picks heavy centring and pays 0.012 for nothing.

## What to take from it

- A test-time rule that reads the batch should be validated on batches that look like the test batch, not like the
  validation folds. Group size is one part of that; how crowded the batch is (how many other groups its clusters can
  merge with) is the other, and a fold of a few dozen groups is far less crowded than a test of a few hundred.
- If the test batch is available unlabelled (it is, inside a code competition's notebook), clustering the
  validation items together with it gives both parts for free, and the cut whose cluster-size histogram matches tells
  you what group size the test behaves like.
- Without that, prefer the mean over several cuts to either extreme: its worst regret here (0.030) is lower than
  full-group CV's (0.037) and much lower than validating on small groups only (0.061).

## Limits

The group effect is purely additive in a Gaussian feature space; content is independent of the author; the heads
are linear discriminant analysis; the embedded rule uses one cut size for the whole validation batch, so a test that
mixes very different group sizes is matched only on average.

## Layout

- `bmv/sim.py`: worlds, batches, subsampling, balanced accuracy
- `bmv/methods.py`: heads, the configurations, content clustering, purity
- `bmv/experiments.py`, `python -m bmv`: selection rules and sweeps; results in `results/exp*.csv`
- `tests/`: `python -m pytest`
- `kaggle/`: builder for the Kaggle dataset and notebook

MIT licence.
