import numpy as np

from bmv.experiments import embedded, one_world, size_hist
from bmv.experiments import test_sizes as draw_sizes
from bmv.methods import Config, Heads, clusters, grid, group_mean, purity
from bmv.sim import World, balanced_accuracy, batch, new_means, subsample


def world(seed=0, **kw):
    w = World(**kw)
    rng = np.random.default_rng(seed)
    mu = new_means(w, rng)
    return w, mu, rng


def test_batch_groups_have_distinct_authors():
    w, mu, rng = world()
    b = batch(w, mu, np.array([18, 6, 1]), rng)
    assert len(b.y) == 25 and b.c.shape == (25, w.content_dim)
    for g in range(3):
        ys = b.y[b.group == g]
        assert len(set(ys)) == len(ys)


def test_subsample_cuts_each_group():
    w, mu, rng = world()
    b = subsample(batch(w, mu, np.full(5, 18), rng), np.array([2, 3, 18, 1, 6]), rng)
    assert np.bincount(b.group).tolist() == [2, 3, 18, 1, 6]


def test_group_mean_and_centred_features():
    x = np.array([[1.0], [3.0], [10.0]])
    assert group_mean(x, np.array([0, 0, 1])).ravel().tolist() == [2.0, 2.0, 10.0]


def test_clean_content_gives_pure_clusters():
    w, mu, rng = world(noise_c=0.3)
    b = batch(w, mu, np.full(20, 6), rng)
    assert purity(clusters(b.c, cut=0.5), b.group) > 0.95


def test_plain_config_is_the_plain_head_and_lo_guards_small_groups():
    w, mu, rng = world()
    heads = Heads(batch(w, mu, np.full(w.train_groups, w.K), rng), w.K)
    b = batch(w, mu, np.array([2] * 50), rng)
    p = heads.prepare(b, b.group)
    assert np.array_equal(Heads.scores(p, Config("plain")), p["lp"])
    assert np.array_equal(Heads.scores(p, Config("twin", 1.0, lo=3)), p["lp"])
    assert not np.array_equal(Heads.scores(p, Config("twin", 1.0, lo=2)), p["lp"])


def test_twin_on_true_full_groups_beats_plain():
    w, mu, rng = world()
    heads = Heads(batch(w, mu, np.full(w.train_groups, w.K), rng), w.K)
    b = batch(w, mu, np.full(60, 18), rng)
    p = heads.prepare(b, b.group)
    plain = balanced_accuracy(b.y, Heads.scores(p, Config("plain")).argmax(1))
    twin = balanced_accuracy(b.y, Heads.scores(p, Config("twin", 1.0)).argmax(1))
    assert twin > plain + 0.05


def test_test_sizes_and_histogram():
    rng = np.random.default_rng(1)
    s = draw_sizes("all 3", 300, rng)
    assert set(s) == {3} and s.sum() >= 300
    h = size_hist(np.array([0, 0, 1, 2, 2, 2]), np.arange(6))
    assert np.isclose(h.sum(), 1) and np.isclose(h[0], 1 / 6) and np.isclose(h[1], 2 / 6)


def test_embedded_picks_small_cut_for_a_pair_test():
    w, mu, rng = world()
    heads = Heads(batch(w, mu, np.full(w.train_groups, w.K), rng), w.K)
    val = batch(w, mu, np.full(30, w.K), rng)
    test = batch(w, mu, np.full(300, 2), rng)
    acc, m = embedded(heads, val, test, grid(), rng)
    assert m in (2, 3) and acc.shape == (len(grid()),)


def test_one_world_reports_every_rule():
    out = one_world(World(), "all 6", 30, np.random.default_rng(3), grid())
    for r in ["full", "small", "grid_mean", "worst_case", "matched", "embedded"]:
        assert out["oracle"] >= out[r] - 1e-12
