"""
End-to-end check that the pipeline on the real dataset computes what sections
5 to 8 of docs/methods.md say it does.

Each test recomputes a quantity straight from the paper's formula and compares
it to what ``matching.py`` produced. Skipped unless the data has been generated
(``python generate_data.py --out data``).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matching as mt

DATA = Path(__file__).resolve().parent.parent / "data"
pytestmark = pytest.mark.skipif(not (DATA / "entities.csv").exists(),
                                reason="run generate_data.py --out data first")
ATTRS = [c.attr for c in mt.COMPARISONS]


@pytest.fixture(scope="module")
def run() -> dict:
    """Run the pipeline once on a sample small enough to keep the suite quick."""
    E, Q = pd.read_csv(DATA / "entities.csv"), pd.read_csv(DATA / "queries.csv")
    n = E.entity_id.nunique()
    e, q = mt.normalize(E, "entity_id"), mt.normalize(Q, "query_id")
    tf = mt.value_tf(e)
    cand = pd.concat([mt.block_exact(q, e, ATTRS), mt.block_name_tokens(q, e)]).drop_duplicates()
    levels = mt.compare(cand, q, e, tf)

    rng = np.random.default_rng(0)
    a, b = rng.choice(e.id.unique(), 300_000), rng.choice(e.id.unique(), 300_000)
    ee = mt.compare(pd.DataFrame({"qid": a[a != b], "eid": b[a != b]}).drop_duplicates(), e, e, tf)
    u = mt.level_dist(ee)

    rounds = {key: mt.em(levels.merge(mt.block_exact(q, e, [key]), on=["qid", "eid"]),
                         u, [a_ for a_ in ATTRS if a_ != key])
              for key in ("name", "phone")}
    m = {a_: np.mean([r.m[a_] for r in rounds.values() if a_ in r.m], axis=0) for a_ in ATTRS}

    scored = mt.score(levels, m, u, n_entities=n)
    posterior, rho = mt.query_posterior(scored, q, e, m, u)
    return {"n": n, "e": e, "q": q, "ee": ee, "u": u, "m": m, "rounds": rounds,
            "scored": scored, "posterior": posterior, "rho": rho}


def test_61_u_from_distinct_entity_pairs(run: dict) -> None:
    for c in mt.COMPARISONS:
        n_lv = np.array([(run["ee"][c.attr].values == lv).sum() for lv in range(c.n_levels)], float)
        paper = (1 + n_lv[1:]) / ((c.n_levels - 1) + n_lv[1:].sum())
        assert run["u"][c.attr][1:] == pytest.approx(paper), c.attr


def test_62_every_attribute_is_estimated_in_some_round(run: dict) -> None:
    assert all(any(a in r.m for r in run["rounds"].values()) for a in ATTRS)
    assert "name" not in run["rounds"]["name"].m
    assert "phone" not in run["rounds"]["phone"].m


def test_5_and_63_score_is_the_sum_of_value_adjusted_weights(run: dict) -> None:
    """S = sum_a w(gamma_a, v), with 6.3's value-specific u and 6.4's cap applied per pair."""
    s, m, u, n = run["scored"], run["m"], run["u"], run["n"]
    lw = mt.level_weights(m, u)
    total = np.zeros(len(s))
    for c in mt.COMPARISONS:
        lv = s[c.attr].values.astype(int)
        w = lw[c.attr][lv].copy()
        if c.tf:
            count = s[f"tf_{c.attr}"].values * n
            hit = (lv >= 2) & ~np.isnan(count)
            others = np.maximum((count[hit] - 1) / n, 0.0)
            per_level = np.array([np.log2(m[c.attr][k] / np.maximum(others, u[c.attr][k]))
                                  for k in range(2, c.n_levels)])
            # the cap: min over the pair's level and every level above it
            reachable = np.arange(2, c.n_levels)[:, None] >= lv[hit][None, :]
            w[hit] = np.where(reachable, per_level, np.inf).min(axis=0)
        total += w
    assert total == pytest.approx(s.score.values)


def test_64_a_near_miss_never_outscores_an_exact_hit_on_the_same_value(run: dict) -> None:
    """The reordering that the per-pair cap exists to prevent."""
    s, m, u, n = run["scored"], run["m"], run["u"], run["n"]
    for c in mt.COMPARISONS:
        if not c.tf or c.n_levels <= 3:
            continue
        top = c.n_levels - 1
        count = s[f"tf_{c.attr}"].values * n
        fuzzy = (s[c.attr].values >= 2) & (s[c.attr].values < top) & ~np.isnan(count)
        if not fuzzy.any():
            continue
        others = np.maximum((count[fuzzy] - 1) / n, 0.0)
        exact_on_same_value = np.log2(m[c.attr][top] / np.maximum(others, u[c.attr][top]))
        assert (s[f"w_{c.attr}"].values[fuzzy] <= exact_on_same_value + 1e-9).all(), c.attr


def test_5_a_missing_attribute_contributes_nothing(run: dict) -> None:
    s = run["scored"]
    for c in mt.COMPARISONS:
        null = s[s[c.attr] == 0]
        if len(null):
            assert (null[f"w_{c.attr}"] == 0.0).all(), c.attr


def test_7_background_factor(run: dict) -> None:
    q, e, m, u, n = run["q"], run["e"], run["m"], run["u"], run["n"]
    share = e.groupby("attr_type").id.nunique() / n
    ratio = {c.attr: m[c.attr][1] / u[c.attr][1] for c in mt.COMPARISONS}
    qa = q[["id", "attr_type"]].drop_duplicates()
    paper = (qa.assign(f=[share.get(a, 0) * ratio[a] + 1 - share.get(a, 0) for a in qa.attr_type])
             .groupby("id").f.prod())
    got = mt.background_bf(q, e, m, u)
    assert paper.sort_index().values == pytest.approx(got.sort_index().values)


def test_7_posterior_and_rho(run: dict) -> None:
    s, q, e, m, u, n = run["scored"], run["q"], run["e"], run["m"], run["u"], run["n"]
    b0 = mt.background_bf(q, e, m, u)
    bf = np.power(2.0, s.score.values)
    agg = pd.DataFrame({"qid": s.qid, "bf": bf}).groupby("qid").bf.agg(["sum", "size"])
    t_q = (n * b0).copy()
    t_q.loc[agg.index] = agg["sum"] + (n - agg["size"]) * b0.reindex(agg.index)

    rho = 0.5
    for _ in range(500):
        new = float(((rho / n * t_q) / (rho / n * t_q + 1 - rho)).mean())
        if abs(new - rho) < 1e-9:
            break
        rho = new
    assert rho == pytest.approx(run["rho"], abs=1e-9)
    paper = (rho / n * bf) / (rho / n * t_q.reindex(s.qid).values + 1 - rho)
    assert paper == pytest.approx(run["posterior"].p.values)


def test_8_candidate_probabilities_sum_to_at_most_one(run: dict) -> None:
    assert run["posterior"].groupby("qid").p.sum().max() <= 1.0 + 1e-9


def test_8_decisions_follow_the_thresholds(run: dict) -> None:
    p = run["posterior"]
    d = mt.decide(p, accept_p=0.95, review_p=0.5).set_index("qid")
    top = (p.sort_values(["qid", "p", "eid"], ascending=[True, False, True])
           .groupby("qid").head(1).set_index("qid"))
    assert (d.eid == top.eid).all()
    assert (d.decision.values == np.where(top.p >= 0.95, "accept",
                                          np.where(top.p >= 0.5, "review", "no_match"))).all()


def test_8_tie_breaking_is_deterministic(run: dict) -> None:
    p = run["posterior"]
    shuffled = mt.decide(p.sample(frac=1.0, random_state=1), accept_p=0.95, review_p=0.5)
    assert (mt.decide(p, accept_p=0.95, review_p=0.5).set_index("qid").eid
            == shuffled.set_index("qid").eid).all()
