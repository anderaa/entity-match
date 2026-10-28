"""
The worked example from section 9 of docs/methods.md, run through the code.

Every number asserted here is quoted in the paper. If one of these fails,
either the implementation drifted or the paper needs updating.
"""

import numpy as np
import pandas as pd
import pytest

import matching as mt

N = 50_000
RHO = 0.85

# m and u exactly as tabulated in section 9.3; the three attributes the
# example does not use are given neutral values so they contribute nothing
M = {"name": np.array([1.0, 0.02, 0.01, 0.07, 0.90]),
     "address": np.array([1.0, 0.08, 0.05, 0.87]),
     "phone": np.array([1.0, 0.27, 0.73]),
     "email": np.array([1.0, 0.5, 0.5]),
     "website": np.array([1.0, 0.5, 0.5]),
     "tax_id": np.array([1.0, 0.5, 0.5])}
U = {"name": np.array([1.0, 0.9994, 0.0005, 0.00002, 0.000002]),
     "address": np.array([1.0, 0.99997, 0.000001, 0.00002]),
     "phone": np.array([1.0, 0.99998, 0.00002]),
     "email": np.array([1.0, 0.5, 0.5]),
     "website": np.array([1.0, 0.5, 0.5]),
     "tax_id": np.array([1.0, 0.5, 0.5])}

# section 9.4: the four relevant entities, plus filler to reach N with the
# coverage the example assumes (name 1.00, address 0.90, phone 0.77)
EXAMPLE_ENTITIES = [
    ("E1", "name", "Acme Tools Inc"), ("E1", "address", "12 Main St, Denver"), ("E1", "phone", "303-555-0100"),
    ("E2", "name", "Acme Tooling LLC"), ("E2", "phone", "303-555-0100"),
    ("E3", "name", "Bolt Supply Co"), ("E3", "address", "12 Main St Ste 200, Denver"),
    ("E3", "phone", "303-555-0199"),
    ("E4", "name", "Cobalt Dental"), ("E4", "address", "12 Main St Ste 310, Denver"),
]


@pytest.fixture(scope="module")
def world() -> dict:
    """Build the paper's reference table, its two queries, and everything derived from them."""
    rows = list(EXAMPLE_ENTITIES)
    n_addr, n_phone = round(0.90 * N) - 3, round(0.77 * N) - 3
    for i in range(N - 4):
        eid = f"F{i:06d}"
        rows.append((eid, "name", f"Filler{i} Holdings Inc"))
        if i < n_addr:
            rows.append((eid, "address", f"{i} Filler Rd, Rome, NY 1{i:04d}"))
        if i < n_phone:
            rows.append((eid, "phone", f"2{i:09d}"))
    queries = pd.DataFrame(
        # 9.4: name, address and a phone that points at a different entity
        [("Q1", "name", "ACME TOOLS"), ("Q1", "address", "12 Main Street, Denver"),
         ("Q1", "phone", "(303) 555-0199"),
         # 9.5: a shared building address and nothing else
         ("Q2", "address", "12 Main St, Denver")],
        columns=["query_id", "attr_type", "value"])
    e = mt.normalize(pd.DataFrame(rows, columns=["entity_id", "attr_type", "value"]), "entity_id")
    q = mt.normalize(queries, "query_id")
    tf = mt.value_tf(e)
    cand = pd.concat([mt.block_exact(q, e, [c.attr for c in mt.COMPARISONS]),
                      mt.block_name_tokens(q, e)]).drop_duplicates()
    levels = mt.compare(cand, q, e, tf)
    scored = mt.score(levels, M, U, n_entities=N)
    posterior, _ = mt.query_posterior(scored, q, e, M, U, rho=RHO)
    return {"e": e, "q": q, "levels": levels.set_index(["qid", "eid"]),
            "scored": scored.set_index(["qid", "eid"]), "posterior": posterior,
            "cand": cand, "b0": mt.background_bf(q, e, M, U)}


class TestSection93Weights:
    """The weight table in section 9.3."""

    @pytest.mark.parametrize("attr, level, want", [
        ("name", 1, -5.64), ("name", 2, 4.32), ("name", 3, 11.77), ("name", 4, 18.78),
        ("address", 1, -3.64), ("address", 3, 15.41),
        ("phone", 1, -1.89), ("phone", 2, 15.16),
    ])
    def test_weight(self, attr: str, level: int, want: float) -> None:
        assert mt.level_weights(M, U)[attr][level] == pytest.approx(want, abs=0.005)

    def test_fuzzy_address_is_capped_at_the_exact_level(self) -> None:
        assert np.log2(M["address"][2] / U["address"][2]) == pytest.approx(15.61, abs=0.005)
        assert mt.level_weights(M, U)["address"][2] == pytest.approx(15.41, abs=0.005)


class TestSection94Scoring:
    """Scoring one query, where the name and the phone point at different entities."""

    def test_entity_coverage_matches_the_example(self, world: dict) -> None:
        s = world["e"].groupby("attr_type").id.nunique() / N
        assert (s["name"], round(s["address"], 2), round(s["phone"], 2)) == (1.0, 0.90, 0.77)

    def test_candidate_set(self, world: dict) -> None:
        assert sorted(world["cand"].query("qid == 'Q1'").eid) == ["E1", "E2", "E3", "E4"]

    @pytest.mark.parametrize("eid, levels", [
        ("E1", {"name": 4, "address": 3, "phone": 1}),   # exact name, shared address, stale phone
        ("E2", {"name": 2, "address": 0, "phone": 1}),   # near-miss name, no address on file
        ("E3", {"name": 1, "address": 3, "phone": 2}),   # different company, same building and phone
        ("E4", {"name": 1, "address": 3, "phone": 0}),   # different company, same building
    ])
    def test_comparison_levels(self, world: dict, eid: str, levels: dict) -> None:
        got = world["levels"].loc[("Q1", eid)]
        assert {a: int(got[a]) for a in levels} == levels

    def test_shared_address_is_discounted_by_one_bit(self, world: dict) -> None:
        assert world["levels"].loc[("Q1", "E1"), "tf_address"] * N == pytest.approx(3)
        u_v = max((3 - 1) / N, U["address"][3])
        assert u_v == pytest.approx(0.00004)
        assert np.log2(M["address"][3] / u_v) == pytest.approx(14.41, abs=0.005)
        assert mt.level_weights(M, U)["address"][3] - 14.41 == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize("eid, want", [("E1", 31.30), ("E2", 2.43), ("E3", 23.92), ("E4", 8.77)])
    def test_score(self, world: dict, eid: str, want: float) -> None:
        assert world["scored"].loc[("Q1", eid), "score"] == pytest.approx(want, abs=0.01)

    def test_background_factor(self, world: dict) -> None:
        assert world["b0"]["Q1"] == pytest.approx(0.0015, abs=5e-5)
        assert world["b0"]["Q2"] == pytest.approx(0.172, abs=5e-4)

    @pytest.mark.parametrize("eid, want, tol", [
        ("E1", 0.994, 0.001), ("E3", 0.006, 0.0005), ("E4", 1.6e-7, 1e-7), ("E2", 2.0e-9, 1e-9)])
    def test_posterior(self, world: dict, eid: str, want: float, tol: float) -> None:
        p = world["posterior"].query("qid == 'Q1'").set_index("eid").p
        assert p[eid] == pytest.approx(want, abs=tol)

    def test_probability_of_nothing_in_the_list(self, world: dict) -> None:
        p = world["posterior"].query("qid == 'Q1'").p
        assert 1 - p.sum() == pytest.approx(0.000003, abs=5e-7)

    def test_the_pairwise_posterior_calls_both_candidates_near_certain(self, world: dict) -> None:
        s = world["scored"].loc["Q1", "score"]
        pw = pd.Series(mt.to_prob(s.values, RHO / N), index=s.index)
        assert pw["E1"] == pytest.approx(0.99998, abs=1e-5)
        assert pw["E3"] == pytest.approx(0.996, abs=0.001)

    def test_decision_is_accept(self, world: dict) -> None:
        d = mt.decide(world["posterior"], accept_p=0.95, review_p=0.5).set_index("qid")
        assert d.loc["Q1", "decision"] == "accept"
        assert d.loc["Q1", "eid"] == "E1"


class TestSection95Tie:
    """A query with only a shared building address: three candidates, none identifiable."""

    def test_three_way_tie(self, world: dict) -> None:
        p = world["posterior"].query("qid == 'Q2'").set_index("eid").p
        assert sorted(p.index) == ["E1", "E3", "E4"]
        assert list(p.round(3)) == [pytest.approx(0.263, abs=0.001)] * 3

    def test_score_and_bayes_factor(self, world: dict) -> None:
        s = world["scored"].loc[("Q2", "E1"), "score"]
        assert s == pytest.approx(14.41, abs=0.005)
        assert 2 ** s == pytest.approx(21_770, abs=50)

    def test_remainder_goes_to_none_listed(self, world: dict) -> None:
        assert 1 - world["posterior"].query("qid == 'Q2'").p.sum() == pytest.approx(0.211, abs=0.002)

    def test_decision_is_no_match(self, world: dict) -> None:
        d = mt.decide(world["posterior"], accept_p=0.95, review_p=0.5).set_index("qid")
        assert d.loc["Q2", "decision"] == "no_match"


class TestEstimatorsMatchTheirFormulas:
    """Sections 6.1, 6.2 and 6.4, checked against the formulas as written."""

    @pytest.fixture
    def levels(self) -> pd.DataFrame:
        rng = np.random.default_rng(3)
        return pd.DataFrame({c.attr: rng.integers(0, c.n_levels, 4000) for c in mt.COMPARISONS})

    def test_61_u_estimator(self, levels: pd.DataFrame) -> None:
        got = mt.level_dist(levels)
        for c in mt.COMPARISONS:
            n = np.array([(levels[c.attr].values == lv).sum() for lv in range(c.n_levels)], float)
            paper = (1 + n[1:]) / ((c.n_levels - 1) + n[1:].sum())
            assert got[c.attr][1:] == pytest.approx(paper)
            assert got[c.attr][0] == 1.0

    def test_62_em_converges_to_a_fixed_point_of_the_papers_steps(self, levels: pd.DataFrame) -> None:
        u = mt.level_dist(levels)
        active = [c.attr for c in mt.COMPARISONS if c.attr != "name"]
        res = mt.em(levels, u, active, tol=1e-10, max_iter=500)
        comps = [c for c in mt.COMPARISONS if c.attr in active]
        g = {c.attr: levels[c.attr].values for c in comps}
        num = res.lam * np.prod([res.m[a][g[a]] for a in g], axis=0)
        p = num / (num + (1 - res.lam) * np.prod([u[a][g[a]] for a in g], axis=0))
        assert p.mean() == pytest.approx(res.lam, abs=1e-6)
        for c in comps:
            cnt = np.array([p[g[c.attr] == lv].sum() for lv in range(c.n_levels)])
            paper = (1 + cnt[1:]) / ((c.n_levels - 1) + cnt[1:].sum())
            assert res.m[c.attr][1:] == pytest.approx(paper, abs=1e-6)

    def test_64_cap_is_the_suffix_minimum(self) -> None:
        m = {"name": np.array([1.0, 0.05, 0.15, 0.30, 0.50])}
        u = {"name": np.array([1.0, 0.99, 1e-6, 1e-4, 1e-3])}
        comps = [next(c for c in mt.COMPARISONS if c.attr == "name")]
        raw = np.log2(m["name"] / u["name"])
        got = mt.level_weights(m, u, comparisons=comps)["name"]
        assert [got[lv] for lv in (2, 3, 4)] == [pytest.approx(raw[lv:].min()) for lv in (2, 3, 4)]
