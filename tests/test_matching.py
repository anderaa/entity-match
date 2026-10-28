"""Unit tests for the matching pipeline."""

import numpy as np
import pandas as pd
import pytest

import matching as mt


@pytest.fixture
def entities() -> pd.DataFrame:
    """Four entities: E1/E2 share a building, E3 shares E1's phone."""
    rows = [
        ("E1", "name", "Acme Widgets, Inc."),
        ("E1", "address", "12 Main St, Ste 400, Denver, CO 80202"),
        ("E1", "phone", "(303) 555-0100"),
        ("E2", "name", "Borealis Design LLC"),
        ("E2", "address", "12 Main Street, Unit 7, Denver, CO 80202"),
        ("E3", "name", "Cascade Freight Co"),
        ("E3", "phone", "303-555-0100"),
        ("E4", "name", "Dynamo Tools Corporation"),
        ("E4", "website", "https://www.dynamotools.com/"),
        ("E4", "tax_id", "12-3456789"),
    ]
    return pd.DataFrame(rows, columns=["entity_id", "attr_type", "value"])


@pytest.fixture
def norm_entities(entities: pd.DataFrame) -> pd.DataFrame:
    return mt.normalize(entities, "entity_id")


class TestNormalize:
    def test_legal_suffixes_and_punctuation_dropped(self) -> None:
        df = pd.DataFrame([("E1", "name", "Acme Widgets, Inc.")], columns=["entity_id", "attr_type", "value"])
        assert mt.normalize(df, "entity_id").norm.iloc[0] == "acme widgets"

    def test_ampersand_becomes_and(self) -> None:
        df = pd.DataFrame([("E1", "name", "Merlt & Velostac Coffee")], columns=["entity_id", "attr_type", "value"])
        assert mt.normalize(df, "entity_id").norm.iloc[0] == "merlt and velostac coffee"

    def test_suite_stripped_so_co_tenants_collide(self, norm_entities: pd.DataFrame) -> None:
        addr = norm_entities[norm_entities.attr_type == "address"].set_index("id").norm
        assert addr["E1"] == addr["E2"] == "12 main st denver co 80202"

    def test_phone_keeps_last_ten_digits(self, norm_entities: pd.DataFrame) -> None:
        phone = norm_entities[norm_entities.attr_type == "phone"].set_index("id").norm
        assert phone["E1"] == phone["E3"] == "3035550100"

    def test_website_scheme_and_www_removed(self, norm_entities: pd.DataFrame) -> None:
        site = norm_entities[norm_entities.attr_type == "website"].set_index("id").norm
        assert site["E4"] == "dynamotools.com"

    def test_tax_id_reduced_to_digits(self, norm_entities: pd.DataFrame) -> None:
        tax = norm_entities[norm_entities.attr_type == "tax_id"].set_index("id").norm
        assert tax["E4"] == "123456789"

    def test_empty_values_dropped(self) -> None:
        df = pd.DataFrame([("E1", "name", "Inc."), ("E2", "name", "Acme")],
                          columns=["entity_id", "attr_type", "value"])
        assert mt.normalize(df, "entity_id").id.tolist() == ["E2"]


class TestBlocking:
    def test_exact_block_finds_shared_address(self, norm_entities: pd.DataFrame) -> None:
        pairs = mt.block_exact(norm_entities, norm_entities, ["address"])
        assert {("E1", "E2"), ("E2", "E1")} <= set(zip(pairs.qid, pairs.eid))

    def test_rare_token_block_survives_a_suffix_change(self, norm_entities: pd.DataFrame) -> None:
        q = mt.normalize(
            pd.DataFrame([("Q1", "name", "Dynamo Tools LLC")], columns=["query_id", "attr_type", "value"]),
            "query_id",
        )
        pairs = mt.block_name_tokens(q, norm_entities)
        assert pairs.eid.tolist() == ["E4"]

    def test_common_tokens_are_ignored(self, norm_entities: pd.DataFrame) -> None:
        q = mt.normalize(
            pd.DataFrame([("Q1", "name", "Dynamo Tools")], columns=["query_id", "attr_type", "value"]),
            "query_id",
        )
        assert mt.block_name_tokens(q, norm_entities, max_df=0).empty


class TestCompare:
    @pytest.fixture
    def levels(self, norm_entities: pd.DataFrame) -> pd.DataFrame:
        queries = pd.DataFrame(
            [
                ("Q1", "name", "Acme Widgets Inc"),
                ("Q1", "address", "12 Main St, Denver, CO 80202"),
                ("Q1", "phone", "3035550100"),
                ("Q2", "name", "Acme Widget Inc"),
                ("Q3", "name", "Totally Unrelated Group"),
            ],
            columns=["query_id", "attr_type", "value"],
        )
        q = mt.normalize(queries, "query_id")
        pairs = pd.DataFrame({"qid": ["Q1", "Q1", "Q2", "Q3"], "eid": ["E1", "E3", "E1", "E1"]})
        return mt.compare(pairs, q, norm_entities, mt.value_tf(norm_entities)).set_index(["qid", "eid"])

    def test_exact_agreement_is_the_top_level(self, levels: pd.DataFrame) -> None:
        name = next(c for c in mt.COMPARISONS if c.attr == "name")
        assert levels.loc[("Q1", "E1"), "name"] == name.n_levels - 1

    def test_missing_attribute_is_null_not_disagreement(self, levels: pd.DataFrame) -> None:
        # E3 has no address; the query does
        assert levels.loc[("Q1", "E3"), "address"] == 0

    def test_near_miss_name_earns_a_fuzzy_level(self, levels: pd.DataFrame) -> None:
        lev = levels.loc[("Q2", "E1"), "name"]
        assert 2 <= lev < next(c for c in mt.COMPARISONS if c.attr == "name").n_levels - 1

    def test_unrelated_name_disagrees(self, levels: pd.DataFrame) -> None:
        assert levels.loc[("Q3", "E1"), "name"] == 1

    def test_tf_recorded_only_for_exact_agreement(self, levels: pd.DataFrame) -> None:
        assert levels.loc[("Q1", "E1"), "tf_name"] == pytest.approx(0.25)
        assert np.isnan(levels.loc[("Q3", "E1"), "tf_name"])


class TestValueTf:
    def test_shared_value_counts_both_holders(self, norm_entities: pd.DataFrame) -> None:
        tf = mt.value_tf(norm_entities).set_index(["attr_type", "norm"]).tf
        assert tf[("phone", "3035550100")] == pytest.approx(0.5)
        assert tf[("name", "acme widgets")] == pytest.approx(0.25)


class TestLevelDist:
    def test_pseudo_count_keeps_unseen_levels_finite(self) -> None:
        levels = pd.DataFrame({c.attr: np.ones(10, dtype=int) for c in mt.COMPARISONS})
        d = mt.level_dist(levels)
        assert np.isfinite(d["phone"]).all()
        assert (d["phone"][1:] > 0).all()

    def test_null_level_is_one_so_it_contributes_no_weight(self) -> None:
        levels = pd.DataFrame({c.attr: np.zeros(10, dtype=int) for c in mt.COMPARISONS})
        assert mt.level_dist(levels)["phone"][0] == 1.0

    def test_distribution_over_non_null_levels_sums_to_one(self) -> None:
        rng = np.random.default_rng(0)
        levels = pd.DataFrame({c.attr: rng.integers(0, c.n_levels, 500) for c in mt.COMPARISONS})
        d = mt.level_dist(levels)
        assert d["name"][1:].sum() == pytest.approx(1.0)


class TestEm:
    @pytest.fixture
    def two_attr_comparisons(self) -> list:
        return [mt.Comparison("address", ["null", "disagree", "agree"]),
                mt.Comparison("phone", ["null", "disagree", "agree"])]

    def test_recovers_m_on_the_worked_example(self, two_attr_comparisons: list) -> None:
        # Section 9.2 of docs/methods.md: 12 name-blocked pairs, u fixed at 2e-5
        counts = {(2, 2): 6, (2, 1): 2, (1, 2): 1, (1, 1): 3}
        rows = [[a, p] for (a, p), n in counts.items() for _ in range(n)]
        levels = pd.DataFrame(rows, columns=["address", "phone"])
        u = {a: np.array([1.0, 1 - 2e-5, 2e-5]) for a in ("address", "phone")}
        res = mt.em(levels, u, active=["address", "phone"], lam_init=0.5, pseudo=0.0,
                    m_init={a: np.array([1.0, 0.1, 0.9]) for a in ("address", "phone")},
                    comparisons=two_attr_comparisons)
        assert res.m["address"][2] == pytest.approx(0.857, abs=0.01)
        assert res.m["phone"][2] == pytest.approx(0.750, abs=0.01)
        assert res.lam == pytest.approx(0.778, abs=0.01)

    def test_converges_before_the_iteration_cap(self, two_attr_comparisons: list) -> None:
        levels = pd.DataFrame({"address": [2] * 20 + [1] * 5, "phone": [2] * 18 + [1] * 7})
        u = {a: np.array([1.0, 1 - 2e-5, 2e-5]) for a in ("address", "phone")}
        res = mt.em(levels, u, active=["address", "phone"], max_iter=200, comparisons=two_attr_comparisons)
        assert res.n_iter < 200
        assert len(res.history) == res.n_iter

    def test_inactive_attributes_are_not_estimated(self, two_attr_comparisons: list) -> None:
        levels = pd.DataFrame({"address": [2, 1, 2], "phone": [2, 2, 1]})
        u = {a: np.array([1.0, 1 - 2e-5, 2e-5]) for a in ("address", "phone")}
        res = mt.em(levels, u, active=["address"], comparisons=two_attr_comparisons)
        assert set(res.m) == {"address"}


class TestWeights:
    def test_agreement_weights_are_forced_non_decreasing(self) -> None:
        # raw u makes the fuzzy level outweigh the exact one
        m = {"name": np.array([1.0, 0.05, 0.15, 0.30, 0.50])}
        u = {"name": np.array([1.0, 0.99, 1e-6, 1e-4, 1e-3])}
        comps = [next(c for c in mt.COMPARISONS if c.attr == "name")]
        raw = mt.level_weights(m, u, monotone=False, comparisons=comps)["name"]
        capped = mt.level_weights(m, u, monotone=True, comparisons=comps)["name"]
        assert raw[2] > raw[4]
        assert np.all(np.diff(capped[2:]) >= 0)
        assert capped[4] == raw[4]

    def test_null_level_weighs_nothing(self) -> None:
        m = {"phone": np.array([1.0, 0.2, 0.8])}
        u = {"phone": np.array([1.0, 0.99, 2e-5])}
        comps = [mt.Comparison("phone", ["null", "disagree", "agree"])]
        assert mt.level_weights(m, u, comparisons=comps)["phone"][0] == 0.0

    def test_weight_table_is_one_row_per_level(self) -> None:
        m = {c.attr: np.r_[1.0, np.full(c.n_levels - 1, 1 / (c.n_levels - 1))] for c in mt.COMPARISONS}
        u = {c.attr: np.r_[1.0, np.full(c.n_levels - 1, 1 / (c.n_levels - 1))] for c in mt.COMPARISONS}
        tbl = mt.weight_table(m, u)
        assert len(tbl) == sum(c.n_levels for c in mt.COMPARISONS)
        assert (tbl.weight.abs() < 1e-12).all()


class TestScore:
    @pytest.fixture
    def params(self) -> tuple[dict, dict]:
        m = {"name": np.array([1.0, 0.05, 0.10, 0.25, 0.60]),
             "address": np.array([1.0, 0.15, 0.20, 0.65]),
             "phone": np.array([1.0, 0.25, 0.75]),
             "email": np.array([1.0, 0.25, 0.75]),
             "website": np.array([1.0, 0.25, 0.75]),
             "tax_id": np.array([1.0, 0.05, 0.95])}
        u = {"name": np.array([1.0, 0.99, 1e-3, 1e-4, 2e-6]),
             "address": np.array([1.0, 0.99, 1e-3, 5e-5]),
             "phone": np.array([1.0, 1 - 2e-5, 2e-5]),
             "email": np.array([1.0, 1 - 2e-5, 2e-5]),
             "website": np.array([1.0, 1 - 2e-5, 2e-5]),
             "tax_id": np.array([1.0, 1 - 1e-6, 1e-6])}
        return m, u

    def _levels(self, **over) -> pd.DataFrame:
        row = {c.attr: 0 for c in mt.COMPARISONS}
        row |= {f"tf_{c.attr}": np.nan for c in mt.COMPARISONS}
        row |= {"qid": "Q1", "eid": "E1"}
        row |= over
        return pd.DataFrame([row])

    def test_shared_value_agreement_scores_far_below_a_unique_one(self, params: tuple[dict, dict]) -> None:
        m, u = params
        unique = mt.score(self._levels(address=3, tf_address=1 / 50_000), m, u, n_entities=50_000)
        shared = mt.score(self._levels(address=3, tf_address=240 / 50_000), m, u, n_entities=50_000)
        assert unique.score.iloc[0] > shared.score.iloc[0] + 5

    def test_unique_value_falls_back_to_the_attribute_u(self, params: tuple[dict, dict]) -> None:
        m, u = params
        tf_scored = mt.score(self._levels(name=4, tf_name=1 / 50_000), m, u, n_entities=50_000)
        plain = mt.score(self._levels(name=4, tf_name=np.nan), m, u, n_entities=50_000, use_tf=False)
        assert tf_scored.score.iloc[0] == pytest.approx(plain.score.iloc[0])

    def test_null_attributes_add_nothing(self, params: tuple[dict, dict]) -> None:
        m, u = params
        assert mt.score(self._levels(), m, u, n_entities=50_000).score.iloc[0] == 0.0

    def test_disagreement_is_negative_and_volatility_sized(self, params: tuple[dict, dict]) -> None:
        m, u = params
        phone = mt.score(self._levels(phone=1), m, u, n_entities=50_000).score.iloc[0]
        tax = mt.score(self._levels(tax_id=1), m, u, n_entities=50_000).score.iloc[0]
        assert tax < phone < 0


class TestProbability:
    def test_to_prob_is_monotone_in_score(self) -> None:
        p = mt.to_prob(np.array([-10.0, 0.0, 10.0, 30.0]), prior=1e-5)
        assert np.all(np.diff(p) > 0)
        assert 0 < p[0] < p[-1] < 1

    def test_zero_score_returns_the_prior(self) -> None:
        assert mt.to_prob(np.array([0.0]), prior=1e-5)[0] == pytest.approx(1e-5)

    def test_tied_candidates_split_probability(self, norm_entities: pd.DataFrame) -> None:
        m = {c.attr: np.r_[1.0, np.full(c.n_levels - 1, 1 / (c.n_levels - 1))] for c in mt.COMPARISONS}
        m["address"] = np.array([1.0, 0.15, 0.20, 0.65])
        u = {c.attr: np.r_[1.0, 0.99, *np.full(c.n_levels - 2, 0.01 / (c.n_levels - 2))] for c in mt.COMPARISONS}
        q = mt.normalize(
            pd.DataFrame([("Q1", "address", "12 Main St, Denver, CO 80202")],
                         columns=["query_id", "attr_type", "value"]),
            "query_id",
        )
        pairs = pd.DataFrame({"qid": ["Q1", "Q1"], "eid": ["E1", "E2"]})
        levels = mt.compare(pairs, q, norm_entities, mt.value_tf(norm_entities))
        scored = mt.score(levels, m, u, n_entities=norm_entities.id.nunique())
        post, rho = mt.query_posterior(scored, q, norm_entities, m, u, rho=0.85)
        assert post.p.iloc[0] == pytest.approx(post.p.iloc[1])
        assert post.p.sum() < 1.0
        assert rho == 0.85


class TestDecide:
    @pytest.fixture
    def posterior(self) -> pd.DataFrame:
        return pd.DataFrame({
            "qid": ["Q1", "Q2", "Q2", "Q3", "Q4", "Q4"],
            "eid": ["E1", "E2", "E3", "E4", "E5", "E6"],
            "score": [30.0, 12.0, 11.0, 2.0, 14.0, 14.0],
            "p": [0.99, 0.80, 0.10, 0.20, 0.45, 0.45],
        })

    def test_one_row_per_query_with_the_top_candidate(self, posterior: pd.DataFrame) -> None:
        out = mt.decide(posterior).set_index("qid")
        assert out.index.tolist() == ["Q1", "Q2", "Q3", "Q4"]
        assert out.loc["Q2", "eid"] == "E2"

    def test_thresholds_assign_the_three_decisions(self, posterior: pd.DataFrame) -> None:
        out = mt.decide(posterior).set_index("qid")
        assert out.loc["Q1", "decision"] == "accept"
        assert out.loc["Q2", "decision"] == "review"
        assert out.loc["Q3", "decision"] == "no_match"

    def test_a_split_tie_is_rejected_without_a_margin_rule(self, posterior: pd.DataFrame) -> None:
        out = mt.decide(posterior).set_index("qid")
        assert out.loc["Q4", "decision"] == "no_match"

    def test_second_place_probability_is_zero_when_alone(self, posterior: pd.DataFrame) -> None:
        assert mt.decide(posterior).set_index("qid").loc["Q1", "p_second"] == 0.0
