"""
Fellegi-Sunter entity matching: normalization, blocking, comparison vectors,
parameter estimation (random-sampling u, EM for m), term-frequency adjustment,
and scoring.

Level coding for every comparison: ``0`` = null (either side missing),
``1`` = disagree, higher = stronger agreement. Because higher always means
better, the best level across multi-valued attributes is a plain ``max``.
"""

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler, Levenshtein

LEGAL_SUFFIXES = {"inc", "llc", "co", "corp", "corporation", "company", "ltd"}
STREET_LONG_TO_SHORT = {"street": "st", "avenue": "ave", "boulevard": "blvd",
                        "road": "rd", "drive": "dr", "lane": "ln"}

# --- normalization --------------------------------------------------------


def _norm_name(s: str) -> str:
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(t for t in s.split() if t not in LEGAL_SUFFIXES)


def _norm_address(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(ste|suite|unit|apt)\.?\s*\w+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(STREET_LONG_TO_SHORT.get(t, t) for t in s.split())


def _norm_phone(s: str) -> str:
    return re.sub(r"\D", "", s)[-10:]


def _norm_website(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"^https?://", "", s)
    return re.sub(r"^www\.", "", s).rstrip("/")


NORMALIZERS = {
    "name": _norm_name,
    "address": _norm_address,
    "phone": _norm_phone,
    "email": lambda s: s.lower().strip(),
    "website": _norm_website,
    "tax_id": lambda s: re.sub(r"\D", "", s),
}


def normalize(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Normalize a long-format attribute table.

    :param df: frame with ``id_col``, ``attr_type``, ``value``
    :param id_col: name of the id column
    :return: frame with columns ``id``, ``attr_type``, ``norm`` (deduplicated, empties dropped)
    """
    out = df.rename(columns={id_col: "id"}).copy()
    out["norm"] = [NORMALIZERS[a](str(v)) for a, v in zip(out.attr_type, out.value)]
    out = out[out.norm.str.len() > 0]
    return out[["id", "attr_type", "norm"]].drop_duplicates().reset_index(drop=True)


# --- blocking -------------------------------------------------------------


def block_exact(q: pd.DataFrame, e: pd.DataFrame, attrs) -> pd.DataFrame:
    """
    Candidate pairs that share an exact normalized value on any of ``attrs``.

    :param q: normalized query table
    :param e: normalized entity table
    :param attrs: attribute types to block on
    :return: frame of distinct ``qid``, ``eid`` pairs
    """
    qq = q[q.attr_type.isin(attrs)].rename(columns={"id": "qid"})
    ee = e[e.attr_type.isin(attrs)].rename(columns={"id": "eid"})
    return qq.merge(ee, on=["attr_type", "norm"])[["qid", "eid"]].drop_duplicates()


def block_name_tokens(q: pd.DataFrame, e: pd.DataFrame, max_df: int = 30) -> pd.DataFrame:
    """
    Candidate pairs that share a rare name token. Catches typos and suffix
    changes that exact name blocking misses, as long as one rare token survives.

    :param q: normalized query table
    :param e: normalized entity table
    :param max_df: tokens held by more than this many entities are ignored
    :return: frame of distinct ``qid``, ``eid`` pairs
    """
    def tokens(df, col):
        t = df[df.attr_type == "name"][["id", "norm"]].copy()
        t["tok"] = t.norm.str.split()
        return t.explode("tok")[["id", "tok"]].drop_duplicates().rename(columns={"id": col})

    et, qt = tokens(e, "eid"), tokens(q, "qid")
    df_ = et.groupby("tok").eid.nunique()
    et = et[et.tok.isin(df_[df_ <= max_df].index)]
    return qt.merge(et, on="tok")[["qid", "eid"]].drop_duplicates()


# --- comparison vectors ---------------------------------------------------


@dataclass
class Comparison:
    """
    Definition of one attribute comparison.

    :param attr: attribute type
    :param labels: level labels, index = level code (0 must be null, 1 disagree)
    :param kind: ``"exact"``, ``"name"`` (Jaro-Winkler) or ``"address"`` (Levenshtein)
    :param thresholds: similarity cutoffs for fuzzy levels, ascending, one per
        level between disagree and exact
    :param tf: apply term-frequency adjustment to the exact level
    """

    attr: str
    labels: list
    kind: str = "exact"
    thresholds: list = field(default_factory=list)
    tf: bool = True

    @property
    def n_levels(self) -> int:
        return len(self.labels)


COMPARISONS = [
    Comparison("name", ["null", "none", "jw>=0.85", "jw>=0.94", "exact"], "name", [0.85, 0.94]),
    Comparison("address", ["null", "disagree", "lev>=0.90", "exact"], "address", [0.90]),
    Comparison("phone", ["null", "disagree", "agree"]),
    Comparison("email", ["null", "disagree", "agree"]),
    Comparison("website", ["null", "disagree", "agree"]),
    Comparison("tax_id", ["null", "disagree", "agree"], tf=False),
]


def value_tf(e: pd.DataFrame) -> pd.DataFrame:
    """
    Term frequency per value: share of entities holding it.

    :param e: normalized entity table
    :return: frame ``attr_type``, ``norm``, ``tf``
    """
    n = e.id.nunique()
    tf = e.groupby(["attr_type", "norm"]).id.nunique().rename("tf") / n
    return tf.reset_index()


def compare(pairs: pd.DataFrame, q: pd.DataFrame, e: pd.DataFrame, tf: pd.DataFrame,
            comparisons=COMPARISONS) -> pd.DataFrame:
    """
    Build comparison vectors for (query, entity) pairs.

    :param pairs: frame with ``qid``, ``eid``
    :param q: normalized query table
    :param e: normalized entity table
    :param tf: output of :func:`value_tf`
    :param comparisons: comparison definitions
    :return: ``pairs`` plus one level column per attribute and a ``tf_<attr>``
        column holding the tf of the reference value the pair agreed on. Where
        several reference values reach the pair's level, the rarest is used.
        NaN when the level is null or disagree, where nothing was agreed on
    """
    out = pairs[["qid", "eid"]].reset_index(drop=True).copy()
    for c in comparisons:
        qa = q[q.attr_type == c.attr][["id", "norm"]].rename(columns={"id": "qid", "norm": "qv"})
        ea = e[e.attr_type == c.attr][["id", "norm"]].rename(columns={"id": "eid", "norm": "ev"})
        x = out[["qid", "eid"]].merge(qa, on="qid").merge(ea, on="eid")
        exact = (x.qv.values == x.ev.values)
        lev = np.where(exact, c.n_levels - 1, 1)
        if c.kind != "exact" and len(x):
            scorer = JaroWinkler.normalized_similarity if c.kind == "name" else Levenshtein.normalized_similarity
            sim = process.cpdist(x.qv.tolist(), x.ev.tolist(), scorer=scorer, workers=-1)
            for k, thr in enumerate(c.thresholds):
                lev = np.where(~exact & (sim >= thr), 2 + k, lev)
        x["lev"] = lev
        best = x.groupby(["qid", "eid"]).lev.max().rename(c.attr)
        out = out.merge(best, on=["qid", "eid"], how="left")
        out[c.attr] = out[c.attr].fillna(0).astype(np.int8)
        # tf of the rarest reference value that reaches the pair's level. the
        # reference side is the one that can be counted: at the exact level it
        # is the shared value, at a fuzzy level the query's own string is held
        # by nobody, so there would be nothing to count on that side
        xa = x.merge(best, on=["qid", "eid"])
        xa = xa[(xa.lev.values == xa[c.attr].values) & (xa.lev.values >= 2)]
        xa = xa.merge(tf[tf.attr_type == c.attr][["norm", "tf"]], left_on="ev", right_on="norm")
        rare = xa.groupby(["qid", "eid"]).tf.min().rename(f"tf_{c.attr}")
        out = out.merge(rare, on=["qid", "eid"], how="left")
    return out


# --- parameter estimation -------------------------------------------------


def level_dist(levels: pd.DataFrame, comparisons=COMPARISONS, weights=None, pseudo: float = 1.0) -> dict:
    """
    Distribution of non-null levels per attribute, P(level | not null).

    :param levels: comparison vectors from :func:`compare`
    :param comparisons: comparison definitions
    :param weights: optional per-row weights (e.g. match posteriors)
    :param pseudo: pseudo-count added to every non-null level. this bounds
        weights for levels that never occur in the sample; without it a
        level seen zero times gets an infinite weight
    :return: ``{attr: array over levels}`` with index 0 (null) set to 1.0
    """
    w = np.ones(len(levels)) if weights is None else np.asarray(weights)
    out = {}
    for c in comparisons:
        g = levels[c.attr].values
        counts = np.array([w[g == lv].sum() for lv in range(c.n_levels)], dtype=float)
        p = (counts + pseudo) / (counts[1:].sum() + pseudo * (c.n_levels - 1))
        p[0] = 1.0
        out[c.attr] = p
    return out


@dataclass
class EMResult:
    """
    EM output.

    :param m: ``{attr: P(level | match, not null)}``
    :param lam: estimated share of matches among the pairs EM saw
    :param n_iter: iterations run
    :param history: lambda per iteration
    """

    m: dict
    lam: float
    n_iter: int
    history: list


def em(levels: pd.DataFrame, u: dict, active, m_init: dict | None = None, lam_init: float = 0.1,
       max_iter: int = 200, tol: float = 1e-6, pseudo: float = 1.0,
       comparisons=COMPARISONS) -> EMResult:
    """
    Estimate m and lambda by EM with u held fixed.

    Only ``active`` attributes enter the likelihood. Leave the blocking
    attribute(s) out: every pair in the block agrees on them, so they carry no
    information about which pairs are matches, and including them with a
    random-sample u would make every pair look like a match.

    :param levels: comparison vectors
    :param u: ``{attr: P(level | non-match)}`` from random sampling
    :param active: attributes to use and estimate
    :param m_init: starting m (defaults to a generic agreement-heavy prior)
    :param lam_init: starting lambda
    :param max_iter: iteration cap
    :param tol: convergence tolerance on lambda and m
    :param pseudo: pseudo-count per level in the m-step (see :func:`level_dist`)
    :param comparisons: comparison definitions
    :return: :class:`EMResult`
    """
    comps = [c for c in comparisons if c.attr in active]
    g = {c.attr: levels[c.attr].values.astype(int) for c in comps}
    if m_init is None:
        m_init = {}
        for c in comps:
            p = np.linspace(1, 4, c.n_levels - 1) ** 2
            m_init[c.attr] = np.r_[1.0, p / p.sum()]
    m = {a: v.copy() for a, v in m_init.items() if a in g}
    lam, hist = lam_init, []
    log_u = {c.attr: np.log(u[c.attr]) for c in comps}
    for it in range(max_iter):
        # e-step
        lm = sum(np.log(m[a])[g[a]] for a in g)
        lu = sum(log_u[a][g[a]] for a in g)
        logit = np.log(lam) - np.log1p(-lam) + lm - lu
        p = 1 / (1 + np.exp(-np.clip(logit, -50, 50)))
        # m-step
        new_m = {}
        for c in comps:
            ga = g[c.attr]
            counts = np.array([p[ga == lv].sum() for lv in range(c.n_levels)])
            pm = (counts + pseudo) / (counts[1:].sum() + pseudo * (c.n_levels - 1))
            pm[0] = 1.0
            new_m[c.attr] = pm
        new_lam = float(np.clip(p.mean(), 1e-6, 1 - 1e-6))
        delta = max([abs(new_lam - lam)] + [np.abs(new_m[a] - m[a]).max() for a in m])
        m, lam = new_m, new_lam
        hist.append(lam)
        if delta < tol:
            break
    return EMResult(m=m, lam=lam, n_iter=it + 1, history=hist)


# --- scoring --------------------------------------------------------------


def weight_table(m: dict, u: dict, comparisons=COMPARISONS) -> pd.DataFrame:
    """
    Tabulate m, u and log2 Bayes-factor weights per attribute level.

    :param m: match level distributions
    :param u: non-match level distributions
    :param comparisons: comparison definitions
    :return: tidy frame
    """
    rows = []
    for c in comparisons:
        for lv, lab in enumerate(c.labels):
            w = 0.0 if lv == 0 else np.log2(m[c.attr][lv] / u[c.attr][lv])
            rows.append({"attr": c.attr, "level": lv, "label": lab,
                         "m": np.nan if lv == 0 else m[c.attr][lv],
                         "u": np.nan if lv == 0 else u[c.attr][lv], "weight": w})
    return pd.DataFrame(rows)


def level_weights(m: dict, u: dict, monotone: bool = True, comparisons=COMPARISONS) -> dict:
    """
    Per-level log2 weights, optionally forced to be non-decreasing in
    agreement strength.

    The monotone cap exists because u for fuzzy levels is often estimated from
    clean data (e.g. entity-vs-entity pairs with no typos), which makes it too
    small and can give a fuzzy match more weight than an exact one.

    :param m: match level distributions
    :param u: non-match level distributions
    :param monotone: cap each agreement level at the weight of the level above it
    :param comparisons: comparison definitions
    :return: ``{attr: weight array}``, null level = 0
    """
    out = {}
    for c in comparisons:
        w = np.log2(m[c.attr] / u[c.attr])
        w[0] = 0.0
        if monotone and c.n_levels > 3:
            agree = w[2:]
            w[2:] = np.minimum.accumulate(agree[::-1])[::-1]
        out[c.attr] = w
    return out


def score(levels: pd.DataFrame, m: dict, u: dict, n_entities: int | None = None,
          use_tf: bool = True, monotone: bool = True, comparisons=COMPARISONS) -> pd.DataFrame:
    """
    Sum per-attribute log2 weights.

    With ``use_tf``, every agreement level's u is replaced by a value-specific
    u built from the reference value the pair agreed on:

    ``u_v(k) = max((count_v - 1) / N, u_k)``

    ``count_v - 1`` is the number of *other* entities holding the value if the
    query's own entity is one of the holders, so a value held by 240 entities
    is nearly worthless and one held by 2 leaves a two-way tie. A unique
    value gives 0, so it falls back to ``u_k``, the rate at which two distinct
    entities coincide at that level. Using ``count_v / N`` (the textbook tf)
    would put a floor of 1/N on every agreement, which is far too high for
    attributes such as name where distinct entities almost never coincide.

    The monotone cap of :func:`level_weights` then has to be applied per pair
    rather than per attribute, because the substitution can reorder the
    levels. Agreeing exactly on a value is at least as strong as agreeing
    approximately on it, so a pair at level ``l`` on value ``v`` is capped at
    what exact agreement on that same ``v`` would earn:

    ``w(l, v) = min over k >= l of log2(m_k / u_v(k))``

    Without this, a near miss on an address held by 240 entities outscores an
    exact hit on it, because the near miss keeps the undiscounted fuzzy u.

    :param levels: comparison vectors (with ``tf_<attr>`` columns)
    :param m: match level distributions
    :param u: non-match level distributions
    :param n_entities: number of entities (required when ``use_tf``)
    :param use_tf: apply term-frequency adjustment
    :param monotone: see :func:`level_weights`
    :param comparisons: comparison definitions
    :return: ``levels`` plus ``w_<attr>`` columns and ``score``
    """
    if use_tf:
        if n_entities is None:
            raise ValueError("n_entities is required when use_tf is True")
        inv_n = 1.0 / n_entities
    out = levels.copy()
    total = np.zeros(len(out))
    lw = level_weights(m, u, monotone, comparisons)
    for c in comparisons:
        g = out[c.attr].values.astype(int)
        w = lw[c.attr][g]
        if use_tf and c.tf:
            tfv = out[f"tf_{c.attr}"].values
            mask = (g >= 2) & ~np.isnan(tfv)
            if mask.any():
                lv = g[mask]
                # (count_v - 1) / N, clamped at 0 for a value only this entity holds
                others = np.maximum(tfv[mask] - inv_n, 0.0)
                capped = np.full(lv.shape, np.inf)
                for k in range(2, c.n_levels):
                    w_k = np.log2(m[c.attr][k] / np.maximum(others, u[c.attr][k]))
                    capped = np.where(lv <= k, np.minimum(capped, w_k), capped)
                w[mask] = capped
        out[f"w_{c.attr}"] = w
        total += w
    out["score"] = total
    return out


def estimate_prior(scored: pd.DataFrame, n_pairs: int, init: float = 1e-5,
                   max_iter: int = 100) -> float:
    """
    Estimate the global pair prior P(match) by EM over lambda alone, with m
    and u fixed. Pairs outside the candidate set are treated as having
    posterior ~0, so they enter only through ``n_pairs``.

    The prior must be global (over all query x entity pairs), not the share of
    matches among candidates: blocking selects pairs on their comparison
    vector, and the Bayes factor already accounts for that vector.

    :param scored: output of :func:`score` for all candidate pairs
    :param n_pairs: n_queries * n_entities
    :param init: starting prior
    :param max_iter: iteration cap
    :return: prior P(match) per pair
    """
    pi = init
    for _ in range(max_iter):
        new = float(to_prob(scored.score.values, pi).sum()) / n_pairs
        if abs(new - pi) / pi < 1e-6:
            break
        pi = new
    return pi


def to_prob(score: np.ndarray, prior: float) -> np.ndarray:
    """
    Convert a log2 Bayes factor to a posterior match probability.

    :param score: summed log2 weights
    :param prior: prior P(match) for a pair
    :return: posterior probabilities
    """
    logit2 = score + np.log2(prior / (1 - prior))
    return 1 / (1 + np.power(2.0, -logit2))


def background_bf(q: pd.DataFrame, e: pd.DataFrame, m: dict, u: dict,
                  comparisons=COMPARISONS) -> pd.Series:
    """
    Approximate Bayes factor for a query against an entity that is *not* a
    candidate, i.e. one that shares no blocking value with it.

    For each attribute the query has, the entity either lacks it (factor 1)
    or disagrees (factor m_disagree / u_disagree). Weighting by how often
    entities carry the attribute and assuming independence:

    ``B0(q) = prod_a [s_a * m_a1 / u_a1 + (1 - s_a)]``

    :param q: normalized query table
    :param e: normalized entity table
    :param m: match level distributions
    :param u: non-match level distributions
    :param comparisons: comparison definitions
    :return: series indexed by query id
    """
    n = e.id.nunique()
    share = e.groupby("attr_type").id.nunique() / n
    ratio = {c.attr: m[c.attr][1] / u[c.attr][1] for c in comparisons}
    qa = q[["id", "attr_type"]].drop_duplicates()
    f = [np.log(share.get(a, 0) * ratio[a] + 1 - share.get(a, 0)) for a in qa.attr_type]
    return np.exp(qa.assign(f=f).groupby("id").f.sum())


def query_posterior(scored: pd.DataFrame, q: pd.DataFrame, e: pd.DataFrame, m: dict, u: dict,
                    rho: float | None = None, max_iter: int = 500) -> tuple[pd.DataFrame, float]:
    """
    Posterior that each candidate is *the* match for its query, assuming each
    query matches at most one entity.

    ``P(j | q) = (rho / N) BF_j / ((rho / N) sum_k BF_k + 1 - rho)``

    ``rho`` is the share of queries whose entity is in the table. The sum runs
    over all N entities: candidates contribute their scored BF, and the
    ``N - n_candidates`` others contribute :func:`background_bf` each.

    Unlike a pairwise posterior with prior 1/N, this uses the fact that the
    other entities mostly *disagree* with the query, and it splits
    probability between competing candidates, so a tie shows up as two
    candidates near 0.5 rather than two near 1.

    :param scored: output of :func:`score`
    :param q: normalized query table
    :param e: normalized entity table
    :param m: match level distributions
    :param u: non-match level distributions
    :param rho: share of queries with a match; estimated by fixed-point EM when None
    :param max_iter: iteration cap for estimating ``rho``
    :return: (``scored`` plus ``bf``, ``total_bf``, ``p``; ``rho``)
    """
    n = e.id.nunique()
    b0 = background_bf(q, e, m, u)
    out = scored.copy()
    out["bf"] = np.power(2.0, out.score)
    agg = out.groupby("qid").agg(sbf=("bf", "sum"), nc=("bf", "size"))
    # queries with no candidates still count toward rho
    total = (n * b0).copy()
    total.loc[agg.index] = agg.sbf + (n - agg.nc) * b0.reindex(agg.index)
    if rho is None:
        rho = 0.5
        for _ in range(max_iter):
            pm = (rho / n * total) / (rho / n * total + 1 - rho)
            new = float(pm.mean())
            if abs(new - rho) < 1e-9:
                break
            rho = new
    out["total_bf"] = total.reindex(out.qid).values
    out["p"] = (rho / n * out.bf) / (rho / n * out.total_bf + 1 - rho)
    return out, rho


def decide(posterior: pd.DataFrame, accept_p: float = 0.95, review_p: float = 0.5) -> pd.DataFrame:
    """
    Pick the most probable candidate per query and assign accept / review / no_match.

    No separate margin rule is needed: :func:`query_posterior` already splits
    probability between competing candidates.

    :param posterior: output of :func:`query_posterior`
    :param accept_p: posterior needed to auto-accept
    :param review_p: posterior needed to send to review instead of rejecting
    :return: one row per query with a candidate: ``qid``, ``eid``, ``score``, ``p``,
        ``p_second``, ``decision``
    """
    # eid breaks exact ties so the pick is reproducible; sort_values is not stable
    s = posterior.sort_values(["qid", "p", "eid"], ascending=[True, False, True])
    s["rank"] = s.groupby("qid").cumcount()
    top = s[s["rank"] == 0].set_index("qid")
    top["p_second"] = s[s["rank"] == 1].set_index("qid").p.reindex(top.index).fillna(0.0)
    top["decision"] = np.select([top.p >= accept_p, top.p >= review_p], ["accept", "review"], "no_match")
    return top.reset_index()
