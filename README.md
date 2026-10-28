# Entity matching

A Fellegi–Sunter probabilistic record linkage model for matching query records against a
reference table of known entities, where the reference data has three awkward properties:

- **Shared values** — distinct entities hold the same address, phone or name (co-tenants,
  registered-agent offices, answering services, franchises). Agreement is weak evidence, and
  how weak depends on the value, not just the attribute.
- **Missing values** — the table is a partial snapshot, so a missing attribute must count as
  neither agreement nor disagreement.
- **Stale values** — an old phone or a pre-move address means disagreement on a volatile
  attribute is weak evidence against a match, while disagreement on a tax ID is strong.

All model parameters are estimated **without labelled data**. The output is a ranked list of
candidate entities per query, each with a probability that is normalized across the whole
reference table, so a genuine tie shows up as two candidates near 0.5 rather than two near 1.

The method is written up in [`docs/methods.md`](docs/methods.md), including notation, the
estimators, a worked numeric example and the assumptions it rests on.

## Layout

| Path | What it is |
|---|---|
| `matching.py` | The library: normalization, blocking, comparison vectors, u estimation, EM for m, term-frequency weights, query-level posterior, decisions |
| `generate_data.py` | Builds the synthetic dataset, with ground truth and realistic drift |
| `walkthrough.ipynb` | Step-by-step estimation without labels, graded against the ground truth at every step |
| `walkthrough.py` | The same walkthrough as a plain script |
| `docs/methods.md` | The method write-up |
| `tests/` | Unit tests, including the paper's worked example |

## Setup

Python 3.12. With pyenv:

```sh
pyenv virtualenv 3.12.11 entity-match
pyenv local entity-match
pip install -r requirements-dev.txt   # or requirements.txt to just run things
```

Or with plain venv:

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Generate the data

The CSVs are not committed; they are reproducible from the default seed in about 7 seconds.

```sh
python generate_data.py --out data
```

This writes:

- `data/entities.csv` — 50,000 known entities in long format `entity_id, attr_type, value`,
  1–10 rows each
- `data/queries.csv` — 20,000 queries in the same format; 15% refer to no entity in the table
- `data/truth.csv` — the true `entity_id` per query (blank = unknown) plus the flags
  `moved`, `phone_changed`, `rebranded`, `contaminated`

Flags: `--entities`, `--queries`, `--p-unknown`, `--seed`.

## Run the walkthrough

```sh
jupyter lab walkthrough.ipynb     # or: python walkthrough.py
```

It takes about 30 seconds. Each step estimates a parameter without labels, then uses the
ground truth to show how far off it is — and how much further off the common default would be:

| Step | Estimates | Without labels via | Default that fails |
|---|---|---|---|
| 4 | u: P(level \| non-match) | entity-vs-entity pairs | random query-vs-entity pairs |
| 5 | m: P(level \| match) | EM per blocking round, u held fixed | one EM over all candidates |
| 6 | value-specific u | (count − 1) / N with a floor | count / N |
| 8 | match probability | query-level posterior | pairwise posterior |

On the synthetic data it ends at 97.8% blocking recall, a 72% auto-accept rate at 99.6%
precision, and 5.8% sent to review.

## Using the library

```python
import pandas as pd
import matching as mt

E = pd.read_csv("data/entities.csv")
Q = pd.read_csv("data/queries.csv")

e, q = mt.normalize(E, "entity_id"), mt.normalize(Q, "query_id")
tf = mt.value_tf(e)

pairs = pd.concat([
    mt.block_exact(q, e, ["name", "address", "phone", "email", "website", "tax_id"]),
    mt.block_name_tokens(q, e, max_df=30),
]).drop_duplicates()

levels = mt.compare(pairs, q, e, tf)
# u from distinct-entity pairs, m from EM per blocking round — see the walkthrough
scored = mt.score(levels, m, u, n_entities=e.id.nunique())
posterior, rho = mt.query_posterior(scored, q, e, m, u)
decisions = mt.decide(posterior, accept_p=0.95, review_p=0.5)
```

## Development

```sh
python generate_data.py --out data   # tests/test_pipeline_on_data.py skips without it
pytest
ruff check .
mypy matching.py generate_data.py walkthrough.py
```

`walkthrough.py` is generated from `walkthrough.ipynb`; edit the notebook.

The tests check the implementation against `docs/methods.md` rather than just
against itself:

- `tests/test_methods_paper.py` runs the worked example from section 9 through
  the code and asserts every number the paper quotes — the weight table, the
  comparison levels, the four scores, the background factor, the posteriors,
  and both decisions.
- `tests/test_pipeline_on_data.py` recomputes the u estimator, the EM fixed
  point, the background factor, rho and `P(e | q)` directly from the formulas
  in sections 5 to 8 and compares them to what the pipeline produced.
