# Probabilistic Entity Matching with Shared, Missing, and Drifting Attributes

## 1. Problem

A reference table describes N known entities. Each entity has between 1 and 10 attributes (name, address, phone, email, website, tax identifier, and so on). The table is in long format, with one row per entity–attribute–value. A query record is described the same way. The goal is to score the known entities that the query might refer to, so that the likely matches can be ranked and inspected. Reducing that list to a single decision is an optional final step (Section 8).

Three properties of the reference data shape the method:

- **Shared values.** Distinct entities can hold the same value. Examples are co-tenants of a building, firms using one registered-agent address or answering service, subsidiaries sharing a switchboard, and franchises sharing a name or website. Agreement on a shared value is weak evidence. How weak depends on the specific value, not only on the attribute type.
- **Missing values.** The table is a partial snapshot. A query may carry an attribute that the entity has no record of. That absence carries no information and must not be counted as disagreement.
- **Stale values.** The table may hold an outdated value, such as an old phone number or a pre-move address. Disagreement on a volatile attribute is therefore weak evidence against a match. Disagreement on a stable identifier is strong evidence.

The method is a Fellegi–Sunter likelihood-ratio model. Its parameters are estimated without labelled data, and it adds a probability that is normalized per query.

## 2. Notation

| Symbol | Meaning |
|:---|:---|
| $`\mathcal{E}`$ | Set of known entities, $`N`$ in total (read “script E”) |
| $`q`$, $`e`$ | A query; an entity |
| $`A_{q}`$ | Attribute types present in query $`q`$ |
| $`V_{e}(a)`$ | Values entity $`e`$ holds for attribute type $`a`$ (possibly several) |
| $`\gamma_{a}(q,e)`$ | Comparison level for attribute $`a`$ (read “gamma”) |
| $`\ell`$ | A particular comparison level (read “ell”) |
| $`L_{a}`$ | Number of levels for attribute $`a`$, including null |
| $`M`$, $`U`$ | Match and non-match classes |
| $`m_{a}(\ell)`$ | Probability of level $`\ell`$ given a match |
| $`u_{a}(\ell)`$ | Probability of level $`\ell`$ given a non-match |
| $`w_{a}(\ell)`$ | Weight of level $`\ell`$, in bits |
| $`S(q,e)`$ | Total score of a pair, in bits |
| $`c_{v}`$ | Number of entities that hold value $`v`$ |
| $`\lambda`$ | Share of matches among the pairs in one EM run (read “lambda”) |
| $`\rho`$ | Share of queries whose entity is in the table (read “rho”) |
| $`C_{q}`$ | Candidate entities for query $`q`$ |
| $`B_{0}(q)`$ | Background Bayes factor for a non-candidate entity (read “B-zero”) |
| $`s_{a}`$ | Share of entities that hold attribute $`a`$ |

## 3. Comparison vectors

### 3.1 Normalization

Values are normalized per attribute type before comparison:

- **Names:** lower-cased, “&” mapped to “and”, punctuation removed, legal suffixes (Inc, LLC, Co, Corp, Ltd, and their variants) dropped.
- **Addresses:** suite, unit and apartment designators removed, street types reduced to a standard abbreviation, punctuation removed.
- **Phone numbers:** reduced to their final ten digits.
- **Websites:** scheme and “www.” removed.
- **Identifiers:** reduced to digits.

Removing suite numbers is deliberate. All tenants of a building then share one normalized address, and the value-specific weighting of Section 6.3 discounts agreement on it.

### 3.2 Comparison levels

For a pair $`(q,e)`$ and each attribute type $`a`$, the comparison $`\gamma_{a}(q,e)`$ takes one of $`L_{a}`$ ordered levels:

- **0, null:** $`a \notin A_{q}`$ or $`V_{e}(a) = \varnothing`$.
- **1, disagree:** both sides have values, and no pair of values reaches an agreement level.
- **2 to** $`L_{a} - 1`$**, increasing agreement.** For text attributes these are similarity bands, such as Jaro–Winkler for names and normalized edit distance for addresses. The top level is exact equality after normalization.

For a company table, a typical set of levels is:

| Attribute | $`\ell = 0`$ | $`\ell = 1`$ | $`\ell = 2`$ | $`\ell = 3`$ | $`\ell = 4`$ |
|:---|:---|:---|:---|:---|:---|
| Name ($`L = 5`$) | null | no meaningful similarity | Jaro–Winkler ≥ 0.85 | Jaro–Winkler ≥ 0.94 | exact |
| Address ($`L = 4`$) | null | disagree | edit similarity ≥ 0.90 | exact |  |
| Phone, email, website, tax ID ($`L = 3`$) | null | disagree | exact |  |  |

Each level has its own $`m`$, $`u`$ and weight, so a near-miss name earns partial credit instead of all or nothing.

When either side holds several values for one attribute type, $`\gamma_{a}`$ is the highest level reached by any pair of values:

``` math
\gamma_{a}(q,e) = \max_{x \in V_{q}(a),\ y \in V_{e}(a)}{level}_{a}(x,y)
```

So one shared phone number counts as agreement even when each side also holds a number the other lacks.

## 4. Candidate generation

Scoring every query against every entity is unnecessary. A pair is scored only if it shares at least one of the following:

1.  an exact normalized value on any attribute, or
2.  a name token held by no more than a small number of entities (30 is a reasonable default). This catches legal-suffix changes and single typos, provided one rare token survives intact.

The matches that blocking misses are lost to every later step, so blocking recall caps the recall of the whole method. Any entity that is not a candidate is still represented in the probability calculation, through the background term of Section 7.

## 5. Scoring model

For each attribute $`a`$ and non-null level $`\ell`$, $`m_{a}(\ell)`$ is the probability that a true-match pair lands at level $`\ell`$, and $`u_{a}(\ell)`$ is the probability that a non-match pair does:

``` math
m_{a}(\ell) = P\left( \gamma_{a} = \ell \mid M,\ \gamma_{a} \neq 0 \right),\quad\quad u_{a}(\ell) = P\left( \gamma_{a} = \ell \mid U,\ \gamma_{a} \neq 0 \right)
```

m measures how reliably true matches show the evidence. It is below 1 because of typos, drift and multiple values. u measures how often the evidence appears by coincidence. It is above 0 because of shared values and common names. Conditioning on a non-null level makes the null level contribute nothing, whatever the attribute’s coverage.

The weight of a level is its log-likelihood ratio in bits:

``` math
w_{a}(\ell) = \log_{2}\frac{m_{a}(\ell)}{u_{a}(\ell)},\quad\quad w_{a}(0) = 0
```

The logarithm is taken in base 2, so weights are measured in bits. Each bit is a factor of 2: a weight of +1 bit means the evidence is twice as likely for a match as for a non-match, +10 bits about 1,000 times, and +20 bits about a million times. A negative weight counts against a match in the same way; −2 bits means the evidence is four times less likely for a match. The choice of base does not affect rankings or probabilities, because a different base would rescale every weight by the same constant.

The score of a pair is the sum of its weights:

``` math
S(q,e) = \sum_{a\,:\,\gamma_{a}(q,e) \neq 0}^{}w_{a}\left( \gamma_{a}(q,e) \right)
```

$`S`$ is the log₂ Bayes factor for “match” against “non-match”, under the assumption that attributes are conditionally independent given match status. Agreement levels produce positive weights. Disagreement produces a negative weight, whose size depends on how often true matches disagree on that attribute. Volatile attributes such as phone and email get mild penalties. Stable identifiers such as tax IDs, and names, get stiff ones.

## 6. Parameter estimation without labels

### 6.1 Non-match probabilities, u

The usual estimator samples random query–entity pairs, on the grounds that almost all of them are non-matches. That reasoning fails when the match rate among random pairs is about $`\rho/N`$, which can be as large as $`u`$ itself for discriminating levels such as exact name. The few true matches in the sample then dominate the estimate and inflate $`u`$.

If the reference table has been deduplicated, every pair of distinct entities $`\left( e_{i},e_{j} \right)`$, $`i \neq j`$, is a non-match by construction. $`u`$ is therefore estimated from a large random sample of such pairs, using the level counts $`n_{a}(\ell)`$:

``` math
{\widehat{u}}_{a}(\ell) = \frac{1 + n_{a}(\ell)}{\left( L_{a} - 1 \right) + \sum_{k \geq 1}^{}n_{a}(k)},\quad\quad\ell \geq 1
```

The pseudo-count of 1 per level keeps every weight finite. For levels that distinct entities almost never reach, the estimate sits near this floor, so it reflects sample size more than measurement (Section 10).

Entity values contain none of the noise found in queries, so this estimator understates $`u`$ for the fuzzy agreement levels. Section 6.4 corrects for this.

### 6.2 Match probabilities, m, by expectation–maximization

$`m`$ needs a sample rich in matches, which means candidate pairs. But candidates were chosen because they agree on a blocking attribute. If EM includes that attribute, every candidate looks like a strong match. EM then assigns non-matches (for example, co-tenants of a building) to the match class, and $`m`$ for the other attributes collapses.

EM is therefore run separately on the pairs produced by each single-key blocking rule, with three constraints:

- $`u`$ is held fixed at the values from Section 6.1.
- The blocked attribute is left out of the likelihood. Within that block it agrees for every pair, so it carries no information.
- The rules are chosen so that every attribute is estimated in at least one round, that is, no attribute is the blocking key in every round. Blocking on exact name in one round and exact phone in another is sufficient.

Let $`\mathcal{A}`$ be the attributes in a round and $`\gamma_{ia}`$ the level of pair $`i`$. The E-step computes each pair’s match probability:

``` math
p_{i} = \frac{\lambda\prod_{a \in \mathcal{A}}^{}m_{a}\left( \gamma_{ia} \right)}{\lambda\prod_{a \in \mathcal{A}}^{}m_{a}\left( \gamma_{ia} \right) + (1 - \lambda)\prod_{a \in \mathcal{A}}^{}u_{a}\left( \gamma_{ia} \right)}
```

with $`m_{a}(0) = u_{a}(0) = 1`$. The M-step updates $`m`$ and $`\lambda`$:

``` math
m_{a}(\ell) = \frac{1 + \sum_{i}^{}p_{i}\,\left\lbrack \gamma_{ia} = \ell \right\rbrack}{\left( L_{a} - 1 \right) + \sum_{i}^{}p_{i}\,\left\lbrack \gamma_{ia} \neq 0 \right\rbrack},\quad\quad\lambda = \frac{1}{n}\sum_{i}^{}p_{i}
```

The two steps alternate until $`\lambda`$ and every $`m_{a}(\ell)`$ change by less than a tolerance. An attribute estimated in more than one round takes the mean of its estimates. $`\lambda`$ applies only to its own round and is then discarded.

### 6.3 Value-specific non-match probability

A single $`u`$ per attribute cannot separate agreement on a unique value from agreement on a value held by hundreds of entities. For agreement at the exact level on value $`v`$, the attribute-level $`u`$ is replaced by

``` math
u_{v} = \max\left( \frac{c_{v} - 1}{N},\ u_{a}\left( \text{exact} \right) \right)
```

The usual term-frequency adjustment, $`c_{v}/N`$, counts the query’s own entity among the holders of the value. That puts a floor of $`1/N`$ on every agreement, which for names is far above the true rate at which distinct entities coincide. Using $`c_{v} - 1`$ counts only the other holders. The floor at $`u_{a}\left( \text{exact} \right)`$ stops a unique value from implying certainty. If several values agree, the rarest one is used.

The adjustment is only meaningful for attributes whose values are legitimately shared. The implementation switches it off per attribute (`tf=False` in `COMPARISONS`), and it is off for the tax identifier, whose values are unique in the reference table.

### 6.4 Monotone weights

Agreement weights must not decrease as agreement strengthens. Because $`u`$ for the fuzzy levels is underestimated (Section 6.1), the raw estimates can give fuzzy agreement more weight than exact agreement. Each agreement level is capped at the weight of the levels above it:

``` math
{\widetilde{w}}_{a}(\ell) = \min_{k \geq \ell}w_{a}(k),\quad\quad\ell \geq 2
```

## 7. Probability of each candidate

A pairwise posterior judges each pair in isolation, with prior odds of about $`\rho/N`$. It ignores two things. The other entities mostly disagree with the query, which is itself evidence. And a query refers to at most one entity. The two omissions pull in opposite directions. Ignoring the disagreement of the other entities makes the pairwise posterior underconfident. Ignoring the one-match constraint makes it overconfident whenever two candidates both score well, so two tied candidates can each receive a probability near 1 (Section 9.4).

The method instead normalizes over the whole reference table. Let $`BF_{e} = 2^{S(q,e)}`$. If each query matches at most one entity, the probability that $`e`$ is the match is

``` math
P(e \mid q) = \frac{(\rho/N)\, BF_{e}}{(\rho/N)\left\lbrack \sum_{k \in C_{q}}^{}BF_{k} + \left( N - \left| C_{q} \right| \right)\, B_{0}(q) \right\rbrack + (1 - \rho)}
```

The $`N - \left| C_{q} \right|`$ entities that were never scored each contribute a background factor. It is computed from each attribute’s coverage $`s_{a}`$ and its disagreement ratio, assuming such entities disagree wherever they hold the attribute:

``` math
B_{0}(q) = \prod_{a \in A_{q}}^{}\left\lbrack s_{a}\,\frac{m_{a}(1)}{u_{a}(1)} + \left( 1 - s_{a} \right) \right\rbrack
```

The share of queries that have a match, $`\rho`$, is estimated by fixed-point iteration over all $`n_{Q}`$ queries, including those with no candidates:

``` math
\rho \leftarrow \frac{1}{n_{Q}}\sum_{q}^{}\frac{(\rho/N)\, T_{q}}{(\rho/N)\, T_{q} + (1 - \rho)},\quad\quad T_{q} = \sum_{k \in C_{q}}^{}BF_{k} + \left( N - \left| C_{q} \right| \right)\, B_{0}(q)
```

The estimate is low by roughly the share of matches that blocking misses, because the model cannot see them.

## 8. Output

For each query, the method returns every candidate entity with its score $`S(q,e)`$ and its probability $`P(e \mid q)`$, ranked from most to least likely. All candidates share one denominator, so their probabilities sum to at most 1. The remainder is

``` math
P\left( \text{none listed} \mid q \right) = 1 - \sum_{e \in C_{q}}^{}P(e \mid q)
```

which is the probability that the query’s entity is not in the table or was missed by blocking. A tie between two candidates shows up as two probabilities near 0.5, not two near 1.

When a single decision is needed, take the top candidate and apply two thresholds:

| Decision | Condition on the top candidate                                   |
|:---------|:-----------------------------------------------------------------|
| Accept   | $`P(e \mid q) \geq \tau_{\text{accept}}`$                        |
| Review   | $`\tau_{\text{review}} \leq P(e \mid q) < \tau_{\text{accept}}`$ |
| No match | $`P(e \mid q) < \tau_{\text{review}}`$, or no candidates         |

The thresholds are policy choices, and should be set from a target precision measured on labelled pairs. Because ties already split probability, no separate margin rule between the first and second candidates is needed.

If confident matches are used to add new values to the reference table, the threshold for that should be stricter than the accept threshold, and the source of each added value should be recorded. A false match that writes its values into the table attracts further false matches.

## 9. Worked example

This section works the method through by hand on a few records. The parameters are illustrative. They are rounded to the size that estimation on a table of tens of thousands of companies produces. Throughout, $`N = 50,000`$ entities and $`\rho = 0.85`$.

### 9.1 Estimating u

A sample of 2,000,000 pairs of distinct entities is drawn. Every entity has a name, so all 2,000,000 pairs have a non-null name comparison, and 3 of them agree exactly on the normalized name. Name has $`L = 5`$ levels, so with the pseudo-count:

``` math
{\widehat{u}}_{\text{name}}\left( \text{exact} \right) = \frac{1 + 3}{4 + 2,000,000} \approx 2.0 \times 10^{- 6}
```

Of the same sample, 1,200,000 pairs have a phone number on both sides, and 23 of those share a number:

``` math
{\widehat{u}}_{\text{phone}}\left( \text{agree} \right) = \frac{1 + 23}{2 + 1,200,000} \approx 2.0 \times 10^{- 5}
```

### 9.2 Estimating m by EM

Take the 12 candidate pairs that were blocked on exact name. Name is left out, and the pairs are compared on address and phone. For brevity, each comparison is either agree or disagree, with no nulls. $`u`$ is held fixed at $`u_{\text{address}}\left( \text{agree} \right) = u_{\text{phone}}\left( \text{agree} \right) = 0.00002`$. The starting values are $`m_{\text{address}}\left( \text{agree} \right) = m_{\text{phone}}\left( \text{agree} \right) = 0.9`$ and $`\lambda = 0.5`$. The pseudo-counts are omitted here: with 12 pairs they would dominate the estimate, while at realistic sample sizes they are negligible.

| Address | Phone | Pairs | $`p_{i}`$ after first E-step | $`p_{i}`$ at convergence |
|:---|:---|:---|:---|:---|
| agree | agree | 6 | 1.0000 | 1.0000 |
| agree | disagree | 2 | 0.9998 | 1.0000 |
| disagree | agree | 1 | 0.9998 | 0.9999 |
| disagree | disagree | 3 | 0.0099 | 0.1111 |

For an agree/disagree pair, the first E-step gives

``` math
p_{i} = \frac{0.5 \times 0.9 \times 0.1}{0.5 \times 0.9 \times 0.1 + 0.5 \times 0.00002 \times 0.99998} = \frac{0.045}{0.04501} = 0.9998
```

The first M-step weights each pair by its $`p_{i}`$. The total weight is $`6 + 2(0.9998) + 0.9998 + 3(0.0099) = 9.029`$, so

``` math
m_{\text{address}}\left( \text{agree} \right) = \frac{6 + 2(0.9998)}{9.029} = 0.886,\quad\quad m_{\text{phone}}\left( \text{agree} \right) = \frac{6 + 0.9998}{9.029} = 0.775,\quad\quad\lambda = \frac{9.029}{12} = 0.752
```

After about 20 iterations, EM converges to $`m_{\text{address}}\left( \text{agree} \right) = 0.857`$, $`m_{\text{phone}}\left( \text{agree} \right) = 0.750`$ and $`\lambda = 0.778`$. The three pairs that disagree on both attributes are mostly treated as different companies that share a name. They keep a match probability of 0.11 because matches are common in a name-matched block. This $`\lambda`$ belongs to this block only and is not reused.

### 9.3 Weights

The weights below come from full-scale estimates of the kind shown in Sections 9.1 and 9.2.

| Attribute and level             | $`m`$ | $`u`$    | Weight (bits)            |
|:--------------------------------|:------|:---------|:-------------------------|
| Name: no similarity             | 0.02  | 0.9994   | −5.64                    |
| Name: Jaro–Winkler ≥ 0.85       | 0.01  | 0.0005   | +4.32                    |
| Name: Jaro–Winkler ≥ 0.94       | 0.07  | 0.00002  | +11.77                   |
| Name: exact                     | 0.90  | 0.000002 | +18.78                   |
| Address: disagree               | 0.08  | 0.99997  | −3.64                    |
| Address: edit similarity ≥ 0.90 | 0.05  | 0.000001 | +15.61, capped to +15.41 |
| Address: exact                  | 0.87  | 0.00002  | +15.41                   |
| Phone: disagree                 | 0.27  | 0.99998  | −1.89                    |
| Phone: agree                    | 0.73  | 0.00002  | +15.16                   |

The fuzzy address level shows the monotone cap of Section 6.4. Its raw weight is above the exact level’s, so it is capped there. The disagreement penalties also differ as Section 5 describes: a different name costs 5.6 bits, while a different phone costs only 1.9.

### 9.4 Scoring one query

Four entities in the reference table are relevant:

| Entity | Name             | Address                    | Phone        |
|:-------|:-----------------|:---------------------------|:-------------|
| E1     | Acme Tools Inc   | 12 Main St, Denver         | 303-555-0100 |
| E2     | Acme Tooling LLC | (none)                     | 303-555-0100 |
| E3     | Bolt Supply Co   | 12 Main St Ste 200, Denver | 303-555-0199 |
| E4     | Cobalt Dental    | 12 Main St Ste 310, Denver | (none)       |

The query is: ACME TOOLS, 12 Main Street, Denver, (303) 555-0199.

**Candidates.** After normalization:

- E1, E3 and E4 share the query’s address. Suite numbers are removed, so all three tenants have the same normalized address.
- E3 shares the query’s phone number.
- E1 and E2 share the rare name token “acme”.

The candidate set is therefore $`C_{q} = \{ E1,E2,E3,E4\}`$.

**Value-specific u.** The address is held by $`c_{v} = 3`$ entities, so

``` math
u_{v} = \max\left( \frac{3 - 1}{50,000},\ 0.00002 \right) = 0.00004,\quad\quad w = \log_{2}\frac{0.87}{0.00004} = 14.41
```

This is one bit less than an address held by a single entity would earn. The phone 303-555-0199 and the name “acme tools” are each held by one entity, so they keep their attribute-level weights.

**Scores.** “acme tools” against “acme tooling” has a Jaro–Winkler similarity of 0.93, which is level 2.

| Candidate | Name | Address | Phone | $`S`$ (bits) | $`BF = 2^{S}`$ |
|:---|:---|:---|:---|:---|:---|
| E1 | exact, +18.78 | exact, +14.41 | disagree, −1.89 | 31.30 | $`2.6 \times 10^{9}`$ |
| E2 | ≥ 0.85, +4.32 | null, 0 | disagree, −1.89 | 2.43 | 5.4 |
| E3 | none, −5.64 | exact, +14.41 | agree, +15.16 | 23.92 | $`1.6 \times 10^{7}`$ |
| E4 | none, −5.64 | exact, +14.41 | null, 0 | 8.77 | 435 |

**Background term.** Entity coverage is $`s_{\text{name}} = 1.0`$, $`s_{\text{address}} = 0.90`$ and $`s_{\text{phone}} = 0.77`$. Each attribute in the query contributes one factor:

- Name: $`1.0 \times 0.02/0.9994 + 0 = 0.0200`$
- Address: $`0.90 \times 0.08/0.99997 + 0.10 = 0.172`$
- Phone: $`0.77 \times 0.27/0.99998 + 0.23 = 0.438`$

``` math
B_{0}(q) = 0.0200 \times 0.172 \times 0.438 = 0.0015
```

**Probabilities.** $`T_{q} = 2.64 \times 10^{9} + 5.4 + 1.59 \times 10^{7} + 435 + 49,996 \times 0.0015 = 2.66 \times 10^{9}`$, and $`\rho/N = 1.7 \times 10^{- 5}`$, so the denominator is $`1.7 \times 10^{- 5} \times 2.66 \times 10^{9} + 0.15 = 45,196`$.

| Rank | Candidate   | $`P(e \mid q)`$ |
|:-----|:------------|:----------------|
| 1    | E1          | 0.994           |
| 2    | E3          | 0.006           |
| 3    | E4          | \< 0.0001       |
| 4    | E2          | \< 0.0001       |
|      | none listed | 0.000003        |

E1 is accepted at $`\tau_{\text{accept}} = 0.95`$. The evidence conflicts: the name points to E1 and the phone points to E3. But an exact match on a unique name (18.8 bits) outweighs a phone match (15.2 bits), and E3 also pays 5.6 bits for its different name.

A pairwise posterior with prior $`\rho/N`$ would give E1 0.99998 and E3 0.996. Both would look certain. Normalizing per query gives E3 only the 0.6% it deserves.

### 9.5 A tie

A second query gives only the address, 12 Main St, Denver. E1, E3 and E4 each score 14.41 bits ($`BF = 21,770`$). With only the address present, $`B_{0}(q) = 0.172`$. Then $`T_{q} = 3 \times 21,770 + 49,997 \times 0.172 = 73,910`$, and the denominator is $`1.7 \times 10^{- 5} \times 73,910 + 0.15 = 1.406`$, so

``` math
P(E1 \mid q) = P(E3 \mid q) = P(E4 \mid q) = \frac{1.7 \times 10^{- 5} \times 21,770}{1.406} = 0.263
```

That leaves 0.211 for “none listed”. About 0.107 of that is the chance the company is not in the table at all. The remaining 0.104 is the chance that it is some other entity at the address that blocking did not reach. The ranked list shows a three-way tie. The top probability is below $`\tau_{\text{review}} = 0.5`$, so the decision is no match: one shared building address cannot identify a tenant.

## 10. Assumptions and limitations

- **Conditional independence.** Summing weights assumes attributes are independent given match status. Correlated drift breaks this: a company that moves usually changes its address and phone together. Disagreement on both is then double-counted, and the m estimated in a phone-blocked round overstates address agreement. Correlated attributes can be combined into a single comparison, or the final probabilities recalibrated (below).
- **Deduplicated reference table.** The estimator in Section 6.1 is clean only if no two entities in the table are the same real-world entity. Undetected duplicates leak matches into $`u`$.
- **Rare levels are bounded by sample size.** When distinct entities almost never agree on an attribute, $`u`$ is fixed by the smoothing floor, and the top weights grow by about one bit each time the sample is doubled. A candidate supported by one such attribute alone is likely to be overconfident.
- **EM reports the data it is given.** If an attribute never changes in the data, EM learns a very large disagreement penalty for it, even if that attribute does change in practice. Estimated weights should be checked against domain knowledge.
- **Approximate background term.** $`B_{0}`$ assumes non-candidates disagree on every attribute they hold, independently across attributes, and it ignores near-miss names that fell outside blocking.
- **At most one match per query.** Where the table legitimately holds several records for one real-world entity, the probability is split between them. The method does not enforce a one-to-one assignment across queries.
- **Calibration.** Mid-range probabilities tend to be underconfident when the independence assumption is violated. A few hundred labelled pairs are enough to audit $`m`$, measure calibration, and fit a monotone (isotonic) mapping from $`P(e \mid q)`$ to an observed match rate. This is the cheapest correction.
