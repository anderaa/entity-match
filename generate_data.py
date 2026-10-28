"""
Generate a synthetic company entity-matching dataset with ground truth.

Outputs (in ``--out``):

- ``entities.csv``: known-entity table, long format (``entity_id, attr_type, value``).
- ``queries.csv``: example entities to match, long format (``query_id, attr_type, value``).
- ``truth.csv``: one row per query with the true ``entity_id`` (empty when the
  query's entity is not in the table) plus diagnostic flags.

Each entity has a full "true" profile. The known-entity table is a partial,
stale snapshot of it. Queries are drawn from the current profile after drift
(moves, number changes, rebrands) and formatting/typo noise. Structure that
makes the problem hard is built in deliberately:

- shared values: office buildings, registered-agent addresses, answering-service
  phones, parent/subsidiary phones, franchise names and websites
- correlated drift: a move changes address and usually phones together
- unknown entities, some of which are hard negatives (same building, similar name)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYLLABLES = (
    "ac me bo lt ka ri ven to sar mo len dra pex qui lor tan vel zor ham bri "
    "cor del fen gar hol jen kel mar nor pal ros sel tor ul vin wex yar zel "
    "ash bel cal dun eld fir gra hel ist jor kra lum mer ost pra rid sto tre"
).split()
INDUSTRIES = [
    "Tools", "Supply", "Logistics", "Dental", "Bakery", "Consulting", "Electric",
    "Plumbing", "Roofing", "Software", "Analytics", "Foods", "Motors", "Realty",
    "Insurance", "Partners", "Labs", "Design", "Media", "Construction",
    "Landscaping", "Cleaning", "Fitness", "Pharmacy", "Imports", "Textiles",
    "Brewing", "Coffee", "Auto Repair", "Legal", "Accounting", "Marketing",
    "Printing", "Security", "Solar", "Heating & Air", "Medical", "Veterinary",
    "Staffing", "Transport",
]
SUFFIXES = ["Inc", "LLC", "Co", "Corp", "Ltd", ""]
SUFFIX_P = [0.3, 0.35, 0.1, 0.1, 0.05, 0.1]
STREET_TYPES = [("St", "Street"), ("Ave", "Avenue"), ("Blvd", "Boulevard"),
                ("Rd", "Road"), ("Dr", "Drive"), ("Ln", "Lane"), ("Way", "Way")]
CITIES = [
    ("Denver", "CO", "802"), ("Boulder", "CO", "803"), ("Fort Collins", "CO", "805"),
    ("Austin", "TX", "787"), ("Dallas", "TX", "752"), ("Houston", "TX", "770"),
    ("Phoenix", "AZ", "850"), ("Tucson", "AZ", "857"), ("Seattle", "WA", "981"),
    ("Spokane", "WA", "992"), ("Portland", "OR", "972"), ("Boise", "ID", "837"),
    ("Salt Lake City", "UT", "841"), ("Omaha", "NE", "681"), ("Chicago", "IL", "606"),
    ("Columbus", "OH", "432"), ("Atlanta", "GA", "303"), ("Miami", "FL", "331"),
    ("Tampa", "FL", "336"), ("Nashville", "TN", "372"), ("Raleigh", "NC", "276"),
    ("Richmond", "VA", "232"), ("Boston", "MA", "021"), ("Albany", "NY", "122"),
    ("Pittsburgh", "PA", "152"), ("Detroit", "MI", "482"), ("Madison", "WI", "537"),
    ("Kansas City", "MO", "641"), ("Tulsa", "OK", "741"), ("Reno", "NV", "895"),
]
AREA_CODES = [303, 720, 970, 512, 214, 713, 602, 520, 206, 509, 503, 208, 801,
              402, 312, 614, 404, 305, 813, 615, 919, 804, 617, 518, 412, 313,
              608, 816, 918, 775]
FIRST_NAMES = ["info", "sales", "contact", "office", "admin", "hello", "billing",
               "jsmith", "mlee", "agarcia", "kpatel", "dnguyen", "rjones"]


class Generator:
    """
    Stateful generator so that phones, tax ids, and tokens stay unique where
    they should be.

    :param seed: random seed
    """

    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.used_phones: set[str] = set()
        self.used_tax: set[str] = set()

    # --- primitive value generators -------------------------------------
    def token(self) -> str:
        """
        Make a pseudo-word from 2-3 syllables.

        :return: capitalized token
        """
        n = self.rng.choice([2, 3], p=[0.6, 0.4])
        return "".join(self.rng.choice(SYLLABLES, n)).capitalize()

    def name(self) -> str:
        """
        Make a company name.

        :return: company name
        """
        r = self.rng.random()
        ind = self.rng.choice(INDUSTRIES)
        suf = self.rng.choice(SUFFIXES, p=SUFFIX_P)
        if r < 0.6:
            base = f"{self.token()} {ind}"
        elif r < 0.75:
            base = f"{self.token()} & {self.token()} {ind}"
        else:
            base = f"{self.token()} {self.token()} {ind}"
        return f"{base} {suf}".strip()

    def address(self, city=None) -> str:
        """
        Make a street address in canonical (abbreviated) form.

        :param city: optional (city, state, zip3) tuple
        :return: address string
        """
        c = city if city is not None else CITIES[self.rng.integers(len(CITIES))]
        st = STREET_TYPES[self.rng.integers(len(STREET_TYPES))][0]
        num = self.rng.integers(1, 9999)
        z = f"{c[2]}{self.rng.integers(0, 99):02d}"
        return f"{num} {self.token()} {st}, {c[0]}, {c[1]} {z}"

    def phone(self) -> str:
        """
        Make a unique phone number in canonical ``AAA-EEE-NNNN`` form.

        :return: phone number
        """
        while True:
            p = (f"{self.rng.choice(AREA_CODES)}-{self.rng.integers(200, 999)}-"
                 f"{self.rng.integers(0, 9999):04d}")
            if p not in self.used_phones:
                self.used_phones.add(p)
                return p

    def tax_id(self) -> str:
        """
        Make a unique EIN-style tax id.

        :return: tax id ``NN-NNNNNNN``
        """
        while True:
            t = f"{self.rng.integers(10, 99)}-{self.rng.integers(0, 9999999):07d}"
            if t not in self.used_tax:
                self.used_tax.add(t)
                return t

    def domain_for(self, name: str) -> str:
        """
        Derive a website domain from a company name.

        :param name: company name
        :return: domain such as ``acmetools.com``
        """
        words = [w for w in name.lower().replace("&", " ").split()
                 if w not in {"inc", "llc", "co", "corp", "ltd"}]
        tld = str(self.rng.choice(["com", "net", "co", "biz"], p=[0.7, 0.12, 0.1, 0.08]))
        return "".join(words[:2]) + "." + tld

    def emails_for(self, domain) -> list[str]:
        """
        Make 0-2 email addresses, some on the company domain, some on gmail.

        :param domain: company domain or None
        :return: list of emails
        """
        out = []
        for _ in range(self.rng.choice([0, 1, 2], p=[0.25, 0.55, 0.2])):
            if domain is not None and self.rng.random() < 0.75:
                out.append(f"{self.rng.choice(FIRST_NAMES)}@{domain}")
            else:
                out.append(f"{self.token().lower()}{self.rng.integers(1, 999)}@gmail.com")
        return sorted(set(out))

    def profile(self) -> dict:
        """
        Make a full true profile for a new, independent entity.

        :return: dict with name, address, phones, emails, website, tax_id
        """
        name = self.name()
        website = self.domain_for(name) if self.rng.random() < 0.7 else None
        return {
            "name": name,
            "address": self.address(),
            "phones": [self.phone() for _ in range(self.rng.choice([1, 2, 3], p=[0.55, 0.35, 0.1]))],
            "emails": self.emails_for(website),
            "website": website,
            "tax_id": self.tax_id(),
        }


def build_entities(g: Generator, n: int) -> list[dict]:
    """
    Build true profiles for ``n`` known entities with shared-value structure.

    :param g: generator
    :param n: number of entities
    :return: list of profile dicts (index = entity number)
    """
    rng = g.rng
    profiles = [g.profile() for _ in range(n)]

    # office buildings: ~12% of entities share one of n/60 building addresses
    buildings = [g.address() for _ in range(max(1, n // 60))]
    for i in rng.choice(n, int(0.12 * n), replace=False):
        b = buildings[rng.integers(len(buildings))]
        # half carry a suite number, which normalization should strip
        profiles[i]["address"] = b if rng.random() < 0.5 else b.replace(",", f" Ste {rng.integers(100, 999)},", 1)

    # registered-agent hubs: 3 addresses each used by ~0.5% of entities
    hubs = [g.address() for _ in range(3)]
    for i in rng.choice(n, int(0.015 * n), replace=False):
        profiles[i]["address"] = hubs[rng.integers(3)]

    # answering-service phones: 2 numbers shared by ~0.3% of entities each
    services = [g.phone() for _ in range(2)]
    for i in rng.choice(n, int(0.006 * n), replace=False):
        profiles[i]["phones"][0] = services[rng.integers(2)]

    # parent/subsidiary: ~4% share a phone with another entity
    for i in rng.choice(n, int(0.04 * n), replace=False):
        j = rng.integers(n)
        if j != i:
            profiles[i]["phones"].append(profiles[j]["phones"][0])

    # franchises: ~5% copy another entity's name (exact or "of <city>") and website
    for i in rng.choice(n, int(0.05 * n), replace=False):
        j = rng.integers(n)
        if j == i:
            continue
        src = profiles[j]
        city = profiles[i]["address"].split(", ")[1]
        profiles[i]["name"] = src["name"] if rng.random() < 0.5 else f"{src['name']} of {city}"
        profiles[i]["website"] = src["website"]
    return profiles


def snapshot(g: Generator, p: dict) -> list[tuple[str, str]]:
    """
    Take a partial snapshot of a profile for the known-entity table.

    :param g: generator
    :param p: true profile
    :return: list of (attr_type, value)
    """
    rng = g.rng
    rows = [("name", p["name"])]
    if rng.random() < 0.9:
        rows.append(("address", p["address"]))
    rows += [("phone", ph) for ph in p["phones"] if rng.random() < 0.65]
    rows += [("email", e) for e in p["emails"] if rng.random() < 0.5]
    if p["website"] is not None and rng.random() < 0.6:
        rows.append(("website", p["website"]))
    if rng.random() < 0.4:
        rows.append(("tax_id", p["tax_id"]))
    return rows


def drift(g: Generator, p: dict) -> tuple[dict, dict]:
    """
    Apply real-world change between the snapshot and the query.

    :param g: generator
    :param p: true profile at snapshot time
    :return: (current profile, flags)
    """
    rng = g.rng
    q = {k: (list(v) if isinstance(v, list) else v) for k, v in p.items()}
    flags = {"moved": False, "phone_changed": False, "rebranded": False}
    if rng.random() < 0.10:
        # a move changes address and most phones together (correlated drift)
        flags["moved"] = True
        q["address"] = g.address()
        for k in range(len(q["phones"])):
            if rng.random() < 0.8:
                q["phones"][k] = g.phone()
                flags["phone_changed"] = True
    else:
        for k in range(len(q["phones"])):
            if rng.random() < 0.07:
                q["phones"][k] = g.phone()
                flags["phone_changed"] = True
    q["emails"] = [f"{g.token().lower()}{rng.integers(1, 999)}@gmail.com" if rng.random() < 0.05 else e
                   for e in q["emails"]]
    if rng.random() < 0.02:
        flags["rebranded"] = True
        q["name"] = g.name()
    return q, flags


def name_noise(rng, name: str) -> str:
    """
    Apply formatting variation and typos to a name.

    :param rng: numpy generator
    :param name: clean name
    :return: noisy name
    """
    parts = name.split()
    if parts and parts[-1] in {"Inc", "LLC", "Co", "Corp", "Ltd"} and rng.random() < 0.35:
        parts = parts[:-1] + ([rng.choice(["Inc.", "LLC", "Corporation", "Company"])] if rng.random() < 0.5 else [])
    s = " ".join(parts)
    if rng.random() < 0.04:
        s = s.replace("&", "and")
    if rng.random() < 0.05:
        s = s.upper()
    if rng.random() < 0.08 and len(s) > 4:
        k = rng.integers(1, len(s) - 1)
        op = rng.integers(3)
        if op == 0:
            s = s[:k] + s[k + 1:]                       # deletion
        elif op == 1:
            s = s[:k] + s[k + 1] + s[k] + s[k + 2:]     # transposition
        else:
            s = s[:k] + rng.choice(list("aeiourstln")) + s[k + 1:]  # substitution
    return s


def address_noise(rng, addr: str) -> str:
    """
    Apply street-type expansion, suite add/drop, and typos to an address.

    :param rng: numpy generator
    :param addr: clean address
    :return: noisy address
    """
    s = addr
    if rng.random() < 0.4:
        for short, long_ in STREET_TYPES:
            s = s.replace(f" {short},", f" {long_},")
    if rng.random() < 0.15 and " Ste " not in s:
        s = s.replace(",", f" Suite {rng.integers(100, 999)},", 1)
    if rng.random() < 0.06:
        k = rng.integers(2, len(s) - 2)
        s = s[:k] + s[k + 1:]
    return s


def phone_noise(rng, ph: str) -> str:
    """
    Reformat a phone number.

    :param rng: numpy generator
    :param ph: canonical phone
    :return: reformatted phone
    """
    a, e, n = ph.split("-")
    return rng.choice([ph, f"({a}) {e}-{n}", f"{a}{e}{n}", f"+1 {a}.{e}.{n}"])


def make_query(g: Generator, p: dict, all_phones: list[str]) -> tuple[list[tuple[str, str]], dict]:
    """
    Sample 1+ attributes from a current profile and add noise.

    :param g: generator
    :param p: current profile
    :param all_phones: phone pool for contamination
    :return: (rows, flags)
    """
    rng = g.rng
    flags = {"contaminated": False}
    while True:
        rows = []
        if rng.random() < 0.92:
            rows.append(("name", name_noise(rng, p["name"])))
        if rng.random() < 0.7:
            rows.append(("address", address_noise(rng, p["address"])))
        if rng.random() < 0.6:
            k = 2 if (len(p["phones"]) > 1 and rng.random() < 0.15) else 1
            for ph in rng.choice(p["phones"], k, replace=False):
                if rng.random() < 0.02:
                    ph = all_phones[rng.integers(len(all_phones))]   # data-entry error
                    flags["contaminated"] = True
                rows.append(("phone", phone_noise(rng, ph)))
        if p["emails"] and rng.random() < 0.35:
            rows.append(("email", rng.choice(p["emails"])))
        if p["website"] is not None and rng.random() < 0.3:
            rows.append(("website", rng.choice([p["website"], "www." + p["website"], "https://" + p["website"]])))
        if rng.random() < 0.2:
            t = p["tax_id"]
            if rng.random() < 0.015:
                t = t[:-1] + str((int(t[-1]) + 1) % 10)
            rows.append(("tax_id", t))
        if rows:
            return rows, flags


def main(n_entities: int, n_queries: int, p_unknown: float, out: Path, seed: int) -> None:
    """
    Generate and write the dataset.

    :param n_entities: number of known entities
    :param n_queries: number of queries
    :param p_unknown: share of queries whose entity is not in the table
    :param out: output directory
    :param seed: random seed
    """
    g = Generator(seed)
    rng = g.rng
    profiles = build_entities(g, n_entities)
    ent_ids = [f"E{i:06d}" for i in range(n_entities)]

    ent_rows = []
    for eid, p in zip(ent_ids, profiles):
        ent_rows += [(eid, a, v) for a, v in snapshot(g, p)]
    entities = pd.DataFrame(ent_rows, columns=["entity_id", "attr_type", "value"]).drop_duplicates()

    all_phones = [ph for p in profiles for ph in p["phones"]]
    q_rows, truth = [], []
    for qi in range(n_queries):
        qid = f"Q{qi:06d}"
        if rng.random() < p_unknown:
            # unknown entity, sometimes a hard negative
            p = g.profile()
            j = rng.integers(n_entities)
            r = rng.random()
            if r < 0.25:
                p["address"] = profiles[j]["address"]            # same building
            elif r < 0.40:
                tok = profiles[j]["name"].split()[0]
                p["name"] = f"{tok} {rng.choice(INDUSTRIES)} {rng.choice(SUFFIXES, p=SUFFIX_P)}".strip()
            elif r < 0.45:
                p["name"] = profiles[j]["name"]                  # same name, different company
            rows, qflags = make_query(g, p, all_phones)
            truth.append({"query_id": qid, "entity_id": None, "moved": False, "phone_changed": False,
                          "rebranded": False, **qflags})
        else:
            j = rng.integers(n_entities)
            cur, dflags = drift(g, profiles[j])
            rows, qflags = make_query(g, cur, all_phones)
            truth.append({"query_id": qid, "entity_id": ent_ids[j], **dflags, **qflags})
        q_rows += [(qid, a, v) for a, v in rows]

    queries = pd.DataFrame(q_rows, columns=["query_id", "attr_type", "value"]).drop_duplicates()
    out.mkdir(parents=True, exist_ok=True)
    entities.to_csv(out / "entities.csv", index=False)
    queries.to_csv(out / "queries.csv", index=False)
    pd.DataFrame(truth).to_csv(out / "truth.csv", index=False)
    print(f"entities: {entities.entity_id.nunique():,} ({len(entities):,} rows)")
    print(f"queries:  {queries.query_id.nunique():,} ({len(queries):,} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--entities", type=int, default=50_000)
    ap.add_argument("--queries", type=int, default=20_000)
    ap.add_argument("--p-unknown", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(a.entities, a.queries, a.p_unknown, a.out, a.seed)
