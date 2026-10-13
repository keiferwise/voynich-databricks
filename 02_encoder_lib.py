# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Encoder Library
# MAGIC
# MAGIC This notebook is **sourced by other notebooks** via `%run ./02_encoder_lib`.
# MAGIC It contains all encoder, decoder, and statistical test code.
# MAGIC Do not add pipeline or I/O logic here.
# MAGIC
# MAGIC **Before running:** update `CATALOG` below if your default catalog is not `main`.

# COMMAND ----------

# ── Configuration ────────────────────────────────────────────────────────────
CATALOG = "workspace"     # ← update if your default catalog is not "main"
SCHEMA  = "voynich"
VOLUME  = "raw_files"

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
VOCAB_PATH  = f"{VOLUME_PATH}/voynichtokencounts.csv"
# ─────────────────────────────────────────────────────────────────────────────

# COMMAND ----------

# MAGIC %md ### 1. Imports and Constants

# COMMAND ----------

import random
import re
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter, defaultdict
from scipy import stats
from sklearn.model_selection import KFold

GALLOWS          = ["k", "t", "f", "p"]   # EVA gallows characters — paragraph markers
MIN_TOKEN_LENGTH = 3

# COMMAND ----------

# MAGIC %md ### 2. Load EVA Vocabulary

# COMMAND ----------

eva_df          = pd.read_csv(VOCAB_PATH)
EVA_WORD_FREQ   = dict(zip(eva_df["token"], eva_df["count"]))
EVA_WORDS_BY_FREQ   = eva_df["token"].tolist()
EVA_WEIGHTS_BY_FREQ = eva_df["count"].tolist()

MIN_ANCHOR_LENGTH = 3

EVA_ANCHOR_CANDIDATES = eva_df[
    (eva_df["token"].str.len() >= MIN_ANCHOR_LENGTH) &
    (~eva_df["token"].isin(GALLOWS))
]["token"].tolist()

EVA_ANCHOR_WEIGHTS = eva_df[
    (eva_df["token"].str.len() >= MIN_ANCHOR_LENGTH) &
    (~eva_df["token"].isin(GALLOWS))
]["count"].tolist()

print(f"Vocabulary loaded: {len(EVA_WORD_FREQ)} tokens")
print(f"Anchor candidates (len >= {MIN_ANCHOR_LENGTH}): {len(EVA_ANCHOR_CANDIDATES)}")

# COMMAND ----------

# MAGIC %md ### 3. EVA Fragment Lists and Parser

# COMMAND ----------

EVA_PREFIXES = ["qo", "ch", "sh", "d", "s", "l", "r", "t", "k", "f", "p"]
EVA_CORES    = ["aiin", "ain", "air", "al", "ar", "cheol", "chol", "dain",
                "iin", "in", "ol", "or", "kain", "kan", "kar",
                "kor", "lain", "olor", "rain", "ral"]
EVA_SUFFIXES = ["eedy", "edy", "eey", "ey", "dy"]

# Longest match first — prevents short prefixes eating into longer ones
EVA_PREFIXES = sorted(EVA_PREFIXES, key=len, reverse=True)
EVA_CORES    = sorted(set(EVA_CORES), key=len, reverse=True)
EVA_SUFFIXES = sorted(EVA_SUFFIXES, key=len, reverse=True)


@dataclass
class ParsedWord:
    prefix: str
    core:   str
    suffix: str

    def reconstruct(self) -> str:
        return self.prefix + self.core + self.suffix

    def __repr__(self):
        return f"[{self.prefix}|{self.core}|{self.suffix}]"


def parse_eva_word(token: str) -> ParsedWord:
    """
    Split an EVA token into (prefix, core, suffix).
    Greedy longest-match: strip prefix from left, suffix from right, core is remainder.
    """
    remaining = token
    prefix = ""
    suffix = ""

    for p in EVA_PREFIXES:
        if remaining.startswith(p):
            prefix    = p
            remaining = remaining[len(p):]
            break

    for s in EVA_SUFFIXES:
        if remaining.endswith(s) and len(remaining) > len(s):
            suffix    = s
            remaining = remaining[:-len(s)]
            break

    return ParsedWord(prefix=prefix, core=remaining, suffix=suffix)

# COMMAND ----------

# MAGIC %md ### 4. Dataclasses

# COMMAND ----------

@dataclass
class Rule:
    """A single encoding rule."""
    name:          str
    distance:      int           # distance this rule encodes (0–25)
    rule_type:     str           # "anchor" | "base_word" | "mutation"
    token:         Optional[str] = None
    mutation_type: Optional[str] = None
    fragment:      Optional[str] = None
    lookback:      int           = 1
    weight:        float         = 1.0


@dataclass
class AnchorWord:
    """Maps a specific EVA token to an absolute alphabet position (0–25)."""
    token:    str
    position: int


@dataclass
class EncoderConfig:
    anchors:            list
    rules:              list
    shaping_rules:      list  = field(default_factory=list)
    anchor_probability: float = 0.15
    force_anchor_start: bool  = True
    paragraph_gallows:  bool  = True

# COMMAND ----------

# MAGIC %md ### 5. Mutation Engine

# COMMAND ----------

def apply_mutation(token: str, rule: Rule, history: list) -> Optional[str]:
    """
    Apply a mutation rule using history[-rule.lookback] as the base token.
    Returns the mutated EVA token, or None if not applicable.
    """
    if len(history) < rule.lookback:
        return None

    base = history[-rule.lookback]
    if not base or base in GALLOWS:
        return None

    parsed = parse_eva_word(base)
    result = None

    match rule.mutation_type:
        case "prefix_change":
            result = ParsedWord(rule.fragment, parsed.core, parsed.suffix).reconstruct()
        case "suffix_change":
            result = ParsedWord(parsed.prefix, parsed.core, rule.fragment).reconstruct()
        case "prefix_remove":
            if not parsed.prefix:
                return None
            result = ParsedWord("", parsed.core, parsed.suffix).reconstruct()
        case "suffix_remove":
            if not parsed.suffix:
                return None
            result = ParsedWord(parsed.prefix, parsed.core, "").reconstruct()
        case "core_insert":
            for vowel in ["a", "o", "i", "e"]:
                if vowel in parsed.core:
                    new_core = parsed.core.replace(vowel, vowel + rule.fragment, 1)
                    result = ParsedWord(parsed.prefix, new_core, parsed.suffix).reconstruct()
                    break
            else:
                return None
        case "core_delete":
            for doubled in ["ii", "ee", "oo", "aa"]:
                if doubled in parsed.core:
                    new_core = parsed.core.replace(doubled, doubled[0], 1)
                    result = ParsedWord(parsed.prefix, new_core, parsed.suffix).reconstruct()
                    break
            else:
                return None
        case "core_replace":
            result = ParsedWord(parsed.prefix, rule.fragment, parsed.suffix).reconstruct()
        case _:
            return None

    if not result or len(result) < MIN_TOKEN_LENGTH or result in GALLOWS:
        return None
    return result


ALL_PREFIXES = ["qo", "ch", "sh", "d", "s", "l", "r", "t", "k", "f", "p"]
ALL_SUFFIXES = ["eedy", "edy", "eey", "ey", "dy", "y"]
ALL_CORES    = ["ain", "aiin", "ol", "or", "al", "in", "iin", "kan", "kor"]


def generate_deterministic_mutations() -> list:
    mutations = []
    for prefix in ALL_PREFIXES:
        mutations.append({"name": f"change_{prefix}_prefix",
                          "mutation_type": "prefix_change", "fragment": prefix})
    mutations.append({"name": "remove_prefix",
                      "mutation_type": "prefix_remove", "fragment": None})
    for suffix in ALL_SUFFIXES:
        mutations.append({"name": f"change_{suffix}_suffix",
                          "mutation_type": "suffix_change", "fragment": suffix})
    mutations.append({"name": "remove_suffix",
                      "mutation_type": "suffix_remove", "fragment": None})
    for core in ALL_CORES:
        mutations.append({"name": f"insert_{core}_core",
                          "mutation_type": "core_insert", "fragment": core})
        mutations.append({"name": f"replace_{core}_core",
                          "mutation_type": "core_replace", "fragment": core})
    mutations.append({"name": "core_delete",
                      "mutation_type": "core_delete", "fragment": None})
    print(f"Generated {len(mutations)} deterministic mutations")
    return mutations


DETERMINISTIC_MUTATIONS = generate_deterministic_mutations()

# COMMAND ----------

# MAGIC %md ### 6. Ruleset Generation

# COMMAND ----------

def synthesize_eva_word(existing_tokens: set, rng: random.Random) -> str:
    """Sample a unique EVA token not already in existing_tokens."""
    candidates = [w for w in EVA_ANCHOR_CANDIDATES if w not in existing_tokens]
    weights    = [EVA_WORD_FREQ[w] for w in candidates]

    if candidates:
        return rng.choices(candidates, weights=weights, k=1)[0]

    # Fallback: build from fragments
    for _ in range(100):
        prefix = rng.choice(EVA_PREFIXES)
        core   = rng.choice(EVA_CORES)
        suffix = rng.choice(EVA_SUFFIXES)
        token  = prefix + core + suffix
        if token not in existing_tokens and len(token) >= MIN_ANCHOR_LENGTH:
            return token

    raise ValueError("Could not synthesize a unique EVA token after 100 attempts")


def generate_random_ruleset(seed=None, num_anchors=26):
    rng = random.Random(seed)

    anchor_tokens   = rng.sample(EVA_ANCHOR_CANDIDATES, k=num_anchors)
    existing_tokens = set(anchor_tokens)
    anchors = [AnchorWord(token=t, position=i) for i, t in enumerate(anchor_tokens)]

    distances = list(range(0, 26))
    rng.shuffle(distances)
    distances = distances + rng.choices(range(0, 26), k=len(DETERMINISTIC_MUTATIONS) - 26)

    rules = []
    for i, mut in enumerate(DETERMINISTIC_MUTATIONS):
        rules.append(Rule(
            name          = mut["name"],
            distance      = distances[i],
            rule_type     = "mutation",
            mutation_type = mut["mutation_type"],
            fragment      = mut["fragment"],
            lookback      = rng.choice([1, 1, 1, 2, 3]),
            weight        = rng.uniform(0.5, 2.0),
        ))

    for distance in range(0, 26):
        token = synthesize_eva_word(existing_tokens, rng)
        existing_tokens.add(token)
        rules.append(Rule(
            name      = f"base_word_{distance}",
            distance  = distance,
            rule_type = "base_word",
            token     = token,
            weight    = 0.3,
        ))

    return anchors, rules


def generate_random_config(seed=None, num_anchors=26, anchor_probability=0.15,
                           shaping_rules=None) -> EncoderConfig:
    anchors, rules = generate_random_ruleset(seed=seed, num_anchors=num_anchors)
    return EncoderConfig(
        anchors            = anchors,
        rules              = rules,
        shaping_rules      = shaping_rules or [],
        anchor_probability = anchor_probability,
        force_anchor_start = True,
        paragraph_gallows  = True,
    )

# COMMAND ----------

# MAGIC %md ### 7. Encoder

# COMMAND ----------

def would_collide(token: str, anchors: list, rules: list) -> bool:
    if not token or len(token) < MIN_TOKEN_LENGTH or token in GALLOWS:
        return True
    anchor_tokens    = {a.token for a in anchors}
    base_word_tokens = {r.token for r in rules if r.rule_type == "base_word"}
    return token in anchor_tokens or token in base_word_tokens


def get_anchor_for_position(position: int, anchors: list,
                            rng: random.Random) -> Optional[AnchorWord]:
    matches = [a for a in anchors if a.position == position]
    return rng.choice(matches) if matches else None


def get_valid_rules_for_distance(distance, history, anchors, rules):
    candidates = []
    for rule in rules:
        if rule.distance != distance:
            continue
        if rule.rule_type == "base_word":
            candidates.append((rule, rule.token))
            continue
        if rule.rule_type == "mutation":
            result = apply_mutation(history[-1] if history else "", rule, history)
            if result is None:
                continue
            if would_collide(result, anchors, rules):
                continue
            ambiguous = False
            for other_rule in rules:
                if other_rule.rule_type != "mutation":
                    continue
                if other_rule.name == rule.name:
                    continue
                other_result = apply_mutation(
                    history[-other_rule.lookback] if len(history) >= other_rule.lookback else "",
                    other_rule, history
                )
                if other_result == result and other_rule.distance != distance:
                    ambiguous = True
                    break
            if not ambiguous:
                candidates.append((rule, result))
    return candidates


def apply_shaping(candidates: list, shaping_rules: list, rng: random.Random) -> str:
    if not shaping_rules:
        rules_list, tokens = zip(*candidates)
        weights = [r.weight for r in rules_list]
        return rng.choices(tokens, weights=weights, k=1)[0]
    weights = []
    for rule, token in candidates:
        w = rule.weight
        for condition_fn, multiplier in shaping_rules:
            if condition_fn(token):
                w *= multiplier
        weights.append(w)
    _, tokens = zip(*candidates)
    return rng.choices(list(tokens), weights=weights, k=1)[0]


def encode(plaintext: str, config: EncoderConfig, rng: random.Random,
           history: list = None, position: int = 0,
           is_first: bool = True) -> list:
    """
    All random choices (gallows, anchor roll, weighted candidate pick) come
    from `rng`, so the same rule set + rng state always gives the same output.
    The choices only pick between equivalent spellings; decoding never needs rng.
    """
    ALPHA    = "abcdefghijklmnopqrstuvwxyz"
    history  = history or []
    output   = []

    for ch in plaintext.lower():
        if ch not in ALPHA:
            output.append(".")
            continue

        target     = ALPHA.index(ch)
        distance   = (target - position) % 26
        candidates = get_valid_rules_for_distance(distance, history,
                                                  config.anchors, config.rules)
        anchor     = get_anchor_for_position(target, config.anchors, rng)

        if is_first:
            if not anchor:
                raise ValueError(f"No anchor for first character '{ch}'")
            if config.paragraph_gallows:
                gallows_token = rng.choice(GALLOWS)
                output.append(gallows_token)
                history.append(gallows_token)
            token    = anchor.token
            is_first = False
        elif not candidates or rng.random() < config.anchor_probability:
            if anchor:
                token = anchor.token
            elif candidates:
                token = apply_shaping(candidates, config.shaping_rules, rng)
            else:
                raise ValueError(
                    f"No rule or anchor for distance {distance} "
                    f"('{ALPHA[position]}' -> '{ch}')"
                )
        else:
            token = apply_shaping(candidates, config.shaping_rules, rng)

        output.append(token)
        history.append(token)
        position = target

    return output


def clean_text(text: str) -> str:
    return re.sub(r'[^a-z]', '', text.lower()).strip()


def encode_text(text: str, config: EncoderConfig, chunk_size: int = 100,
                encode_seed: int = 0) -> list:
    """
    Two seeds define a ciphertext:
      - rule seed   (in `config`): which rule set, i.e. the key needed to decode
      - encode_seed:             which of that rule set's many valid spellings
    Each call gets its own RNG, so threads never share random state.
    """
    rng      = random.Random(encode_seed)
    cleaned  = clean_text(text)
    chunks   = [cleaned[i:i+chunk_size] for i in range(0, len(cleaned), chunk_size)]
    output   = []
    ALPHA    = "abcdefghijklmnopqrstuvwxyz"
    history  = []
    position = 0
    is_first = True

    for chunk in chunks:
        if not chunk:
            continue
        encoded = encode(chunk, config, rng, history=history,
                         position=position, is_first=is_first)
        output.extend(encoded)
        output.append("\n")
        history  = encoded
        is_first = False
        last_char = chunk[-1]
        if last_char in ALPHA:
            position = ALPHA.index(last_char)

    return output

# COMMAND ----------

# MAGIC %md ### 8. Decoder

# COMMAND ----------

def decode(tokens: list, anchors: list, rules: list) -> str:
    ALPHA    = "abcdefghijklmnopqrstuvwxyz"
    history  = []
    output   = []
    position = 0

    for token in tokens:
        if token == ".":
            output.append(" ")
            continue
        if token in GALLOWS or token == "\n":
            history.append(token)
            continue

        anchor_match = next((a for a in anchors if a.token == token), None)
        if anchor_match:
            output.append(ALPHA[anchor_match.position])
            position = anchor_match.position
            history.append(token)
            continue

        base_match = next((r for r in rules
                           if r.rule_type == "base_word" and r.token == token), None)
        if base_match:
            position = (position + base_match.distance) % 26
            output.append(ALPHA[position])
            history.append(token)
            continue

        decoded = False
        for rule in rules:
            if rule.rule_type != "mutation":
                continue
            if len(history) < rule.lookback:
                continue
            result = apply_mutation(history[-rule.lookback], rule, history)
            if result == token:
                position = (position + rule.distance) % 26
                output.append(ALPHA[position])
                history.append(token)
                decoded = True
                break

        if not decoded:
            output.append("?")
            history.append(token)

    return "".join(output)


def decode_text(tokens: list, anchors: list, rules: list) -> str:
    flat = [t for t in tokens if t != "\n"]
    return decode(flat, anchors, rules)

# COMMAND ----------

# MAGIC %md ### 9. Statistical Tests

# COMMAND ----------

def compute_zipf(tokens: list, text_name: str) -> dict:
    freq   = Counter(tokens)
    ranks  = np.arange(1, len(freq) + 1)
    counts = np.array([c for _, c in freq.most_common()])
    slope, intercept, r_value, *_ = stats.linregress(np.log(ranks), np.log(counts))
    return {
        f"{text_name}_zipf_slope":     round(float(slope), 4),
        f"{text_name}_zipf_r_squared": round(float(r_value**2), 4),
        f"{text_name}_vocab_size":     len(freq),
        f"{text_name}_token_count":    len(tokens),
    }


def compute_entropy(tokens: list, text_name: str) -> dict:
    freq    = Counter(tokens)
    total   = len(tokens)
    probs   = np.array([c / total for c in freq.values()])
    entropy = -np.sum(probs * np.log2(probs))
    return {f"{text_name}_unigram_entropy": round(float(entropy), 4)}


def compute_basic_stats(tokens: list, text_name: str) -> dict:
    freq  = Counter(tokens)
    hapax = sum(1 for c in freq.values() if c == 1)
    return {
        f"{text_name}_word_count":        len(tokens),
        f"{text_name}_unique_words":      len(freq),
        f"{text_name}_hapax_count":       hapax,
        f"{text_name}_hapax_ratio":       round(hapax / len(freq), 4),
        f"{text_name}_most_common_word":  freq.most_common(1)[0][0],
        f"{text_name}_most_common_count": freq.most_common(1)[0][1],
    }


def compute_token_lengths(tokens: list, text_name: str) -> dict:
    lengths = [len(t) for t in tokens]
    return {
        f"{text_name}_avg_length":    round(float(np.mean(lengths)), 4),
        f"{text_name}_median_length": round(float(np.median(lengths)), 4),
        f"{text_name}_std_length":    round(float(np.std(lengths)), 4),
        f"{text_name}_min_length":    int(min(lengths)),
        f"{text_name}_max_length":    int(max(lengths)),
    }


def compute_letter_predictability(tokens: list, text_name: str,
                                  n_splits: int = 10) -> dict:
    word_types = list(set(tokens))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    unigram_scores, bigram_scores = [], []

    for train_idx, test_idx in kf.split(word_types):
        train_words = [word_types[i] for i in train_idx]
        test_words  = [word_types[i] for i in test_idx]

        char_freq  = Counter(c for w in train_words for c in w)
        best_char  = char_freq.most_common(1)[0][0]
        test_chars = [c for w in test_words for c in w]
        unigram_scores.append(
            sum(1 for c in test_chars if c == best_char) / len(test_chars) * 100
        )

        bigram_freq = defaultdict(Counter)
        for word in train_words:
            for a, b in zip(word, word[1:]):
                bigram_freq[a][b] += 1

        correct, total = 0, 0
        for word in test_words:
            for a, b in zip(word, word[1:]):
                if a in bigram_freq:
                    if bigram_freq[a].most_common(1)[0][0] == b:
                        correct += 1
                total += 1
        if total > 0:
            bigram_scores.append(correct / total * 100)

    return {
        f"{text_name}_letter_unigram_predictability": round(float(np.mean(unigram_scores)), 2),
        f"{text_name}_letter_bigram_predictability":  round(float(np.mean(bigram_scores)), 2),
    }

# COMMAND ----------

# MAGIC %md ### 10. Stat Test Registry
# MAGIC
# MAGIC To add a test: write the function above, then add it here.
# MAGIC To remove a test: comment out its line.
# MAGIC Each function must accept `(tokens: list, text_name: str)` and return a `dict`.

# COMMAND ----------

STAT_TESTS = {
    "zipf":          compute_zipf,
    "entropy":       compute_entropy,
    "hapax":         compute_basic_stats,
    "token_lengths": compute_token_lengths,
    "letter_pred":   compute_letter_predictability,
}


def run_all_stats(encoded: list, text_name: str) -> dict:
    """Run every registered stat test and merge results into one dict."""
    tokens  = [t for t in encoded if t not in GALLOWS and t != "." and t != "\n"]
    results = {}
    for test_fn in STAT_TESTS.values():
        results.update(test_fn(tokens, text_name))
    return results


print("Encoder library loaded.")
print(f"  Volume path: {VOLUME_PATH}")
print(f"  Stat tests:  {list(STAT_TESTS.keys())}")