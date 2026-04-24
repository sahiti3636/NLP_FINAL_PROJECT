# =============================================================
# STEPS 1 – 3: DATA INGESTION, PREPROCESSING & SILVER ANNOTATION
# =============================================================
# Outputs
#   dream_annotations.json  — one record per sentence segment
# =============================================================
# Install deps:
#   pip install spacy nrclex pandas pyarrow
#   python -m spacy download en_core_web_sm
# =============================================================

import json
import time
import re

import pandas as pd
import spacy
from nrclex import NRCLex

# ════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL LOADING
# ════════════════════════════════════════════════════════════════

print("Loading models...")

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer", first=True)

# NRC canonical 8 emotions (fixed order — never change this list)
NRC_EMOTIONS = ["anger", "anticipation", "disgust", "fear",
                "joy", "sadness", "surprise", "trust"]

print("Models ready.\n")


# ════════════════════════════════════════════════════════════════
# SECTION 2 — STEP 1: DATA INGESTION
# ════════════════════════════════════════════════════════════════

def load_dreambank(limit=None):
    print("Loading DreamBank dataset...")
    df = pd.read_parquet(
        "hf://datasets/DReAMy-lib/DreamBank-dreams-en/"
        "data/train-00000-of-00001-24937aef854be1c9.parquet"
    )
    dreams = df["dreams"].dropna().tolist()
    dreams = [str(d).strip() for d in dreams if len(str(d).strip()) > 20]
    if limit:
        dreams = dreams[:limit]
    print(f"Loaded {len(dreams)} dreams.")
    return dreams


# ════════════════════════════════════════════════════════════════
# SECTION 3 — STEP 2: PREPROCESSING HELPERS
# ════════════════════════════════════════════════════════════════

def simple_tokenize(text: str):
    """
    Lowercase + split on non-alphanumeric.
    Identical to step4_5_combined.py — must never diverge.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def filter_tokens(doc):
    """
    Punct-filtered lowercase token list from a SpaCy doc.
    Applies the same regex as simple_tokenize per token so
    indices stay consistent across the entire pipeline.
    """
    result = []
    for t in doc:
        result.extend(re.findall(r"[a-z0-9]+", t.text.lower()))
    return result


def span_to_filtered_tokens(spacy_token_iter):
    """
    Convert any iterable of SpaCy Token objects to the same
    punct-filtered list that filter_tokens produces.
    Used for building NER/SRL span strings that exactly match
    the stored token list (fixes the old span-mismatch bug).
    """
    result = []
    for t in spacy_token_iter:
        result.extend(re.findall(r"[a-z0-9]+", t.text.lower()))
    return result


# ════════════════════════════════════════════════════════════════
# SECTION 4 — STEP 3a: SILVER NER
# ════════════════════════════════════════════════════════════════

SPACY_TO_PIPELINE_LABEL = {
    "PERSON":      "PER",
    "GPE":         "GPE",
    "LOC":         "LOC",
    "FAC":         "LOC",
    "ORG":         "ORG",
    "NORP":        "GPE",
    "PRODUCT":     "OBJ",
    "WORK_OF_ART": "MISC",
    "EVENT":       "EVT",
    "LANGUAGE":    "MISC",
    "DATE":        "MISC",
    "TIME":        "MISC",
    "CARDINAL":    "MISC",
    "ORDINAL":     "MISC",
    "MONEY":       "MISC",
    "QUANTITY":    "MISC",
    "PERCENT":     "MISC",
}

DREAM_SURREAL = {
    "shadow", "darkness", "void", "abyss", "monster", "creature",
    "beast", "ghost", "spirit", "demon", "angel", "dragon", "witch",
    "wizard", "zombie", "vampire", "skeleton", "phantom", "specter",
    "apparition", "figure", "silhouette", "shape",
}

DREAM_BODY = {
    "teeth", "tooth", "hair", "hand", "hands", "eye", "eyes",
    "face", "mouth", "leg", "legs", "arm", "arms", "foot", "feet",
    "skin", "blood", "heart", "throat", "finger", "fingers", "head",
    "back", "chest", "stomach", "neck", "ear", "ears", "nose",
}

DREAM_OBJECT = {
    "mirror", "door", "window", "key", "knife", "gun", "car",
    "house", "room", "bed", "staircase", "stairs", "ladder",
    "bridge", "wall", "ceiling", "floor", "clock", "phone",
    "book", "map", "letter", "box", "cage", "rope", "mask",
    "ring", "sword", "torch", "light",
}

DREAM_LOCATION = {
    "corridor", "hallway", "forest", "woods", "ocean", "sea",
    "river", "lake", "mountain", "cave", "desert", "sky",
    "space", "building", "castle", "mansion", "school",
    "hospital", "church", "prison", "basement", "attic", "garden",
    "field", "road", "street", "alley", "tunnel", "island",
    "moon", "sun", "cloud", "clouds","house", "room", "building",
}


def apply_dream_rules(filtered_tokens):
    extra = []
    for tok in filtered_tokens:
        if tok in DREAM_SURREAL:
            extra.append((tok, "SURREAL"))
        elif tok in DREAM_BODY:
            extra.append((tok, "OBJ"))
        elif tok in DREAM_OBJECT:
            extra.append((tok, "OBJ"))
        elif tok in DREAM_LOCATION:
            extra.append((tok, "LOC"))
    return extra


def extract_entities(doc, filtered_tokens):
    entities = []
    seen = set()
    for ent in doc.ents:
        label    = SPACY_TO_PIPELINE_LABEL.get(ent.label_, "MISC")
        span_str = " ".join(span_to_filtered_tokens(ent))
        if span_str and span_str not in seen:
            entities.append((span_str, label))
            seen.add(span_str)
    for tok_str, label in apply_dream_rules(filtered_tokens):
        if tok_str not in seen:
            entities.append((tok_str, label))
            seen.add(tok_str)
    return entities


# ════════════════════════════════════════════════════════════════
# SECTION 5 — STEP 3b: SILVER SRL
# ════════════════════════════════════════════════════════════════

TIME_PREPS = {"on", "in", "before", "after", "during", "since", "until", "by"}
DIR_PREPS  = {"to", "toward", "towards", "into", "onto", "from", "off", "out", "through"}

TIME_NOUNS = {
    "night", "day", "morning", "evening", "afternoon", "midnight",
    "dawn", "dusk", "moment", "time", "hour", "minute", "second",
    "week", "month", "year", "yesterday", "today", "tomorrow",
}


def _tight_np(head_token):
    """
    Tight NP: det + amod + compound + head only.
    Stops before prep/clause nodes to prevent cross-clause bleed.
    """
    SAFE_DEPS = {"det", "amod", "compound", "poss", "nummod", "nn", "nmod"}
    collected = {head_token}
    for child in head_token.children:
        if child.dep_ in SAFE_DEPS:
            collected.add(child)
            for grandchild in child.children:
                if grandchild.dep_ in SAFE_DEPS:
                    collected.add(grandchild)
    return sorted(collected, key=lambda t: t.i)


def _is_temporal_pobj(pobj_token):
    if pobj_token.ent_type_ in ("DATE", "TIME"):
        return True
    if pobj_token.lemma_.lower() in TIME_NOUNS:
        return True
    return False


def extract_srl(doc, filtered_tokens):
    relations = []

    for token in doc:
        if token.pos_ != "VERB":
            continue

        args = []

        for child in token.lefts:
            if child.dep_ in ("nsubj", "nsubjpass"):
                span_str = " ".join(span_to_filtered_tokens(_tight_np(child)))
                if span_str:
                    args.append({"role": "ARG0", "span": span_str})

        for child in token.rights:
            if child.dep_ in ("dobj", "attr", "oprd"):
                span_str = " ".join(span_to_filtered_tokens(_tight_np(child)))
                if span_str:
                    args.append({"role": "ARG1", "span": span_str})

        for child in token.rights:
            if child.dep_ == "dative":
                span_str = " ".join(span_to_filtered_tokens(_tight_np(child)))
                if span_str:
                    args.append({"role": "ARG2", "span": span_str})

        for child in token.rights:
            if child.dep_ != "prep":
                continue
            prep_filtered = span_to_filtered_tokens([child])
            if not prep_filtered:
                continue
            prep_str = prep_filtered[0]
            raw_prep = child.text.lower()
            for pobj in [c for c in child.children if c.dep_ == "pobj"]:
                pobj_filtered = span_to_filtered_tokens(list(pobj.subtree))
                if not pobj_filtered:
                    continue
                full_span = prep_str + " " + " ".join(pobj_filtered)
                if raw_prep in TIME_PREPS and _is_temporal_pobj(pobj):
                    role = "ARGM-TMP"
                elif raw_prep in DIR_PREPS:
                    role = "ARGM-DIR"
                else:
                    role = "ARGM-LOC"
                args.append({"role": role, "span": full_span})

        for child in token.children:
            if child.dep_ == "advmod":
                span_str = " ".join(span_to_filtered_tokens([child]))
                if span_str:
                    args.append({"role": "ARGM-MNR", "span": span_str})

        for child in token.children:
            if child.dep_ == "neg":
                span_str = " ".join(span_to_filtered_tokens([child]))
                if span_str:
                    args.append({"role": "ARGM-NEG", "span": span_str})

        for child in token.rights:
            if child.dep_ == "advcl":
                markers = [c.text.lower() for c in child.children if c.dep_ == "mark"]
                if any(m in {"because", "since", "as", "given"} for m in markers):
                    span_str = " ".join(span_to_filtered_tokens(list(child.subtree)))
                    if span_str:
                        args.append({"role": "ARGM-CAU", "span": span_str})

        verb_filtered = span_to_filtered_tokens([token])
        verb_str      = verb_filtered[0] if verb_filtered else token.text.lower()
        verb_index    = next(
            (i for i, t in enumerate(filtered_tokens) if t == verb_str), None
        )

        relations.append({
            "verb":       verb_str,
            "verb_index": verb_index,
            "args":       args,
        })

    return relations


# ════════════════════════════════════════════════════════════════
# SECTION 6 — STEP 3c: SILVER COREFERENCE
# ════════════════════════════════════════════════════════════════

PRONOUN_MAP = {
    "i": "the dreamer", "me": "the dreamer", "my": "the dreamer",
    "myself": "the dreamer", "mine": "the dreamer",
    "he": None, "him": None, "his": None,
    "she": None, "her": None, "hers": None,
    "it": None, "its": None,
    "they": None, "them": None, "their": None,
    "we": None, "us": None, "our": None,
}


def resolve_coref(doc):
    substitutions = []
    last_person = None
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG"):
            last_person = ent.text.lower()
    for token in doc:
        lower = token.text.lower()
        if lower not in PRONOUN_MAP:
            continue
        resolved = PRONOUN_MAP[lower]
        if resolved is None:
            resolved = last_person if last_person else lower
        substitutions.append({"pronoun": lower, "resolved": resolved})
    return substitutions


# ════════════════════════════════════════════════════════════════
# SECTION 7 — STEP 3d: NRC EMOTION VECTORS
#
# Replaces GoEmotions BERT entirely.
#
# WHY NRC INSTEAD OF GOEMOTIONS:
#   GoEmotions was trained on Reddit comments — short reactive
#   social text. Dream narratives are symbolic and first-person
#   experiential. "teeth shattered in a mirror" has no Reddit
#   analogue so the model pattern-matched "laughing" → amusement.
#
#   NRC Emotion Lexicon maps ~14,000 words to 8 emotions via
#   crowdsourced human annotation. It covers abstract and symbolic
#   vocabulary (teeth → fear, mirror → disgust/fear) which makes
#   it far more appropriate for dream text.
#
#   Output: 8-dim normalised float vector per segment.
#   Columns follow NRC_EMOTIONS order (fixed at top of file).
# ════════════════════════════════════════════════════════════════

def get_nrc_emotion_vector(text: str) -> dict:
    """
    Returns a normalised dict {emotion: score} for all 8 NRC emotions.
    Scores are raw lexicon hit counts normalised to sum to 1.
    If no lexicon words are found, returns uniform distribution.
    """
    nrc    = NRCLex(text)
    raw    = nrc.raw_emotion_scores

    # Keep only the 8 canonical emotions, fill missing with 0
    scores = {emo: raw.get(emo, 0) for emo in NRC_EMOTIONS}
    total  = sum(scores.values())

    if total == 0:
        # No lexicon hits — return uniform so downstream gets a
        # non-zero vector rather than a meaningless zero vector
        return {emo: round(1.0 / len(NRC_EMOTIONS), 4) for emo in NRC_EMOTIONS}

    return {emo: round(scores[emo] / total, 4) for emo in NRC_EMOTIONS}


# ════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN PIPELINE
# ════════════════════════════════════════════════════════════════

def run_pipeline(limit=None):
    start = time.time()

    dreams = load_dreambank(limit=limit)
    if not dreams:
        print("No dreams loaded. Exiting.")
        return

    # ── Step 2: Sentence segmentation (fast pass) ─────────────
    print("Segmenting dreams into sentences...")
    all_sentences = []
    for doc in nlp.pipe(dreams, disable=["ner", "parser"], batch_size=256):
        for sent in doc.sents:
            text = sent.text.strip()
            if len(text) > 5:
                all_sentences.append(text)
    print(f"Total sentence segments: {len(all_sentences)}")

    # ── Step 3a/b/c: NER + SRL + Coref ───────────────────────
    print("Running NER, SRL, and coreference extraction...")
    final_results = []
    for doc in nlp.pipe(all_sentences, batch_size=128):
        filtered_tokens = filter_tokens(doc)
        entities        = extract_entities(doc, filtered_tokens)
        relations       = extract_srl(doc, filtered_tokens)
        coref           = resolve_coref(doc)

        # ── Step 3d: NRC emotion vector (CPU, no GPU needed) ──
        emotion_vector = get_nrc_emotion_vector(doc.text)

        final_results.append({
            "text":           doc.text,
            "tokens":         filtered_tokens,
            "entities":       entities,
            "relations":      relations,
            "coref":          coref,
            "emotion_vector": emotion_vector,
        })

    # ── Save ──────────────────────────────────────────────────
    print("Saving dream_annotations.json...")
    with open("jsons/dream_annotations.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start
    print(f"\nDone. {len(final_results)} segments saved. ({elapsed:.1f}s)")

    # ── Sanity check: first 3 segments ────────────────────────
    for sample in final_results[:3]:
        print(f"\n{'─'*60}")
        print(f"TEXT   : {sample['text']}")
        print(f"TOKENS : {sample['tokens']}")
        print("ENTITIES:")
        for e in sample["entities"]:
            print(f"  [{e[1]}] {e[0]}")
        print("SRL:")
        for rel in sample["relations"]:
            print(f"  VERB '{rel['verb']}' (idx={rel['verb_index']})")
            for arg in rel["args"]:
                print(f"    {arg['role']:<12} -> '{arg['span']}'")
        top_emo = sorted(sample["emotion_vector"].items(), key=lambda x: -x[1])[:3]
        print(f"TOP EMOTIONS: {top_emo}")


if __name__ == "__main__":
    run_pipeline(limit=None)
