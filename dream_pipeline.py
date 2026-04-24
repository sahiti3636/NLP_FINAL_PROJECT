"""
dream_pipeline.py
=================
Dreamscape Mapper — End-to-End Integration Pipeline
Simulates Steps 1-10 for a single user-supplied dream string and
returns the structured thematic-emotional JSON record.

NRC Emotion Lexicon is used for emotion classification (not GoEmotions).
NRC emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, trust,
               negative, positive  (the 8 basic + 2 valence dimensions)

Drop-in design: each step is a self-contained function so you can swap in the
real trained models (BiLSTM, T5, UMAP/HDBSCAN, etc.) without touching the
integration harness.
"""

import re
import json
import math
import string
from collections import defaultdict, Counter
from typing import Dict, List, Tuple  # Added for Python 3.8 compatibility

# ---------------------------------------------------------------------------
# Optional heavy-model imports — wrapped so the script still runs with stubs
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

# ============================================================
# STEP 1 — Raw Dream Text (identity pass-through)
# ============================================================

def step1_ingest(raw_text: str) -> str:
    """Accept and validate a raw dream string."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Dream text must be a non-empty string.")
    return raw_text.strip()


# ============================================================
# STEP 2 — Text Preprocessing
# ============================================================

def step2_preprocess(raw_text: str) -> dict:
    """
    Tokenize, lowercase, lemmatize, sentence-segment, and split into
    narrative segments (one segment per sentence).
    Returns a dict with keys: tokens, lemmas, sentences, segments.
    """
    if _NLTK_AVAILABLE:
        lemmatizer = WordNetLemmatizer()
        sentences = sent_tokenize(raw_text)
        tokens = word_tokenize(raw_text.lower())
        tokens = [t for t in tokens if t not in string.punctuation]
        lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    else:
        # Minimal fallback
        sentences = re.split(r'[.!?]+', raw_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        tokens = re.findall(r"\b\w+\b", raw_text.lower())
        lemmas = tokens  # no lemmatization without NLTK

    segments = [s.strip() for s in sentences if s.strip()]

    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "sentences": sentences,
        "segments": segments,
    }


# ============================================================
# STEP 3 — Silver Annotation (NER + SRL + Coreference)
# ============================================================

# --- NER rule-based heuristics (mocked silver labels) ---
_CHARACTER_HINTS = {
    "mother", "father", "sister", "brother", "friend", "man", "woman",
    "child", "teacher", "doctor", "stranger", "he", "she", "they", "i",
    "me", "we", "him", "her", "them", "us",
}
_BODY_OBJECT_HINTS = {
    "teeth", "tooth", "hand", "hands", "eye", "eyes", "hair", "face",
    "arm", "leg", "skin", "blood", "mouth", "finger", "fingers", "nail",
    "nails", "head", "back", "chest",
}
_SETTING_OBJECT_HINTS = {
    "mirror", "door", "window", "room", "house", "school", "car", "road",
    "street", "bridge", "water", "ocean", "sea", "forest", "building",
    "staircase", "stairs", "elevator", "hall", "hallway", "office", "garden",
    "park", "sky", "ceiling", "floor", "wall", "bed", "table",
}
_ACTION_VERBS = {
    "ran", "run", "fell", "fall", "falling", "fly", "flew", "flying",
    "laughed", "laugh", "screamed", "scream", "cried", "cry", "chased",
    "chase", "looked", "look", "saw", "see", "felt", "feel", "heard",
    "hear", "grabbed", "grab", "pushed", "push", "pulled", "pull",
    "opened", "open", "closed", "close", "walked", "walk", "stood", "stand",
    "shattered", "shatter", "broke", "break", "started", "start",
}

def step3_silver_annotate(preprocessed: dict) -> dict:
    """
    Produce silver NER tags, a single SRL triplet, basic coreference, and
    per-segment emotion stubs (filled properly in Step 5).
    """
    tokens = preprocessed["tokens"]
    lemmas = preprocessed["lemmas"]
    segments = preprocessed["segments"]

    # --- NER ---
    entities = {"Character": [], "Body-Object": [], "Setting-Object": []}
    for tok in tokens:
        if tok in _CHARACTER_HINTS and tok not in ("i", "me", "we", "us"):
            entities["Character"].append(tok.capitalize())
        elif tok in _BODY_OBJECT_HINTS:
            entities["Body-Object"].append(tok.capitalize())
        elif tok in _SETTING_OBJECT_HINTS:
            entities["Setting-Object"].append(tok.capitalize())

    # Deduplicate while preserving order
    for k in entities:
        seen = set()
        unique = []
        for v in entities[k]:
            if v not in seen:
                seen.add(v)
                unique.append(v)
        entities[k] = unique

    # --- SRL (heuristic: find all action verbs, assign agent/target) ---
    srl_list = []
    all_text_tokens = tokens  # lowercase
    for i, tok in enumerate(all_text_tokens):
        if tok in _ACTION_VERBS:
            srl = {"Agent": "Unknown", "Action": tok.capitalize(), "Target": "Unknown"}
            # Agent = first character entity found before verb
            chars = entities["Character"]
            srl["Agent"] = chars[0] if chars else "I"
            # Target = next non-stopword after verb (naïve)
            stopwords = {"at", "to", "the", "a", "an", "and", "in", "on", "with", "by"}
            for j in range(i + 1, min(i + 4, len(all_text_tokens))):
                if all_text_tokens[j] not in stopwords:
                    srl["Target"] = all_text_tokens[j].capitalize()
                    break
            srl_list.append(srl)

    if not srl_list:
        srl_list = [{"Agent": "Unknown", "Action": "Unknown", "Target": "Unknown"}]

    # --- Coreference (trivial: map gendered pronouns to first named character) ---
    coref = {}
    first_char = entities["Character"][0] if entities["Character"] else None
    if first_char:
        for pronoun in ("she", "he", "they", "her", "him", "them"):
            if pronoun in tokens:
                coref[pronoun] = first_char

    # --- Emotion stubs (empty dict; Step 5 fills these) ---
    emotion_stubs = {seg: {} for seg in segments}

    return {
        "entities": entities,
        "srl": srl_list,
        "coreference": coref,
        "emotion_stubs": emotion_stubs,
    }


# ============================================================
# STEP 4 — Multi-Task BiLSTM (mocked — returns enriched entity/SRL signals)
# ============================================================

def step4_bilstm_representations(preprocessed: dict, annotations: dict) -> dict:
    """
    In production this loads step4_bilstm.pt and encodes tokens.
    Here we return a lightweight symbolic embedding: a bag-of-entity-counts
    vector that downstream steps treat as the 'contextual representation'.
    """
    entity_counts = {}
    for etype, elist in annotations["entities"].items():
        entity_counts[etype] = len(elist)

    # Symbolic 'embedding': normalised entity counts + action flag
    total = max(sum(entity_counts.values()), 1)
    embedding_stub = {k: round(v / total, 4) for k, v in entity_counts.items()}
    srl_list = annotations.get("srl", [{"Action": "Unknown"}])
    embedding_stub["has_action"] = 1 if any(s["Action"] != "Unknown" for s in srl_list) else 0

    return {
        "contextual_embedding": embedding_stub,
        "entity_presence_vector": entity_counts,
        "srl_verb_signature": srl_list[0]["Action"] if srl_list else "Unknown",
    }


# ============================================================
# STEP 5 — Emotion Prediction via NRC Lexicon
# ============================================================

# Compact NRC-style keyword table (subset sufficient for dream analysis)
_NRC_LEXICON: Dict[str, List[str]] = {
    # word -> list of NRC emotions it carries
    "fear":        ["fear", "negative"],
    "scared":      ["fear", "negative"],
    "afraid":      ["fear", "negative"],
    "terror":      ["fear", "negative"],
    "panic":       ["fear", "negative"],
    "nightmare":   ["fear", "negative"],
    "horror":      ["fear", "disgust", "negative"],
    "shame":       ["sadness", "negative"],
    "embarrassed": ["sadness", "negative"],
    "embarrass":   ["sadness", "negative"],
    "ashamed":     ["sadness", "negative"],
    "humiliated":  ["sadness", "anger", "negative"],
    "guilt":       ["sadness", "negative"],
    "angry":       ["anger", "negative"],
    "anger":       ["anger", "negative"],
    "rage":        ["anger", "negative"],
    "furious":     ["anger", "negative"],
    "sad":         ["sadness", "negative"],
    "sadness":     ["sadness", "negative"],
    "crying":      ["sadness", "negative"],
    "cry":         ["sadness", "negative"],
    "tears":       ["sadness", "negative"],
    "grief":       ["sadness", "negative"],
    "happy":       ["joy", "positive"],
    "happiness":   ["joy", "positive"],
    "joy":         ["joy", "positive"],
    "excited":     ["joy", "anticipation", "positive", "anticipation"], # duplicate cleaned below 
    "laugh":       ["joy", "positive"],
    "laughed":     ["joy", "positive"],  # context-dependent; overridden by shame cues
    "smile":       ["joy", "positive"],
    "love":        ["joy", "trust", "positive"],
    "trust":       ["trust", "positive"],
    "safe":        ["trust", "positive"],
    "peace":       ["trust", "positive"],
    "calm":        ["trust", "positive"],
    "surprise":    ["surprise"],
    "shocked":     ["surprise", "negative"],
    "suddenly":    ["surprise"],
    "unexpected":  ["surprise"],
    "disgust":     ["disgust", "negative"],
    "disgusting":  ["disgust", "negative"],
    "gross":       ["disgust", "negative"],
    "blood":       ["fear", "disgust", "negative"],
    "teeth":       ["fear", "negative"],
    "falling":     ["fear", "negative"],
    "fell":        ["fear", "negative"],
    "fall":        ["fear", "negative"],
    "flying":      ["anticipation", "joy", "positive"],
    "chase":       ["fear", "negative"],
    "chased":      ["fear", "negative"],
    "trapped":     ["fear", "sadness", "negative"],
    "lost":        ["sadness", "fear", "negative"],
    "dark":        ["fear", "negative"],
    "darkness":    ["fear", "negative"],
    "death":       ["fear", "sadness", "negative"],
    "dead":        ["fear", "sadness", "negative"],
    "monster":     ["fear", "negative"],
    "run":         ["fear", "anticipation"],
    "ran":         ["fear", "anticipation"],
    "hope":        ["anticipation", "trust", "positive"],
    "waiting":     ["anticipation"],
    "water":       ["fear", "anticipation"],
    "mirror":      ["surprise", "fear"],
    "shatter":     ["fear", "surprise", "negative"],
    "broken":      ["sadness", "negative"],
}

# deduplicate 'excited' list just in case
_NRC_LEXICON["excited"] = ["joy", "anticipation", "positive"]

_NRC_EMOTIONS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust",
    "negative", "positive",
]

def step5_emotion_prediction(preprocessed: dict) -> dict:
    """
    Score each segment against the NRC lexicon and return per-segment
    emotion probability vectors (normalised hit counts).
    """
    segments = preprocessed["segments"]
    lemmas_set = set(preprocessed["lemmas"])

    segment_emotions: Dict[str, Dict[str, float]] = {}

    for seg in segments:
        seg_tokens = set(re.findall(r"\b\w+\b", seg.lower()))
        counts = Counter({e: 0 for e in _NRC_EMOTIONS})

        for tok in seg_tokens:
            if tok in _NRC_LEXICON:
                for emo in _NRC_LEXICON[tok]:
                    counts[emo] += 1

        # Heuristic boost: "laughed at me" in presence of embarrassment → shame proxy
        if ("laughed" in seg_tokens or "laugh" in seg_tokens) and (
            "embarrassed" in lemmas_set or "ashamed" in lemmas_set
            or "shame" in lemmas_set or "embarrass" in lemmas_set
        ):
            counts["sadness"] += 2  # shame mapped to sadness in NRC

        total_hits = sum(counts.values())
        if total_hits == 0:
            # Uniform low prior
            vec = {e: round(1 / len(_NRC_EMOTIONS), 4) for e in _NRC_EMOTIONS}
        else:
            vec = {e: round(counts[e] / total_hits, 4) for e in _NRC_EMOTIONS}

        segment_emotions[seg] = vec

    # Also compute a document-level aggregate
    agg = {e: 0.0 for e in _NRC_EMOTIONS}
    for vec in segment_emotions.values():
        for e, v in vec.items():
            agg[e] += v
    n = max(len(segment_emotions), 1)
    agg = {e: round(v / n, 4) for e, v in agg.items()}

    return {
        "segment_emotions": segment_emotions,
        "document_emotion_vector": agg,
    }


# ============================================================
# STEP 6 — Enriched Embedding Construction
# ============================================================

def step6_enrich_embeddings(bilstm_out: dict, emotion_out: dict) -> dict:
    """
    Concatenate contextual embedding + entity presence vector +
    SRL verb signature + weighted emotion signal.
    Returns a flat dict representing the enriched feature vector.
    """
    doc_vec = emotion_out["document_emotion_vector"]
    enriched = {}
    enriched.update(bilstm_out["contextual_embedding"])
    enriched.update({f"entity_{k}": v for k, v in bilstm_out["entity_presence_vector"].items()})
    enriched["srl_verb"] = bilstm_out["srl_verb_signature"]
    # Weighted emotion signal (weight = 0.5 as per paper intent)
    enriched.update({f"emo_{k}": round(0.5 * v, 4) for k, v in doc_vec.items()})
    return enriched


# ============================================================
# STEP 7 — Unsupervised Theme Discovery (rule-based assignment)
# ============================================================

# Predefined thematic clusters (replaces UMAP + HDBSCAN for single-dream mode)
_THEME_PROFILES: List[Dict] = [
    {
        "name": "Body Integrity Disturbance",
        "keywords": {"teeth", "tooth", "mirror", "blood", "face", "falling", "shatter", "body", "broken", "nails", "hair"},
    },
    {
        "name": "Pursuit and Escape",
        "keywords": {"chase", "chased", "run", "ran", "escape", "trapped", "hiding", "monster", "darkness", "door"},
    },
    {
        "name": "Flying and Freedom",
        "keywords": {"fly", "flew", "flying", "soar", "float", "sky", "cloud", "wind", "freedom", "high"},
    },
    {
        "name": "Social Anxiety and Exposure",
        "keywords": {"embarrassed", "shame", "laugh", "laughed", "crowd", "naked", "humiliation", "judge", "people", "public"},
    },
    {
        "name": "Loss and Grief",
        "keywords": {"dead", "death", "died", "funeral", "lost", "gone", "crying", "tears", "grief", "miss"},
    },
    {
        "name": "Water and Submersion",
        "keywords": {"water", "ocean", "sea", "drown", "flood", "wave", "swimming", "river", "sink", "rain"},
    },
    {
        "name": "Examination and Performance",
        "keywords": {"exam", "test", "school", "class", "fail", "unprepared", "teacher", "late", "grade", "forgot"},
    },
    {
        "name": "Unknown / Uncategorized",
        "keywords": set(),
    },
]

def step7_assign_cluster(preprocessed: dict) -> str:
    """
    Assign the dream to the best-matching thematic cluster by counting
    keyword overlaps with the lemma/token set.
    """
    token_set = set(preprocessed["tokens"] + preprocessed["lemmas"])
    scores = []
    for profile in _THEME_PROFILES:
        score = len(token_set & profile["keywords"])
        scores.append((score, profile["name"]))

    # Pick highest; fall back to last (Unknown) if all zero
    scores.sort(reverse=True)
    best_score, best_name = scores[0]
    if best_score == 0:
        return "Unknown / Uncategorized"
    return best_name


# ============================================================
# STEP 8 — Keyword Extraction (c-TF-IDF proxy)
# ============================================================

_STOPWORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "them", "their", "this", "that", "the", "a", "an", "and", "or", "but",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "about",
    "into", "through", "during", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "very", "just", "more",
    "also", "then", "so", "when", "there", "here", "not", "no", "if", "as",
    "its", "out", "what", "which", "who", "how", "all", "each", "any",
    "suddenly", "started", "felt", "looked",
}

def step8_extract_keywords(preprocessed: dict, top_n: int = 6) -> List[str]:
    """
    Extract top-N content-bearing lemmas as cluster keywords.
    """
    freq = Counter(preprocessed["lemmas"])
    keywords = [
        word for word, _ in freq.most_common(50)
        if word not in _STOPWORDS and len(word) > 2
    ]
    return keywords[:top_n]


# ============================================================
# STEP 9 — Topic Label Generation (T5 proxy via keyword-to-label map)
# ============================================================

def step9_generate_topic_label(cluster_name: str, keywords: List[str]) -> str:
    """
    In production: T5 summarisation + cosine-similarity consolidation.
    Here: return the cluster name directly (already human-readable) and
    append keyword flavour if uncategorized.
    """
    if cluster_name != "Unknown / Uncategorized":
        return cluster_name
    # Fallback: capitalise top keywords into a label
    label = " + ".join(k.capitalize() for k in keywords[:3]) if keywords else "Dream Fragment"
    return label


# ============================================================
# STEP 10 — Dominant Emotion Computation
# ============================================================

def step10_dominant_emotion(emotion_out: dict) -> Tuple[str, Dict[str, float]]:
    """
    Average segment-level NRC vectors, return (dominant_emotion_name, avg_vector).
    Handles zero-vectors gracefully.
    """
    segment_vectors = list(emotion_out["segment_emotions"].values())

    if not segment_vectors:
        return "unknown", {e: 0.0 for e in _NRC_EMOTIONS}

    # Average across segments
    avg = {e: 0.0 for e in _NRC_EMOTIONS}
    for vec in segment_vectors:
        for e, v in vec.items():
            avg[e] += v
    n = len(segment_vectors)
    avg = {e: round(v / n, 4) for e, v in avg.items()}

    # Exclude valence dimensions (positive/negative) when picking emotion name
    core_emotions = {k: v for k, v in avg.items() if k not in ("positive", "negative")}

    if all(v == 0 for v in core_emotions.values()):
        dominant = max(avg, key=avg.get)
    else:
        dominant = max(core_emotions, key=core_emotions.get)

    return dominant, avg


# ============================================================
# MAIN PIPELINE — orchestrates Steps 1-10
# ============================================================

def run_pipeline(raw_dream: str) -> dict:
    """
    Run the full Dreamscape Mapper pipeline on a single dream string.
    Returns the structured thematic-emotional JSON record.
    """
    # Step 1
    dream = step1_ingest(raw_dream)

    # Step 2
    preprocessed = step2_preprocess(dream)

    # Step 3
    annotations = step3_silver_annotate(preprocessed)

    # Step 4
    bilstm_out = step4_bilstm_representations(preprocessed, annotations)

    # Step 5
    emotion_out = step5_emotion_prediction(preprocessed)

    # Step 6
    enriched = step6_enrich_embeddings(bilstm_out, emotion_out)

    # Step 7
    cluster_name = step7_assign_cluster(preprocessed)

    # Step 8
    keywords = step8_extract_keywords(preprocessed)

    # Step 9
    topic_label = step9_generate_topic_label(cluster_name, keywords)

    # Step 10
    dominant_emotion, emotion_vector = step10_dominant_emotion(emotion_out)

    # --- Assemble Key Entities ---
    entities = annotations["entities"]
    key_entities = (
        entities["Character"]
        + entities["Body-Object"]
        + entities["Setting-Object"]
    )
    key_entities = list(dict.fromkeys(key_entities))  # deduplicate, preserve order

    # --- Build final structured output ---
    result = {
        "Topic_Cluster": topic_label,
        "Dominant_Emotion": dominant_emotion.capitalize(),
        "Key_Entities": key_entities if key_entities else ["Unknown"],
        "Semantic_Relation": annotations["srl"],
        "Emotion_Vector": emotion_vector,
        "Cluster_Keywords": keywords,
        "Coreference_Map": annotations["coreference"],
        "Global_Stat": (
            f"Dominant theme '{topic_label}' with emotion '{dominant_emotion}' "
            f"detected. Top emotion score: {emotion_vector.get(dominant_emotion, 0.0):.2f}."
        ),
    }

    return result


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dream_input = " ".join(sys.argv[1:])
    else:
        dream_input = input("Enter dream: ").strip()

    output = run_pipeline(dream_input)
    print(json.dumps(output, indent=2))