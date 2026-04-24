# =============================================================
# TEST SCRIPT — Step 4 (NER + SRL) + NRC Emotions
# =============================================================
# Run after training:  python3 test_step4.py
# =============================================================

import torch
import pickle
import re
from collections import defaultdict
from nrclex import NRCLex

from step4_combined import (
    DreamscapeMultiTaskBiLSTM,
    Vocab,
    NRC_EMOTIONS,
    simple_tokenize,
)


# ════════════════════════════════════════════════════════════════
# SECTION 1 — LOADERS
# ════════════════════════════════════════════════════════════════

def load_ner_srl_model(device):
    try:
        with open("data_models/step4_vocabs.pkl", "rb") as f:
            vocabs = pickle.load(f)
        token_vocab = vocabs["token"]
        ner_vocab   = vocabs["ner"]
        srl_vocab   = vocabs["srl"]
    except FileNotFoundError:
        print("Could not find step4_vocabs.pkl. Did Step 4 finish training?")
        return None, None, None, None

    model = DreamscapeMultiTaskBiLSTM(
        vocab_size     = len(token_vocab),
        num_ner_labels = len(ner_vocab),
        num_srl_labels = len(srl_vocab),
    ).to(device)

    try:
        model.load_state_dict(
            torch.load("data_models/step4_bilstm.pt", map_location=device, weights_only=True)
        )
        model.eval()
    except FileNotFoundError:
        print("Could not find step4_bilstm.pt.")
        return None, None, None, None

    return model, token_vocab, ner_vocab, srl_vocab


# ════════════════════════════════════════════════════════════════
# SECTION 2 — NRC EMOTION (replaces GoEmotions entirely)
# ════════════════════════════════════════════════════════════════

def get_nrc_emotion_vector(text: str) -> dict:
    """
    Returns normalised {emotion: score} for the 8 NRC emotions.
    No GPU needed — pure lexicon lookup.
    Falls back to uniform if no lexicon words are found.
    """
    nrc    = NRCLex(text)
    raw    = nrc.raw_emotion_scores
    scores = {emo: raw.get(emo, 0) for emo in NRC_EMOTIONS}
    total  = sum(scores.values())
    if total == 0:
        return {emo: round(1.0 / len(NRC_EMOTIONS), 4) for emo in NRC_EMOTIONS}
    return {emo: round(scores[emo] / total, 4) for emo in NRC_EMOTIONS}


# ════════════════════════════════════════════════════════════════
# SECTION 3 — DISPLAY HELPERS
# ════════════════════════════════════════════════════════════════

def group_ner_spans(tokens, ner_labels):
    """
    Merges consecutive B-X / I-X BIO tags into
    (span_string, entity_type) tuples.
    """
    entities = []
    i = 0
    while i < len(tokens):
        lbl = ner_labels[i]
        if lbl.startswith("B-"):
            etype = lbl[2:]
            span  = [tokens[i]]
            j = i + 1
            while j < len(tokens) and ner_labels[j] == f"I-{etype}":
                span.append(tokens[j])
                j += 1
            entities.append((" ".join(span), etype))
            i = j
        else:
            i += 1
    return entities


def group_srl_spans(tokens, srl_labels):
    """
    FIX — ISSUE 1 (display) + ISSUE 2 (grouping):
      Merges consecutive tokens sharing the same SRL role into a
      single span string, then associates each span with its
      nearest preceding verb using position-based scoping.

    Previous version printed one line per token
    (e.g. four ARGM-DIR lines instead of one phrase).
    """
    verb_positions = [i for i, r in enumerate(srl_labels) if r == "V"]

    def owning_verb_idx(tok_idx):
        owner = None
        for vp in verb_positions:
            if vp <= tok_idx:
                owner = vp
            else:
                break
        return owner

    verb_token_at = {vp: tokens[vp] for vp in verb_positions}
    verb_args     = defaultdict(list)

    i = 0
    while i < len(tokens):
        role = srl_labels[i]
        if role in ("O", "V", "<PAD>", "<UNK>"):
            i += 1
            continue

        # Collect all consecutive tokens with the same role
        span_tokens = [tokens[i]]
        j = i + 1
        while j < len(tokens) and srl_labels[j] == role:
            span_tokens.append(tokens[j])
            j += 1

        owner_pos = owning_verb_idx(i)
        if owner_pos is not None:
            verb_str = verb_token_at[owner_pos]
            verb_args[verb_str].append((role, " ".join(span_tokens)))

        i = j

    return verb_args


# ════════════════════════════════════════════════════════════════
# SECTION 4 — MAIN PREDICTION FUNCTION
# ════════════════════════════════════════════════════════════════

def predict(test_sentence: str, top_k_emotions: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading on: {device}")
    print("=" * 65)

    tokens = simple_tokenize(test_sentence)
    print(f"Tokens fed to model: {tokens}\n")

    # ════════════════════════════════════════
    # BLOCK 1 — NER + SRL
    # ════════════════════════════════════════
    ner_srl_model, token_vocab, ner_vocab, srl_vocab = load_ner_srl_model(device)

    if ner_srl_model is not None:
        token_ids = torch.tensor(
            [[token_vocab.encode(t) for t in tokens]], dtype=torch.long
        ).to(device)

        with torch.no_grad():
            outputs   = ner_srl_model(token_ids)
            ner_preds = outputs["ner_preds"][0]
            srl_preds = outputs["srl_preds"][0]

        ner_labels = (
            [ner_vocab.idx2token[idx] for idx in ner_preds]
            if isinstance(ner_preds, list)
            else [ner_vocab.idx2token[idx.item()] for idx in ner_preds]
        )
        srl_labels = [srl_vocab.idx2token[idx.item()] for idx in srl_preds]

        # ── Token-level table ─────────────────────────────────
        print(f"DREAM: '{test_sentence}'\n")
        print(f"{'TOKEN':<16} | {'NER':<16} | {'SRL ROLE'}")
        print("-" * 55)
        for tok, ner, srl in zip(tokens, ner_labels, srl_labels):
            ner_disp = "" if ner in ("O", "<PAD>", "<UNK>") else ner
            srl_disp = "" if srl in ("O", "<PAD>", "<UNK>") else srl
            print(f"{tok:<16} | {ner_disp:<16} | {srl_disp}")

        # ── NER entity summary ────────────────────────────────
        print("\nExtracted Entities:")
        entities = group_ner_spans(tokens, ner_labels)
        if entities:
            for span, etype in entities:
                print(f"  [{etype:<8}]  {span}")
        else:
            print("  (none detected)")

        # ── SRL argument summary (grouped spans) ──────────────
        print("\nSRL Argument Structure:")
        verb_args = group_srl_spans(tokens, srl_labels)
        if verb_args:
            for verb_str, arg_list in verb_args.items():
                parts    = [f"{role}: '{span}'" for role, span in arg_list]
                print(f"  VERB: '{verb_str}'  →  {',  '.join(parts)}")
        else:
            print("  (no predicates detected)")

    # ════════════════════════════════════════
    # BLOCK 2 — NRC EMOTION VECTOR
    # No model loading needed — pure lexicon
    # ════════════════════════════════════════
    emo_vec = get_nrc_emotion_vector(test_sentence)

    print(f"\nEmotion Vector  (NRC Lexicon — 8 emotions):")
    print("-" * 45)

    sorted_emo = sorted(emo_vec.items(), key=lambda x: -x[1])
    for rank, (label, score) in enumerate(sorted_emo, 1):
        bar = "█" * int(score * 30)
        print(f"  {rank}. {label:<14} {score:.4f}  {bar}")

    print("\n" + "=" * 65)


# ════════════════════════════════════════════════════════════════
# SECTION 5 — ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_sentences = [
        "My mother was laughing at me while my teeth shattered in the mirror.",
        "India is the country James lives in.",
        "A dark shadow chased me through an endless corridor.",
        "I was flying over the ocean and felt completely free.",
        "The ghost screamed and I ran out of the burning house.",
    ]

    for sentence in test_sentences:
        predict(sentence)
        print()
