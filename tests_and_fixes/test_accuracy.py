# =============================================================
# ACCURACY TEST — Dreamscape NLP Pipeline
# =============================================================
# Evaluates quantitative accuracy for:
#   1. NER  — token-level accuracy, span-level Precision/Recall/F1
#   2. SRL  — token-level accuracy, per-role breakdown
#   3. NRC Emotion — coverage, lexicon hit-rate, top-emotion consistency
#
# Run:  python test_accuracy.py
# =============================================================

import json
import pickle
import re
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, random_split

from step4_combined import (
    DreamscapeMultiTaskBiLSTM,
    DreamSegmentDataset,
    Vocab,
    NRC_EMOTIONS,
    build_vocabs,
    align_ner_labels,
    align_srl_labels,
    collate_fn,
    simple_tokenize,
)

try:
    from nrclex import NRCLex
    _NRC_AVAILABLE = True
except ImportError:
    _NRC_AVAILABLE = False
    print("  [warning] nrclex not installed — emotion accuracy checks skipped")


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def _section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _pass(msg):  print(f"  ✓  {msg}")
def _warn(msg):  print(f"  ⚠  {msg}")
def _fail(msg):  print(f"  ✗  {msg}")


# ════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD MODEL + VOCABS
# ════════════════════════════════════════════════════════════════

def load_model_and_vocabs(device):
    """Load the trained BiLSTM and its vocabularies."""
    try:
        with open("data_models/step4_vocabs.pkl", "rb") as f:
            vocabs = pickle.load(f)
        token_vocab = vocabs["token"]
        ner_vocab   = vocabs["ner"]
        srl_vocab   = vocabs["srl"]
    except FileNotFoundError:
        raise FileNotFoundError(
            "step4_vocabs.pkl not found — run step4_combined.py first."
        )

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
        raise FileNotFoundError(
            "step4_bilstm.pt not found — run step4_combined.py first."
        )

    return model, token_vocab, ner_vocab, srl_vocab


# ════════════════════════════════════════════════════════════════
# SECTION 2 — TOKEN-LEVEL ACCURACY
# ════════════════════════════════════════════════════════════════

def token_accuracy(model, loader, ner_vocab, srl_vocab, device):
    """
    Returns:
        ner_acc  (float) — token-level NER label accuracy (ignoring PAD)
        srl_acc  (float) — token-level SRL label accuracy (ignoring PAD)
    """
    ner_correct = ner_total = 0
    srl_correct = srl_total = 0

    with torch.no_grad():
        for token_ids, ner_labels, srl_labels in loader:
            token_ids  = token_ids.to(device)
            ner_labels = ner_labels.to(device)
            srl_labels = srl_labels.to(device)

            outputs   = model(token_ids)
            ner_preds = outputs["ner_preds"]   # list of lists (CRF decode)
            srl_preds = outputs["srl_preds"]   # Tensor (B, T)

            mask = (token_ids != 0)

            # NER — CRF decode returns list[list[int]]
            for b_idx, pred_seq in enumerate(ner_preds):
                length = mask[b_idx].sum().item()
                gold   = ner_labels[b_idx, :length].cpu().tolist()
                pred   = pred_seq[:length]
                ner_correct += sum(g == p for g, p in zip(gold, pred))
                ner_total   += length

            # SRL — argmax tensor
            for b_idx in range(srl_labels.size(0)):
                length = mask[b_idx].sum().item()
                gold   = srl_labels[b_idx, :length].cpu().tolist()
                pred   = srl_preds[b_idx, :length].cpu().tolist()
                srl_correct += sum(g == p for g, p in zip(gold, pred))
                srl_total   += length

    ner_acc = ner_correct / ner_total if ner_total else 0.0
    srl_acc = srl_correct / srl_total if srl_total else 0.0
    return ner_acc, srl_acc


# ════════════════════════════════════════════════════════════════
# SECTION 3 — SPAN-LEVEL NER  (Precision / Recall / F1)
# ════════════════════════════════════════════════════════════════

def extract_spans(label_seq):
    """
    Convert a BIO label list (strings) into a set of
    (start, end, entity_type) tuples.
    """
    spans = set()
    i = 0
    while i < len(label_seq):
        lbl = label_seq[i]
        if lbl.startswith("B-"):
            etype = lbl[2:]
            start = i
            j = i + 1
            while j < len(label_seq) and label_seq[j] == f"I-{etype}":
                j += 1
            spans.add((start, j, etype))
            i = j
        else:
            i += 1
    return spans


def span_f1(model, loader, ner_vocab, device):
    """
    Computes span-level NER Precision, Recall, F1 and a per-type breakdown.
    """
    tp_total = fp_total = fn_total = 0
    type_tp  = defaultdict(int)
    type_fp  = defaultdict(int)
    type_fn  = defaultdict(int)

    with torch.no_grad():
        for token_ids, ner_labels, _ in loader:
            token_ids  = token_ids.to(device)
            ner_labels = ner_labels.to(device)
            mask       = (token_ids != 0)

            outputs   = model(token_ids)
            ner_preds = outputs["ner_preds"]

            for b_idx, pred_seq in enumerate(ner_preds):
                length = mask[b_idx].sum().item()
                gold_ids = ner_labels[b_idx, :length].cpu().tolist()
                pred_ids = pred_seq[:length]

                gold_lbls = [ner_vocab.idx2token[i] for i in gold_ids]
                pred_lbls = [ner_vocab.idx2token[i] for i in pred_ids]

                gold_spans = extract_spans(gold_lbls)
                pred_spans = extract_spans(pred_lbls)

                for span in pred_spans:
                    if span in gold_spans:
                        tp_total += 1
                        type_tp[span[2]] += 1
                    else:
                        fp_total += 1
                        type_fp[span[2]] += 1
                for span in gold_spans:
                    if span not in pred_spans:
                        fn_total += 1
                        type_fn[span[2]] += 1

    prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    rec  = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    f1   = 2 * prec * rec / (prec + rec)   if (prec + rec)          else 0.0

    # Per-type F1
    all_types = sorted(set(list(type_tp) + list(type_fp) + list(type_fn)))
    per_type  = {}
    for t in all_types:
        tp = type_tp[t]; fp = type_fp[t]; fn = type_fn[t]
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f  = 2 * p * r / (p + r) if (p + r) else 0.0
        per_type[t] = {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}

    return prec, rec, f1, per_type


# ════════════════════════════════════════════════════════════════
# SECTION 4 — SRL ROLE BREAKDOWN
# ════════════════════════════════════════════════════════════════

def srl_role_accuracy(model, loader, srl_vocab, device):
    """
    Per-role SRL accuracy breakdown (ignores PAD / O tokens).
    """
    role_correct = defaultdict(int)
    role_total   = defaultdict(int)

    pad_idx = srl_vocab.token2idx.get("<PAD>", 0)
    o_idx   = srl_vocab.token2idx.get("O",    0)

    with torch.no_grad():
        for token_ids, _, srl_labels in loader:
            token_ids  = token_ids.to(device)
            srl_labels = srl_labels.to(device)
            mask       = (token_ids != 0)

            outputs   = model(token_ids)
            srl_preds = outputs["srl_preds"]

            for b_idx in range(srl_labels.size(0)):
                length = mask[b_idx].sum().item()
                gold   = srl_labels[b_idx, :length].cpu().tolist()
                pred   = srl_preds[b_idx, :length].cpu().tolist()
                for g, p in zip(gold, pred):
                    if g in (pad_idx,):
                        continue   # skip PAD
                    role_name = srl_vocab.idx2token.get(g, "?")
                    role_correct[role_name] += int(g == p)
                    role_total[role_name]   += 1

    return {
        role: {
            "correct": role_correct[role],
            "total":   role_total[role],
            "accuracy": role_correct[role] / role_total[role] if role_total[role] else 0.0,
        }
        for role in sorted(role_total)
    }


# ════════════════════════════════════════════════════════════════
# SECTION 5 — NRC EMOTION ACCURACY
# ════════════════════════════════════════════════════════════════

def emotion_accuracy(annotations_path="jsons/dream_annotations.json", sample_n=500):
    """
    Checks NRC emotion label consistency:
      - Lexicon hit-rate (% of segments where at least 1 NRC word is found)
      - Dominant emotion agreement between annotation and live NRC call
      - Reports per-emotion coverage
    """
    if not _NRC_AVAILABLE:
        return None

    with open(annotations_path, "r") as f:
        data = json.load(f)

    # Sample for speed
    import random
    random.seed(42)
    sample = random.sample(data, min(sample_n, len(data)))

    hit         = 0        # NRC found ≥1 emotion word
    agree       = 0        # dominant emotion matches annotation
    total       = 0
    emo_counts  = defaultdict(int)  # which emotion dominated in annotation
    agree_per   = defaultdict(int)
    total_per   = defaultdict(int)

    for seg in sample:
        tokens = seg.get("tokens", [])
        if not tokens:
            continue
        text = " ".join(tokens)
        ann_vec = seg.get("emotion_vector", {})
        if not ann_vec:
            continue

        # nrc = NRCLex(text)
        # raw = nrc.raw_emotion_scores
        # live_scores = {emo: raw.get(emo, 0) for emo in NRC_EMOTIONS}
        nrc = NRCLex(text)
        # Newer versions of NRCLex use 'affect_frequencies' or 'raw_emotion_scores'
        # but 'affect_frequencies' is the most reliable cross-version attribute.
        raw = getattr(nrc, 'raw_emotion_scores', nrc.affect_frequencies)

        # Ensure we handle cases where 'raw' might be None or differently structured
        live_scores = {emo: raw.get(emo, 0) if raw else 0 for emo in NRC_EMOTIONS}

        # Hit-rate
        if sum(live_scores.values()) > 0:
            hit += 1

        # Dominant emotion in annotation
        ann_dom = max(ann_vec, key=lambda e: ann_vec.get(e, 0))
        # Dominant emotion from live NRC call
        if sum(live_scores.values()) > 0:
            live_dom = max(live_scores, key=live_scores.get)
        else:
            live_dom = None

        emo_counts[ann_dom] += 1
        total_per[ann_dom]  += 1
        if live_dom == ann_dom:
            agree += 1
            agree_per[ann_dom] += 1
        total += 1

    return {
        "sample_size"     : total,
        "lexicon_hit_rate": hit / total if total else 0.0,
        "dominant_agree"  : agree / total if total else 0.0,
        "per_emotion"     : {
            emo: {
                "count":    emo_counts[emo],
                "agree":    agree_per[emo],
                "accuracy": agree_per[emo] / total_per[emo] if total_per[emo] else 0.0,
            }
            for emo in NRC_EMOTIONS
        },
    }


# ════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN
# ════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  ACCURACY TEST — Dreamscape NLP Pipeline")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # ── Load model ────────────────────────────────────────────
    _section("Loading Model + Vocabs")
    model, token_vocab, ner_vocab, srl_vocab = load_model_and_vocabs(device)
    _pass("Model loaded from step4_bilstm.pt")
    _pass(f"Vocab sizes — token: {len(token_vocab)}, "
          f"NER: {len(ner_vocab)}, SRL: {len(srl_vocab)}")

    # ── Build dataset & validation split ─────────────────────
    _section("Building Dataset (10% validation split)")
    DATA_PATH = "jsons/dream_annotations.json"
    with open(DATA_PATH, "r") as f:
        annotations = json.load(f)

    dataset    = DreamSegmentDataset(annotations, token_vocab, ner_vocab, srl_vocab)
    val_size   = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    _, val_ds  = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    _pass(f"Validation set: {val_size:,} samples  |  "
          f"Training set: {train_size:,} samples")

    # ────────────────────────────────────────────────────────
    # CHECK 1 — TOKEN-LEVEL ACCURACY
    # ────────────────────────────────────────────────────────
    _section("CHECK 1 — Token-Level Accuracy (Validation Set)")
    ner_acc, srl_acc = token_accuracy(model, val_loader, ner_vocab, srl_vocab, device)

    ner_icon = "✓" if ner_acc >= 0.70 else ("⚠" if ner_acc >= 0.50 else "✗")
    srl_icon = "✓" if srl_acc >= 0.70 else ("⚠" if srl_acc >= 0.50 else "✗")
    print(f"  {ner_icon}  NER Token Accuracy : {ner_acc*100:6.2f}%")
    print(f"  {srl_icon}  SRL Token Accuracy : {srl_acc*100:6.2f}%")

    if ner_acc < 0.50:
        _warn("NER accuracy below 50% — model may need more epochs or tuning.")
    if srl_acc < 0.50:
        _warn("SRL accuracy below 50% — model may need more epochs or tuning.")

    # ────────────────────────────────────────────────────────
    # CHECK 2 — SPAN-LEVEL NER F1
    # ────────────────────────────────────────────────────────
    _section("CHECK 2 — Span-Level NER  (Precision / Recall / F1)")
    prec, rec, f1, per_type = span_f1(model, val_loader, ner_vocab, device)

    f1_icon = "✓" if f1 >= 0.60 else ("⚠" if f1 >= 0.35 else "✗")
    print(f"  {f1_icon}  Overall  → P: {prec*100:.2f}%  R: {rec*100:.2f}%  F1: {f1*100:.2f}%\n")

    if per_type:
        print(f"  {'ENTITY TYPE':<14} | {'PREC':>7} | {'REC':>7} | {'F1':>7} | TP / FP / FN")
        print("  " + "-" * 62)
        for etype, m in sorted(per_type.items(), key=lambda x: -x[1]["f1"]):
            icon = "✓" if m["f1"] >= 0.60 else ("⚠" if m["f1"] >= 0.30 else "✗")
            print(f"  {icon} {etype:<12} | {m['precision']*100:6.2f}% | "
                  f"{m['recall']*100:6.2f}% | {m['f1']*100:6.2f}% | "
                  f"{m['tp']} / {m['fp']} / {m['fn']}")

    # ────────────────────────────────────────────────────────
    # CHECK 3 — SRL ROLE BREAKDOWN
    # ────────────────────────────────────────────────────────
    _section("CHECK 3 — SRL Role Accuracy Breakdown")
    role_stats = srl_role_accuracy(model, val_loader, srl_vocab, device)

    print(f"  {'ROLE':<14} | {'CORRECT':>8} | {'TOTAL':>8} | {'ACCURACY':>9}")
    print("  " + "-" * 50)
    for role, stats in sorted(role_stats.items(), key=lambda x: -x[1]["accuracy"]):
        acc  = stats["accuracy"]
        icon = "✓" if acc >= 0.70 else ("⚠" if acc >= 0.40 else "✗")
        print(f"  {icon} {role:<12} | {stats['correct']:>8} | "
              f"{stats['total']:>8} | {acc*100:8.2f}%")

    # ────────────────────────────────────────────────────────
    # CHECK 4 — NRC EMOTION ACCURACY
    # ────────────────────────────────────────────────────────
    _section("CHECK 4 — NRC Emotion Accuracy (500-sample spot check)")
    emo_stats = emotion_accuracy()

    if emo_stats is not None:
        hit_icon = "✓" if emo_stats["lexicon_hit_rate"] >= 0.70 else "⚠"
        agr_icon = "✓" if emo_stats["dominant_agree"]   >= 0.50 else "⚠"
        print(f"  {hit_icon}  Lexicon Hit-Rate     : "
              f"{emo_stats['lexicon_hit_rate']*100:.2f}%  "
              f"(% of segments with ≥1 NRC word)")
        print(f"  {agr_icon}  Dominant Emo. Agree  : "
              f"{emo_stats['dominant_agree']*100:.2f}%  "
              f"(live NRC matches stored annotation)")
        print(f"\n  {'EMOTION':<14} | {'COUNT':>6} | {'AGREE':>6} | {'ACCURACY':>9}")
        print("  " + "-" * 45)
        for emo, s in emo_stats["per_emotion"].items():
            icon = "✓" if s["accuracy"] >= 0.50 else "⚠"
            print(f"  {icon} {emo:<12} | {s['count']:>6} | "
                  f"{s['agree']:>6} | {s['accuracy']*100:8.2f}%")
    else:
        _warn("NRC emotion checks skipped (nrclex not installed).")

    # ────────────────────────────────────────────────────────
    # FINAL VERDICT
    # ────────────────────────────────────────────────────────
    _section("FINAL VERDICT")
    issues = []
    if ner_acc  < 0.50: issues.append(f"NER token accuracy too low ({ner_acc*100:.1f}%)")
    if srl_acc  < 0.50: issues.append(f"SRL token accuracy too low ({srl_acc*100:.1f}%)")
    if f1       < 0.35: issues.append(f"NER span F1 too low ({f1*100:.1f}%)")
    if emo_stats and emo_stats["lexicon_hit_rate"] < 0.50:
        issues.append("NRC lexicon hit-rate below 50%")

    if issues:
        print("  Issues found:")
        for iss in issues:
            _fail(iss)
        print("\n  Review model training and/or data quality before deployment.")
    else:
        _pass("All accuracy checks passed!")
        _pass("Pipeline is ready for inference.")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  NER Token Accuracy   : {ner_acc*100:6.2f}%")
    print(f"  SRL Token Accuracy   : {srl_acc*100:6.2f}%")
    print(f"  NER Span F1          : {f1*100:6.2f}%  "
          f"(P {prec*100:.1f}% / R {rec*100:.1f}%)")
    if emo_stats:
        print(f"  NRC Lexicon Hit-Rate : {emo_stats['lexicon_hit_rate']*100:6.2f}%")
        print(f"  NRC Dominant Agree   : {emo_stats['dominant_agree']*100:6.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
