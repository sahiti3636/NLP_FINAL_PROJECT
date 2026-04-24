# =============================================================
# TEST SCRIPT — Step 9 Output Verification (T5 Topic Labels)
# =============================================================
# Run: python test_step9.py
# =============================================================

import json
from collections import Counter

print("Loading step9_results.json...")
with open("jsons/step9_results.json", "r") as f:
    results = json.load(f)
print(f"  Loaded {len(results)} cluster entries.\n")


# ════════════════════════════════════════════════════════════════
# CHECK 1 — STRUCTURE
# ════════════════════════════════════════════════════════════════

print("=" * 55)
print("CHECK 1 — Structure & Required Keys")
print("=" * 55)

required = {"cluster_id", "size", "keywords", "dominant_emotion", "topic_label"}
missing_keys  = []
blank_labels  = []
very_short    = []

for entry in results:
    missing = required - set(entry.keys())
    if missing:
        missing_keys.append((entry.get("cluster_id"), missing))
    label = str(entry.get("topic_label", "")).strip()
    if not label or label.lower() in ("null", "none", ""):
        blank_labels.append(entry.get("cluster_id"))
    elif len(label.split()) < 2:
        very_short.append((entry.get("cluster_id"), label))

if missing_keys:
    print(f"  MISSING KEYS in {len(missing_keys)} entries:")
    for cid, mk in missing_keys[:5]:
        print(f"    Cluster {cid}: {mk}")
else:
    print(f"  All required keys present.")

print(f"  Blank topic_labels    : {len(blank_labels)}")
print(f"  Single-word labels    : {len(very_short)}")
if blank_labels:
    print(f"  Blank cluster IDs     : {blank_labels[:10]}")
if very_short:
    print(f"  Single-word examples  : {very_short[:5]}")


# ════════════════════════════════════════════════════════════════
# CHECK 2 — LABEL UNIQUENESS
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("CHECK 2 — Label Uniqueness & Diversity")
print("=" * 55)

all_labels   = [str(e.get("topic_label", "")).strip() for e in results]
label_counts = Counter(all_labels)
duplicates   = {k: v for k, v in label_counts.items() if v > 1}
unique_count = len(label_counts)

print(f"  Total labels          : {len(all_labels)}")
print(f"  Unique labels         : {unique_count}")
print(f"  Duplicate labels      : {len(duplicates)}")

if duplicates:
    print(f"\n  Most repeated labels:")
    for label, count in sorted(duplicates.items(), key=lambda x: -x[1])[:8]:
        print(f"    '{label}'  ×{count}")

if unique_count < len(all_labels) * 0.7:
    print("\n  WARNING: Too many duplicate labels — "
          "T5 may be generating generic phrases.")
else:
    print("\n  Label diversity acceptable.")


# ════════════════════════════════════════════════════════════════
# CHECK 3 — LABEL QUALITY SPOT CHECK
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("CHECK 3 — Label Quality (top 10 clusters by size)")
print("=" * 55)

sorted_results = sorted(
    [r for r in results if r.get("cluster_id") != -1],
    key=lambda x: -x.get("size", 0)
)

print(f"  {'CLUSTER':>8} | {'SIZE':>7} | {'EMOTION':<14} | "
      f"{'TOPIC LABEL':<35} | KEYWORDS")
print("  " + "-" * 110)

for entry in sorted_results[:10]:
    cid   = entry.get("cluster_id", "?")
    size  = entry.get("size", 0)
    emo   = entry.get("dominant_emotion", "?")[:12]
    label = str(entry.get("topic_label", ""))[:33]
    kws   = ", ".join(entry.get("keywords", [])[:4])
    print(f"  {cid:>8} | {size:>7} | {emo:<14} | {label:<35} | {kws}")


# ════════════════════════════════════════════════════════════════
# CHECK 4 — LABEL-KEYWORD ALIGNMENT
# Checks whether the generated label semantically relates to
# its cluster keywords — flags obvious mismatches.
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("CHECK 4 — Label-Keyword Overlap (semantic alignment)")
print("=" * 55)

# Simple heuristic: at least one keyword root should appear
# somewhere in the label, OR the label should be longer than
# 2 words (indicating T5 generated a real phrase not a default).
weak_alignments = []
for entry in results:
    label = str(entry.get("topic_label", "")).lower().strip()
    kws   = [k.lower() for k in entry.get("keywords", [])]
    if not label or len(label.split()) < 2:
        continue
    # Check if any keyword stem appears in label
    overlap = any(
        kw[:5] in label or label[:5] in kw
        for kw in kws if len(kw) > 4
    )
    if not overlap:
        weak_alignments.append({
            "cluster_id": entry.get("cluster_id"),
            "label":      entry.get("topic_label"),
            "keywords":   kws[:5],
        })

print(f"  Labels with no keyword overlap : {len(weak_alignments)} / {len(results)}")
if weak_alignments:
    print(f"\n  Examples (may still be valid — T5 can paraphrase):")
    for wa in weak_alignments[:5]:
        print(f"    Cluster {wa['cluster_id']}: '{wa['label']}'")
        print(f"      Keywords: {wa['keywords']}")


# ════════════════════════════════════════════════════════════════
# CHECK 5 — EMOTION DISTRIBUTION ACROSS CLUSTERS
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("CHECK 5 — Emotion Distribution Across Clusters")
print("=" * 55)

emo_counts = Counter(
    e.get("dominant_emotion", "unknown")
    for e in results
    if e.get("cluster_id") != -1
)
total = sum(emo_counts.values())
print(f"  {'EMOTION':<16} | COUNT | PERCENTAGE")
print("  " + "-" * 38)
for emo, count in sorted(emo_counts.items(), key=lambda x: -x[1]):
    pct = 100 * count / total
    bar = "█" * int(pct / 2)
    print(f"  {emo:<16} | {count:>5} | {pct:>5.1f}%  {bar}")

# Warn if one emotion dominates everything
top_emo, top_count = emo_counts.most_common(1)[0]
if top_count / total > 0.6:
    print(f"\n  WARNING: '{top_emo}' dominates {top_count/total*100:.1f}% "
          f"of clusters — emotion aggregation may still need tuning.")
else:
    print(f"\n  Emotion distribution looks varied.")


# ════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("VERDICT")
print("=" * 55)

issues = []
if missing_keys:
    issues.append(f"{len(missing_keys)} entries have missing keys")
if blank_labels:
    issues.append(f"{len(blank_labels)} blank topic labels — T5 didn't fill all")
if unique_count < len(all_labels) * 0.7:
    issues.append("Too many duplicate labels — low label diversity")
if top_count / total > 0.6:
    issues.append(f"Emotion '{top_emo}' dominates — check aggregation")

if issues:
    print("  ISSUES FOUND:")
    for iss in issues:
        print(f"    - {iss}")
    print("\n  Fix issues before proceeding to Step 10 (Knowledge Graph).")
else:
    print("  All checks passed. Ready for Step 10 (Knowledge Graph).")
