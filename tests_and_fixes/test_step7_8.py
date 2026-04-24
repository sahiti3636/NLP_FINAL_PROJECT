# =============================================================
# TEST SCRIPT — Steps 7 & 8 Output Verification
# =============================================================
# Run: python test_step7_8.py
# =============================================================

import json
import numpy as np
from collections import Counter

# ════════════════════════════════════════════════════════════════
# LOAD ALL FILES
# ════════════════════════════════════════════════════════════════

print("Loading files...")

cluster_labels  = np.load("data_models/step7_cluster_labels.npy")
umap_reduced    = np.load("data_models/step7_umap_reduced.npy")

with open("jsons/step8_keywords.json", "r") as f:
    keywords = json.load(f)

with open("jsons/step7_8_results.json", "r") as f:
    results = json.load(f)

print("All files loaded.\n")

# ════════════════════════════════════════════════════════════════
# CHECK 1 — SHAPE CONSISTENCY
# ════════════════════════════════════════════════════════════════

print("=" * 55)
print("CHECK 1 — Shape & Size Consistency")
print("=" * 55)

N = len(cluster_labels)
print(f"  cluster_labels shape  : {cluster_labels.shape}")
print(f"  umap_reduced shape    : {umap_reduced.shape}")
print(f"  step7_8_results count : {len(results)} clusters")
print(f"  step8_keywords count  : {len(keywords)} clusters")

assert umap_reduced.shape[0] == N, \
    f"MISMATCH: umap rows {umap_reduced.shape[0]} != labels {N}"
print(f"\n  Segment count consistent across all files: {N}")

# ════════════════════════════════════════════════════════════════
# CHECK 2 — CLUSTER DISTRIBUTION
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("CHECK 2 — Cluster Distribution")
print("=" * 55)

label_counts  = Counter(cluster_labels.tolist())
num_clusters  = sum(1 for k in label_counts if k != -1)
noise_count   = label_counts.get(-1, 0)
noise_pct     = 100 * noise_count / N

print(f"  Total segments        : {N}")
print(f"  Discovered clusters   : {num_clusters}")
print(f"  Noise points (-1)     : {noise_count}  ({noise_pct:.1f}%)")

# Cluster sizes
sizes = sorted(
    [(k, v) for k, v in label_counts.items() if k != -1],
    key=lambda x: -x[1]
)
print(f"\n  Largest 5 clusters:")
for cid, size in sizes[:5]:
    print(f"    Cluster {cid:>4}  →  {size:>6} segments")
print(f"\n  Smallest 5 clusters:")
for cid, size in sizes[-5:]:
    print(f"    Cluster {cid:>4}  →  {size:>6} segments")

# Warn if noise is too high
if noise_pct > 30:
    print(f"\n  WARNING: {noise_pct:.1f}% noise — consider lowering "
          f"HDBSCAN min_cluster_size")
else:
    print(f"\n  Noise level acceptable.")

# ════════════════════════════════════════════════════════════════
# CHECK 3 — UMAP SANITY
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("CHECK 3 — UMAP Reduced Vectors")
print("=" * 55)

print(f"  UMAP dimensions       : {umap_reduced.shape[1]}")
print(f"  Value range           : [{umap_reduced.min():.4f}, {umap_reduced.max():.4f}]")
print(f"  Any NaN               : {np.isnan(umap_reduced).any()}")
print(f"  Any Inf               : {np.isinf(umap_reduced).any()}")

assert not np.isnan(umap_reduced).any(), "NaN values found in UMAP output"
assert not np.isinf(umap_reduced).any(), "Inf values found in UMAP output"
print("  UMAP vectors clean.")

# ════════════════════════════════════════════════════════════════
# CHECK 4 — RESULTS FILE STRUCTURE
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("CHECK 4 — step7_8_results.json Structure")
print("=" * 55)

required_keys = {"cluster_id", "size", "keywords", "dominant_emotion", "topic_label"}
missing_keys  = []
blank_labels  = 0

for entry in results:
    missing = required_keys - set(entry.keys())
    if missing:
        missing_keys.append((entry.get("cluster_id"), missing))
    if entry.get("topic_label") in (None, "", "null"):
        blank_labels += 1

if missing_keys:
    print(f"  MISSING KEYS in {len(missing_keys)} entries:")
    for cid, mk in missing_keys[:5]:
        print(f"    Cluster {cid}: missing {mk}")
else:
    print(f"  All required keys present in every entry.")

print(f"  Blank topic_labels    : {blank_labels} / {len(results)}  "
      f"(expected — T5 fills these in Step 9)")

# ════════════════════════════════════════════════════════════════
# CHECK 5 — KEYWORD QUALITY SPOT CHECK
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("CHECK 5 — Keyword Quality (top 5 clusters by size)")
print("=" * 55)

results_sorted = sorted(
    [r for r in results if r.get("cluster_id") != -1],
    key=lambda x: -x.get("size", 0)
)

for entry in results_sorted[:5]:
    cid       = entry.get("cluster_id")
    size      = entry.get("size")
    kws       = entry.get("keywords", [])
    dom_emo   = entry.get("dominant_emotion", "?")
    print(f"\n  Cluster {cid}  ({size} segments)")
    print(f"    Keywords  : {kws[:8]}")
    print(f"    Dom. emo  : {dom_emo}")
    print(f"    Topic lbl : '{entry.get('topic_label', '')}'"
          f"  ← blank = waiting for T5")

# ════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("VERDICT")
print("=" * 55)

issues = []
if np.isnan(umap_reduced).any():
    issues.append("NaN in UMAP vectors")
if missing_keys:
    issues.append(f"{len(missing_keys)} entries missing required keys")
if num_clusters == 0:
    issues.append("Zero clusters found — HDBSCAN may need tuning")
if noise_pct > 50:
    issues.append(f"Noise too high: {noise_pct:.1f}%")

if issues:
    print("  ISSUES FOUND:")
    for iss in issues:
        print(f"    - {iss}")
    print("\n  Fix issues above before proceeding to Step 9.")
else:
    print("  All checks passed. Ready for Step 9 (Topic Labeling).")
