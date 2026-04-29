"""
step 7 & 8
step 7: unsupervised theme discovery  (UMAP → HDBSCAN)
step 8: keyword extraction per cluster (c-TF-IDF)

input  : step6_enriched_embeddings.npy  (296000, 556)
         step6_metadata.json            (296000 segments)
output : step7_cluster_labels.npy       — cluster id per segment (-1 = noise)
         step8_keywords.json            — top keywords per cluster
         step7_8_results.json           — full per-cluster summary
"""

import numpy as np
import json
import pickle
from collections import defaultdict, Counter
import math
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("Loading embeddings and metadata ...")
embeddings = np.load("data_models/step6_enriched_embeddings.npy")   # (296000, 556)
with open("jsons/step6_metadata.json") as f:
    metadata = json.load(f)

print(f"  Embeddings : {embeddings.shape}  dtype={embeddings.dtype}")
print(f"  Segments   : {len(metadata)}")

# UMAP on full 296k is very slow — fit on a sample, transform all
# standard bertopic-style approach for large corpora
SAMPLE_SIZE = 50_000          # fit umap on this; transform full set after
UMAP_COMPONENTS = 10          # dims fed to HDBSCAN
RANDOM_STATE = 42

print("\n" + "=" * 60)
print("STEP 7a — UMAP dimensionality reduction")
print(f"  Fitting on {SAMPLE_SIZE} samples, then transforming all 296k ...")

from umap import UMAP

rng = np.random.default_rng(RANDOM_STATE)
sample_idx = rng.choice(len(embeddings), size=SAMPLE_SIZE, replace=False)
sample_emb  = embeddings[sample_idx]

umap_model = UMAP(
    n_components=UMAP_COMPONENTS,
    n_neighbors=15,          # local neighborhood size
    min_dist=0.0,            # keep clusters tight — better for HDBSCAN
    metric="cosine",
    random_state=RANDOM_STATE,
    low_memory=True,         # important for large datasets
    verbose=True,
)

umap_model.fit(sample_emb)
print("  UMAP fitted. Transforming full corpus ...")

# transform in batches to avoid memory spikes
BATCH = 10_000
reduced = np.zeros((len(embeddings), UMAP_COMPONENTS), dtype=np.float32)
for start in range(0, len(embeddings), BATCH):
    end = min(start + BATCH, len(embeddings))
    reduced[start:end] = umap_model.transform(embeddings[start:end])
    if (start // BATCH) % 5 == 0:
        print(f"    ... transformed {end}/{len(embeddings)}")

print(f"  Reduced shape: {reduced.shape}")
np.save("data_models/step7_umap_reduced.npy", reduced)
print("  Saved → step7_umap_reduced.npy")

print("\n" + "=" * 60)
print("STEP 7b — HDBSCAN clustering ...")

import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=100,    # min segments per theme cluster
    min_samples=10,          # controls noise sensitivity
    metric="euclidean",      # on UMAP-reduced space
    cluster_selection_method="eom",   # excess of mass — finds varied sizes
    prediction_data=True,    # needed for soft membership later
    core_dist_n_jobs=-1,     # all CPU cores
)

clusterer.fit(reduced)
labels = clusterer.labels_          # -1 = noise
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_count = np.sum(labels == -1)

print(f"  Clusters found : {n_clusters}")
print(f"  Noise segments : {noise_count} ({100*noise_count/len(labels):.1f}%)")
print(f"  Label distribution (top 10 clusters by size):")

label_counts = Counter(labels)
for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1])[:11]:
    name = "NOISE" if lbl == -1 else f"Cluster {lbl:>3}"
    print(f"    {name}: {cnt:>7} segments")

np.save("data_models/step7_cluster_labels.npy", labels)
print("  Saved → step7_cluster_labels.npy")

print("\n" + "=" * 60)
print("STEP 8 — c-TF-IDF keyword extraction per cluster ...")

import re
from math import log

STOPWORDS = {
    "i","me","my","myself","we","our","you","your","he","she","it","they",
    "them","this","that","was","is","are","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","a","an","the","and","or","but","in","on","at","to","for",
    "of","with","as","by","from","up","about","into","through","then",
    "than","so","if","when","while","there","just","also","not","no",
    "its","it's","i'm","i've","i'll","i'd","don't","can't","didn't",
    "got","get","go","went","came","come","back","out","one","two","more",
    "some","all","very","really","around","suddenly","felt","feel","started",
}

def tokenize(text):
    """lowercase, letters only, min length 3"""
    if not isinstance(text, str):
        return []
    return [w for w in re.findall(r"[a-z]+", text.lower())
            if len(w) >= 3 and w not in STOPWORDS]

# concat all segment tokens per cluster
print("  Building cluster documents ...")
cluster_docs = defaultdict(list)   # cluster_id → flat token list
for idx, lbl in enumerate(labels):
    if lbl == -1:
        continue
    tokens = tokenize(metadata[idx].get("text", ""))
    cluster_docs[lbl].extend(tokens)

# TF per cluster
print("  Computing TF per cluster ...")
cluster_tf = {}
for lbl, tokens in cluster_docs.items():
    total = len(tokens)
    tf = Counter(tokens)
    # normalize by cluster size
    cluster_tf[lbl] = {w: c / total for w, c in tf.items()}

# IDF across clusters
print("  Computing IDF across clusters ...")
all_cluster_ids = list(cluster_tf.keys())
num_clusters = len(all_cluster_ids)

# df[word] = number of clusters where word appears
df = Counter()
for lbl, tf in cluster_tf.items():
    for word in tf:
        df[word] += 1

idf = {word: log(1 + num_clusters / (1 + freq)) for word, freq in df.items()}

# c-TF-IDF = TF_cluster * IDF_global
TOP_K = 15   # keywords per cluster
cluster_keywords = {}
for lbl in all_cluster_ids:
    scores = {w: cluster_tf[lbl][w] * idf[w] for w in cluster_tf[lbl]}
    top = sorted(scores.items(), key=lambda x: -x[1])[:TOP_K]
    cluster_keywords[lbl] = [w for w, _ in top]

print(f"  Keywords extracted for {len(cluster_keywords)} clusters.")
print("\n  Sample keywords (first 5 clusters):")
for lbl in sorted(cluster_keywords)[:5]:
    print(f"    Cluster {lbl:>3}: {', '.join(cluster_keywords[lbl])}")

print("\n  Computing dominant emotion per cluster ...")

EMOTION_LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization","relief",
    "remorse","sadness","surprise","neutral"
]

cluster_emotion_sums  = defaultdict(lambda: np.zeros(len(EMOTION_LABELS)))
cluster_emotion_counts = defaultdict(int)

for idx, lbl in enumerate(labels):
    if lbl == -1:
        continue
    ev = metadata[idx].get("emotion_vector")
    if ev and isinstance(ev, list) and len(ev) == len(EMOTION_LABELS):
        cluster_emotion_sums[lbl]  += np.array(ev)
        cluster_emotion_counts[lbl] += 1

cluster_dominant_emotion = {}
cluster_emotion_avg = {}
for lbl in all_cluster_ids:
    if cluster_emotion_counts[lbl] > 0:
        avg = cluster_emotion_sums[lbl] / cluster_emotion_counts[lbl]
        dominant_idx = int(np.argmax(avg))
        cluster_dominant_emotion[lbl] = EMOTION_LABELS[dominant_idx]
        cluster_emotion_avg[lbl] = {
            EMOTION_LABELS[i]: round(float(avg[i]), 4) for i in range(len(EMOTION_LABELS))
        }
    else:
        cluster_dominant_emotion[lbl] = "unknown"
        cluster_emotion_avg[lbl] = {}

print("\n" + "=" * 60)
print("Saving outputs ...")

with open("jsons/step8_keywords.json", "w") as f:
    json.dump({str(k): v for k, v in cluster_keywords.items()}, f, indent=2)
print("  Saved → step8_keywords.json")

# full cluster summary — step9 topic label gets filled in later
results = []
for lbl in sorted(all_cluster_ids):
    size = label_counts[lbl]
    results.append({
        "cluster_id":        lbl,
        "size":              size,
        "keywords":          cluster_keywords.get(lbl, []),
        "dominant_emotion":  cluster_dominant_emotion.get(lbl, "unknown"),
        "emotion_avg":       cluster_emotion_avg.get(lbl, {}),
        "topic_label":       "",  # step9 fills this
    })

with open("jsons/step7_8_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved → step7_8_results.json")

print("\n" + "=" * 60)
print("DONE — Summary")
print(f"  Total segments       : {len(embeddings):>8}")
print(f"  Clustered segments   : {len(embeddings) - noise_count:>8}")
print(f"  Noise segments       : {noise_count:>8}")
print(f"  Clusters discovered  : {n_clusters:>8}")
print(f"  Keywords per cluster : {TOP_K:>8}")
print("\nOutput files:")
print("  step7_umap_reduced.npy   — UMAP 10-d vectors")
print("  step7_cluster_labels.npy — cluster id per segment")
print("  step8_keywords.json      — c-TF-IDF keywords per cluster")
print("  step7_8_results.json     — full cluster summaries → input for Step 9")
print("=" * 60)
