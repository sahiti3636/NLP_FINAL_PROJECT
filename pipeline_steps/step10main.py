import json
from collections import defaultdict

STEP9_RESULTS_PATH = "jsons/step9_results.json"
STEP10_PROFILES_OUT = "jsons/step10_emotion_profiles.json"
STEP10_FINAL_OUT = "jsons/step10_final_results.json"


# re-derive dominant emotion from step9 emotion_avg dicts
def recompute_from_step9_json(step9_data):
    profiles = []

    for c in step9_data:
        avg = c.get("emotion_avg", {})

        # values might be strings if json was edited manually — coerce to float
        total = sum(float(v) for v in avg.values()) if avg else 0.0

        if total == 0.0:
            dominant = "undetermined"
        else:
            dominant = max(avg, key=lambda k: float(avg[k]))

        profiles.append({
            "cluster_id": c["cluster_id"],
            "segment_count": c.get("size", 0),
            "emotion_avg": avg,
            "emotion_std": {},
            "dominant_emotion": dominant
        })

    return profiles


# join step9 cluster metadata with step10 emotion profiles by cluster_id
def merge_with_step9(step9_data, step10_profiles):
    step10_index = {
        p["cluster_id"]: p for p in step10_profiles
    }

    final_results = []

    for cluster in step9_data:
        cid = cluster["cluster_id"]
        em = step10_index.get(cid, {})

        record = {
            "cluster_id": cid,
            "topic_label": cluster.get("topic_label", "Unknown"),
            "size": cluster.get("size", 0),
            "keywords": cluster.get("keywords", []),
            "dominant_emotion": em.get(
                "dominant_emotion",
                cluster.get("dominant_emotion", "unknown")
            ),
            "emotion_avg": em.get(
                "emotion_avg",
                cluster.get("emotion_avg", {})
            ),
            "emotion_std": em.get("emotion_std", {}),
            "segment_count": em.get(
                "segment_count",
                cluster.get("size", 0)
            )
        }

        final_results.append(record)

    return final_results


# entry point — load, recompute, merge, save
def main():
    print("=" * 60)
    print("STEP 10 - DOMINANT EMOTION COMPUTATION")
    print("=" * 60)

    with open(STEP9_RESULTS_PATH, "r") as f:
        step9_data = json.load(f)

    print("Loaded {} clusters from Step 9".format(len(step9_data)))

    profiles = recompute_from_step9_json(step9_data)

    with open(STEP10_PROFILES_OUT, "w") as f:
        json.dump(profiles, f, indent=2)

    final_results = merge_with_step9(step9_data, profiles)

    with open(STEP10_FINAL_OUT, "w") as f:
        json.dump(final_results, f, indent=2)

    emotion_counts = defaultdict(int)

    for p in profiles:
        emotion_counts[p["dominant_emotion"]] += 1

    print("\nDominant Emotion Distribution")
    print("-" * 40)

    for emotion, count in sorted(
        emotion_counts.items(),
        key=lambda x: -x[1]
    ):
        print("{:<15} {}".format(emotion, count))

    print("\nDone successfully.")
    print("Saved:", STEP10_PROFILES_OUT)
    print("Saved:", STEP10_FINAL_OUT)
    print("=" * 60)


if __name__ == "__main__":
    main()