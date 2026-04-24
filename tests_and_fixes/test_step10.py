import json
from collections import Counter

def test_step10_output():
    FILE_PATH = "jsons/step10_final_results.json"
    print(f"Checking Step 10 Output: {FILE_PATH}...")

    try:
        with open(FILE_PATH, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load file. {e}")
        return

    total_clusters = len(data)
    print(f"Successfully loaded {total_clusters} clusters.\n")

    # --- CHECK 1: Key Validation (Matching your script's schema) ---
    required_keys = ["cluster_id", "topic_label", "dominant_emotion", "emotion_avg", "size"]
    missing_count = 0
    for entry in data:
        if not all(k in entry for k in required_keys):
            missing_count += 1
    
    print("CHECK 1 — Schema Consistency")
    if missing_count == 0:
        print(f"  Pass: All {total_clusters} entries contain the required keys.")
    else:
        print(f"  FAIL: {missing_count} entries are missing required keys.")

    # --- CHECK 2: Mathematical Accuracy of Dominant Emotion ---
    # Verifies that the 'dominant_emotion' label actually matches the highest value in 'emotion_avg'
    logic_errors = 0
    undetermined_count = 0

    for entry in data:
        avg = entry.get("emotion_avg", {})
        dom = entry.get("dominant_emotion")

        if dom == "undetermined":
            undetermined_count += 1
            continue

        if avg:
            # Find the key with the maximum float value
            max_key = max(avg, key=lambda k: float(avg[k]))
            if dom != max_key:
                logic_errors += 1

    print("\nCHECK 2 — Emotional Logic")
    if logic_errors == 0:
        print(f"  Pass: Dominant emotions match the highest average probability.")
    else:
        print(f"  FAIL: Found {logic_errors} clusters where the label does not match the max value.")
    
    if undetermined_count > 0:
        print(f"  Note: {undetermined_count} clusters marked as 'undetermined' (zero-vector clusters).")

    # --- CHECK 3: Data Readiness for Step 11 ---
    # Step 11 needs topic labels and emotions to compute conditional probabilities
    unique_topics = len(set(entry.get("topic_label") for entry in data))
    print("\nCHECK 3 — Diversity & Readiness")
    print(f"  Unique Topic Labels: {unique_topics}")
    
    emotion_dist = Counter(entry.get("dominant_emotion") for entry in data)
    print("\nFinal Emotion Distribution for Step 11 Statistics:")
    for emo, count in emotion_dist.most_common():
        pct = (count / total_clusters) * 100
        bar = "█" * int(pct / 2)
        print(f"  {emo:15} | {count:4} | {pct:5.1f}% {bar}")

    # --- VERDICT ---
    print("\n" + "="*60)
    if missing_count == 0 and logic_errors == 0:
        print("VERDICT: STEP 10 VALIDATED. Ready for Step 11 Statistical Analysis.")
    else:
        print("VERDICT: FAIL. Please re-run Step 10 and verify key names/logic.")
    print("="*60)

if __name__ == "__main__":
    test_step10_output()