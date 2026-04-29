import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

INPUT_FILE = "jsons/step10_final_results.json"
STATS_OUT = "jsons/step11_global_statistics.json"

# compute theme-emotion probs, correlation matrix, transitions, global emotion avg
def perform_statistical_analysis():
    print("=" * 60)
    print("STEP 11 - GLOBAL STATISTICAL ANALYSIS")
    print("=" * 60)

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    # expand emotion_avg dict into columns — easier to compute on
    df = pd.DataFrame(data)
    emotions_df = df['emotion_avg'].apply(pd.Series).fillna(0.0)
    full_df = pd.concat([df.drop(columns=['emotion_avg']), emotions_df], axis=1)

    stats_report = {}

    # which themes dominate when a specific emotion is primary
    print("[1/4] Computing Theme-Emotion Conditional Probabilities...")
    emotion_list = [e for e in emotions_df.columns if e != 'undetermined']
    theme_emotion_likelihood = {}

    for emotion in emotion_list:
        top_themes = full_df[full_df['dominant_emotion'] == emotion] \
            .sort_values(by='size', ascending=False) \
            .head(5)['topic_label'].tolist()
        theme_emotion_likelihood[emotion] = top_themes

    stats_report["top_themes_per_emotion"] = theme_emotion_likelihood

    # pairwise correlation between emotion dimensions across clusters
    print("[2/4] Generating Correlation Coefficients...")
    correlation_matrix = emotions_df.corr().round(3).to_dict()
    stats_report["emotion_correlation_matrix"] = correlation_matrix

    # simulated narrative transitions — real impl would use original dream seq IDs
    # here we use cluster order as a proxy, which is imperfect but fine for now
    print("[3/4] Calculating Temporal Transition Probabilities...")
    transitions = defaultdict(Counter)
    labels = full_df['topic_label'].tolist()

    for i in range(len(labels) - 1):
        transitions[labels[i]][labels[i+1]] += 1

    # normalize to probs — only keep transitions > 10% to avoid noise
    prob_transitions = {}
    for theme, targets in transitions.items():
        total = sum(targets.values())
        prob_transitions[theme] = {t: round(c/total, 3) for t, c in targets.items() if (c/total) > 0.1}

    stats_report["temporal_transitions"] = prob_transitions

    print("[4/4] Finalizing Global Affective Profile...")
    global_emotion_means = emotions_df.mean().round(4).to_dict()
    stats_report["global_affective_averages"] = global_emotion_means

    with open(STATS_OUT, "w") as f:
        json.dump(stats_report, f, indent=2)

    print("\nSUCCESS: Global Statistics Generated.")
    print(f"File Saved: {STATS_OUT}")

    sample_emo = max(global_emotion_means, key=global_emotion_means.get)
    print(f"\nINSIGHT: The most frequent global affective state is '{sample_emo}' ({global_emotion_means[sample_emo]}).")
    print("=" * 60)

if __name__ == "__main__":
    perform_statistical_analysis()