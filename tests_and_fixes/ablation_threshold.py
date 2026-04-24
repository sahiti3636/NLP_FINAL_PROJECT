import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def run_threshold_ablation():
    print("[1/4] Loading step9_results.json...")
    try:
        with open('jsons/step9_results.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: step9_results.json not found.")
        return

    # Extract all topic labels
    labels = [entry.get('topic_label', '') for entry in data if 'topic_label' in entry]
    print(f"[2/4] Processing {len(labels)} cluster labels...")

    # Vectorize labels to calculate similarities
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(labels)
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Define a range of thresholds to test
    thresholds = np.linspace(0.3, 0.95, 15)
    topic_counts = []

    print("[3/4] Simulating merges at various thresholds...")
    for tau in thresholds:
        # Simple merging logic: 
        # Groups labels together if their similarity > tau
        merged_indices = set()
        count = 0
        for i in range(len(labels)):
            if i not in merged_indices:
                count += 1
                # Find all other labels similar to this one
                matches = np.where(sim_matrix[i] > tau)[0]
                for m in matches:
                    merged_indices.add(m)
        topic_counts.append(count)

    # 4. Visualization
    print("[4/4] Generating Sensitivity Plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, topic_counts, marker='o', linestyle='-', color='#2c3e50', linewidth=2)
    
    # Highlight the "Sweet Spot" (assuming 0.8 is your default)
    plt.axvline(x=0.8, color='#e74c3c', linestyle='--', label='Aethera Default (τ=0.8)')
    plt.fill_between(thresholds, topic_counts, color='#ecf0f1', alpha=0.5)

    plt.title('Ablation: Consolidation Threshold vs. Topic Resolution')
    plt.xlabel('Similarity Threshold (τ)')
    plt.ylabel('Number of Final Unique Topics')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Annotate the "Over-merging" vs "Redundancy" zones
    plt.annotate('Over-generalization\n(Loss of Nuance)', xy=(0.4, topic_counts[2]), xytext=(0.35, topic_counts[2]+20),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    plt.annotate('High Redundancy\n(Noise)', xy=(0.9, topic_counts[-2]), xytext=(0.75, topic_counts[-2]+30),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.savefig('results_images/ablation_threshold_results.png')
    print("Done! Sensitivity graph saved to ablation_threshold_results.png")

if __name__ == "__main__":
    run_threshold_ablation()