import json
import matplotlib.pyplot as plt

def run_label_ablation():
    try:
        with open('jsons/step9_results.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: step9_results.json not found.")
        return

    # Before consolidation: Every cluster has a unique label
    ablated_count = len(data)
    
    # After consolidation: We count unique topic labels
    unique_labels = set(entry.get('topic_label', '') for entry in data if 'topic_label' in entry)
    original_count = len(unique_labels)

    # Visualization
    labels = ['Without Consolidation\n(Raw Clusters)', 'With Consolidation\n(Canonical Topics)']
    counts = [ablated_count, original_count]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, counts, color=['#bdc3c7', '#34495e'], width=0.5)
    
    plt.ylabel('Number of Unique Labels')
    plt.title('Ablation: Impact of Semantic Label Consolidation')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), 
                 ha='center', va='bottom', fontweight='bold')

    plt.ylim(0, max(counts) * 1.2)
    plt.tight_layout()
    plt.savefig('results_images/ablation_labels_results.png')
    print(f"Success: Consolidation reduced labels from {ablated_count} down to {original_count}.")

if __name__ == "__main__":
    run_label_ablation()