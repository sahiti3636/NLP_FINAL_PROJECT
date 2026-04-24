import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
from datetime import datetime

def log_print(message, log_file):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    with open(log_file, "a") as f:
        f.write(formatted_msg + "\n")

def run_emotion_ablation():
    log_file = "tests_and_fixes/log_ablation_emotion.txt"
    # Clear previous log
    with open(log_file, "w") as f:
        f.write(f"--- Ablation Study Log: {datetime.now().strftime('%Y-%m-%d')} ---\n")

    log_print("Starting Emotion Signal Ablation Study...", log_file)

    # 1. Load Data
    try:
        log_print("Loading .npy files from directory...", log_file)
        embeddings = np.load('data_models/step6_enriched_embeddings.npy')
        labels = np.load('data_models/step7_cluster_labels.npy')
        log_print(f"Successfully loaded. Embeddings shape: {embeddings.shape}", log_file)
        log_print(f"Successfully loaded. Labels shape: {labels.shape}", log_file)
    except Exception as e:
        log_print(f"CRITICAL ERROR: Could not load .npy files. {str(e)}", log_file)
        return

    # 2. Perform Ablation
    log_print("Ablating data: Removing 28-dimensional emotion vector...", log_file)
    # The last 28 indices are the GoEmotions scores (as per Step 6 logic)
    semantic_only = embeddings[:, :-28]
    log_print(f"Ablated shape (Semantic only): {semantic_only.shape}", log_file)

    # 3. TSNE for Original Data
    log_print("Computing TSNE for ORIGINAL embeddings (This may take a moment)...", log_file)
    start_time = time.time()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    vis_orig = tsne.fit_transform(embeddings)
    duration = round(time.time() - start_time, 2)
    log_print(f"Original TSNE completed in {duration} seconds.", log_file)

    # 4. TSNE for Ablated Data
    log_print("Computing TSNE for ABLATED embeddings...", log_file)
    start_time = time.time()
    vis_abla = tsne.fit_transform(semantic_only)
    duration = round(time.time() - start_time, 2)
    log_print(f"Ablated TSNE completed in {duration} seconds.", log_file)

    # 5. Visualization and Saving
    log_print("Generating comparison plots...", log_file)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.scatter(vis_orig[:, 0], vis_orig[:, 1], c=labels, cmap='viridis', s=10, alpha=0.6)
    ax1.set_title('Original (Semantic + Emotion Signal)')
    
    ax2.scatter(vis_abla[:, 0], vis_abla[:, 1], c=labels, cmap='viridis', s=10, alpha=0.6)
    ax2.set_title('Ablated (Semantic Only - No Emotion)')

    plt.suptitle('Ablation: Thematic Clustering without Affective Weighting')
    
    output_img = 'results_images/ablation_emotion_results.png'
    plt.savefig(output_img)
    log_print(f"Visualization saved to {output_img}", log_file)
    log_print("Ablation study complete. Check the generated image for cluster shift analysis.", log_file)

if __name__ == "__main__":
    run_emotion_ablation()