import json
import torch
import random
from collections import defaultdict
from torch.utils.data import random_split

# Standard Aethera Emotion Set
NRC_EMOTIONS = [
    "anger", "anticipation", "disgust", "fear", 
    "joy", "sadness", "surprise", "trust"
]

try:
    from nrclex import NRCLex
    _NRC_AVAILABLE = True
except ImportError:
    _NRC_AVAILABLE = False
    print("Error: nrclex not installed. Run 'pip install nrclex'")

def get_safe_nrc_scores(nrc_obj):
    """
    Universal fallback for NRCLex. 
    If the library fails to provide scores, we manually calculate them 
    from the identified emotional words.
    """
    # 1. Check standard attributes
    for attr in ['raw_emotion_scores', 'affect_frequencies']:
        val = getattr(nrc_obj, attr, None)
        if val and isinstance(val, dict) and any(val.values()):
            return val
            
    # 2. Manual Reconstruction Fallback (The 'Deep Fix')
    # NRCLex objects usually have 'affect_dict' or 'raw_emotion_scores'
    # If those are empty, we check the words directly.
    manual_scores = defaultdict(int)
    
    # Access the internal dictionary that maps words to emotions
    # This is usually available even if the summary attributes fail.
    affect_dict = getattr(nrc_obj, 'affect_dict', {})
    
    if affect_dict:
        for word, emotions in affect_dict.items():
            for emo in emotions:
                if emo in NRC_EMOTIONS:
                    manual_scores[emo] += 1
                    
    return dict(manual_scores) if manual_scores else {}

def run_val_emotion_check(annotations_path="jsons/dream_annotations.json"):
    if not _NRC_AVAILABLE:
        return

    print(f"\n{'='*60}")
    print(f"  EMOTION ACCURACY — FIXED VALIDATION PIPELINE")
    print(f"{'='*60}")

    # 1. Load Data
    try:
        with open(annotations_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {annotations_path} not found.")
        return

    # 2. Replicate the 10% Validation Split (Seed 42)
    val_size = max(1, int(0.1 * len(data)))
    train_size = len(data) - val_size
    _, val_subset = random_split(
        data, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_data = [data[i] for i in val_subset.indices]
    print(f"Isolated {len(val_data)} validation samples.")

    hits = 0
    total = 0
    agreements = 0
    emo_stats = {emo: {"count": 0, "agree": 0} for emo in NRC_EMOTIONS}

    print(f"\n{'TEXT SAMPLE':<35} | {'DOMINANT':<12} | {'MATCH'}")
    print("-" * 65)

    for seg in val_data:
        tokens = seg.get("tokens", [])
        raw_text = " ".join(tokens).strip().replace('\n', ' ')
        ann_vec = seg.get("emotion_vector", {})
        
        if not raw_text or not ann_vec:
            continue

        # FIX: Truncate text to 2000 chars to avoid OS "File Name Too Long" error.
        # This ensures NRCLex treats it as a string, not a file path.
        safe_text = raw_text[:2000] if len(raw_text) > 2000 else raw_text

        try:
            nrc = NRCLex(safe_text)
            scores = get_safe_nrc_scores(nrc)
            
            ann_dom = max(ann_vec, key=ann_vec.get)
            emo_stats[ann_dom]["count"] += 1
            
            has_nrc_data = sum(scores.values()) > 0
            if has_nrc_data:
                hits += 1
                live_dom = max(scores, key=scores.get)
                
                if live_dom == ann_dom:
                    agreements += 1
                    emo_stats[ann_dom]["agree"] += 1
                    match_str = "✓ YES"
                else:
                    match_str = f"✗ {live_dom}"
            else:
                match_str = "N/A (Miss)"

            # Print first 10 for progress verification
            if total < 10:
                print(f"{safe_text[:32]:<35}... | {ann_dom:<12} | {match_str}")
            
            total += 1
            if total % 1000 == 0:
                print(f"Processed {total}/{len(val_data)} samples...")

        except OSError:
            continue

    # 3. Final Summary
    print(f"\n{'='*60}")
    print(f"  FINAL METRICS")
    print(f"{'='*60}")
    print(f"  Lexicon Hit-Rate     : {(hits/total)*100:.2f}%")
    print(f"  Dominant Agree       : {(agreements/total)*100:.2f}%")
    print(f"\n  {'EMOTION':<14} | {'SAMPLES':>7} | {'ACCURACY':>9}")
    print("  " + "-" * 36)
    
    for emo in NRC_EMOTIONS:
        count = emo_stats[emo]["count"]
        acc = (emo_stats[emo]["agree"] / count * 100) if count > 0 else 0
        print(f"  {emo:<14} | {count:>7} | {acc:8.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    run_val_emotion_check()