"""
step 9 final
keyword-driven label construction + emotion normalization
"""

import json, re
from collections import Counter

# trust and anticipation tend to dominate; downweight so other emotions surface
DOWNWEIGHT = {"trust": 0.7, "anticipation": 0.75}

# pick dominant emotion with trust/anticipation downweighted so they don't always win
def normalized_dominant_emotion(emotion_avg):
    if not emotion_avg:
        return "unknown"
    scores = {e: v * DOWNWEIGHT.get(e, 1.0) for e, v in emotion_avg.items()}
    return max(scores, key=scores.get)

# keyword → short theme fragment
# body/physical
KEYWORD_TAGS = {
    "teeth":"Dental Anxiety", "tooth":"Dental Anxiety", "hair":"Body Image",
    "blood":"Bodily Harm", "naked":"Vulnerability", "legs":"Physical Mobility",
    "feet":"Physical Grounding", "arms":"Physical Reach", "skin":"Body Sensation",
    # locations
    "school":"School Setting", "hospital":"Medical Setting", "church":"Sacred Space",
    "office":"Workplace", "street":"Urban Navigation", "road":"Journey Path",
    "forest":"Wilderness", "ocean":"Deep Waters", "river":"Flowing Water",
    "mountain":"High Ground", "building":"Architectural Space", "stairs":"Vertical Transition",
    "floor":"Ground Level", "wall":"Boundary", "window":"Threshold",
    "bathroom":"Private Space", "basement":"Hidden Depths", "attic":"Past Storage",
    # actions / states
    "flying":"Aerial Freedom", "falling":"Loss of Control", "chase":"Pursuit",
    "chasing":"Pursuit", "running":"Flight Response", "driving":"Navigation Control",
    "swimming":"Water Immersion", "crying":"Emotional Release", "fighting":"Conflict",
    "hiding":"Avoidance", "searching":"Seeking", "escaping":"Escape Attempt",
    "talking":"Social Exchange", "arguing":"Interpersonal Conflict",
    "laughing":"Social Joy", "kissing":"Romantic Contact",
    # emotions / states
    "afraid":"Fear Response", "scared":"Fear Response", "lost":"Disorientation",
    "confused":"Mental Confusion", "angry":"Anger Arousal", "upset":"Emotional Distress",
    "anxious":"Anxiety State", "relieved":"Relief Response", "happy":"Positive Affect",
    "sad":"Sadness State", "guilty":"Guilt Experience", "ashamed":"Shame Response",
    "terrified":"Terror Response", "panicking":"Panic State",
    # dream archetypes
    "death":"Mortality Theme", "dead":"Death Encounter", "dying":"Death Process",
    "ghost":"Supernatural Presence", "demon":"Dark Entity", "angel":"Benevolent Spirit",
    "wedding":"Commitment Ritual", "baby":"New Beginning", "pregnancy":"Creation",
    "fire":"Destruction Force", "flood":"Overwhelming Force", "earthquake":"Upheaval",
    "exam":"Performance Pressure", "test":"Evaluation Anxiety", "late":"Time Pressure",
    "money":"Financial Concern", "naked":"Exposure Anxiety", "mirror":"Self Reflection",
    "phone":"Communication Need", "knife":"Threat Symbol", "gun":"Power Symbol",
    # people
    "stranger":"Unknown Presence", "police":"Authority Figure", "doctor":"Healing Figure",
    "teacher":"Authority Knowledge",
}

# fallback when no keyword tags match
EMOTION_TEMPLATES = {
    "fear":        "{kw1} and {kw2} — Fear",
    "anger":       "{kw1} and {kw2} — Anger",
    "sadness":     "{kw1} and {kw2} — Sadness",
    "joy":         "{kw1} and {kw2} — Joy",
    "trust":       "{kw1} and {kw2} — Trust",
    "anticipation":"{kw1} and {kw2} — Anticipation",
    "disgust":     "{kw1} and {kw2} — Disgust",
    "surprise":    "{kw1} and {kw2} — Surprise",
    "unknown":     "{kw1} and {kw2} Dream",
}

# title-case a string but keep small words lowercase
def title(s):
    small = {'and','or','of','the','a','an','in','on','at','to','for'}
    words = s.split()
    return ' '.join(
        w.title() if i == 0 or w.lower() not in small else w.lower()
        for i, w in enumerate(words)
    )

# build a readable theme label from matched keyword tags, falls back to templates
def generate_label(keywords, dominant_emotion):
    if not keywords:
        return f"{dominant_emotion.title()} Dream Pattern"

    kw_lower = [k.lower() for k in keywords]

    # collect matched tags in keyword order — stop at 2
    matched_tags = []
    for kw in kw_lower:
        tag = KEYWORD_TAGS.get(kw)
        if tag and tag not in matched_tags:
            matched_tags.append(tag)
        if len(matched_tags) == 2:
            break

    if len(matched_tags) == 2:
        return title(f"{matched_tags[0]} and {matched_tags[1]}")

    if len(matched_tags) == 1:
        emo = dominant_emotion.title() if dominant_emotion != "unknown" else ""
        if emo:
            return title(f"{matched_tags[0]} with {emo}")
        return matched_tags[0]

    # no tag match — use top 2 content keywords + emotion template
    content = [k for k in kw_lower if len(k) > 4][:2]
    if len(content) >= 2:
        tmpl = EMOTION_TEMPLATES.get(dominant_emotion, "{kw1} and {kw2} Dream")
        return title(tmpl.format(kw1=content[0], kw2=content[1]))
    elif len(content) == 1:
        return title(f"{content[0]} {dominant_emotion} Dream")

    # last resort
    return title(f"{kw_lower[0]} Dream Theme")

with open("jsons/step7_8_results.json") as f:
    clusters = json.load(f)
print(f"Loaded {len(clusters)} clusters.")

results = []
label_counts = Counter()

for cluster in clusters:
    cluster["dominant_emotion"] = normalized_dominant_emotion(
        cluster.get("emotion_avg", {})
    )
    label = generate_label(cluster["keywords"], cluster["dominant_emotion"])
    cluster["topic_label"] = label
    label_counts[label] += 1
    results.append(cluster)

with open("jsons/step9_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved step9_results.json with {len(results)} clusters.")

print("\nTop 10 largest clusters:")
for r in sorted(results, key=lambda x: -x["size"])[:10]:
    print(f"  Cluster {r['cluster_id']:>3} | "
          f"size={r['size']:>6} | "
          f"{r['dominant_emotion']:>12} | {r['topic_label']}")

print("\nEmotion distribution:")
emo_counts = Counter(r["dominant_emotion"] for r in results)
for emo, cnt in emo_counts.most_common():
    pct = 100 * cnt / len(results)
    bar = '█' * int(pct / 2)
    print(f"  {emo:>15}: {cnt:>3} ({pct:4.1f}%) {bar}")

print(f"\nLabel diversity : {len(label_counts)} unique labels out of {len(results)} clusters")
print("Most repeated labels:")
for label, cnt in label_counts.most_common(8):
    print(f"  {cnt:>3}x — {label}")

print("\nSample structured output (paper format):")
for r in sorted(results, key=lambda x: -x["size"])[:5]:
    print(f"\n  Topic_Cluster    : {r['topic_label']}")
    print(f"  Dominant_Emotion : {r['dominant_emotion']}")
    print(f"  Keywords         : {r['keywords'][:5]}")
    print(f"  Cluster_Size     : {r['size']}")

print("\nStep 9 complete.")
