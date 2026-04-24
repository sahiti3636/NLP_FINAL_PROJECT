"""
Dreamscape Mapper — Step 9
Topic Label Generation — keyword-driven theme mapping
Input  : step7_8_results.json
Output : step9_results.json
"""

import json
from collections import Counter

# ── Theme map — ordered by specificity (most specific first) ─────────────────
THEME_MAP = [
    # Specific vivid dream themes
    ("Flying & Levitation",           ["flying","flight","float","soar","hover","airborne","glide","wings","sky","altitude"]),
    ("Falling & Plunging",            ["falling","fell","plunge","drop","cliff","tumble","slip","edge","plummet","abyss"]),
    ("Chase & Pursuit",               ["chase","chasing","chased","escape","flee","pursuer","hunter","catch","running","cornered"]),
    ("Teeth & Body Integrity",        ["teeth","tooth","dentist","bite","mouth","shatter","crumble","loose","gums","dental"]),
    ("Death & Dying",                 ["death","dead","die","dying","funeral","grave","corpse","burial","killed","deceased","murdered"]),
    ("Violence & Attack",             ["attack","fight","gun","knife","wound","shoot","stab","punch","weapon","assault","blood","battle","murder"]),
    ("Romantic & Intimate",           ["kiss","romantic","intimate","desire","attracted","attraction","embrace","love","sexuality","sensual","sexually"]),
    ("Grief & Crying",                ["crying","grief","mourning","tears","weeping","sob","loss","bereaved","heartbroken","devastated"]),
    ("Anger & Frustration",           ["annoyed","appalled","frustrated","furious","rage","angry","irritated","outraged","disgusted","disappointed","shocked"]),
    ("Fear & Terror",                 ["terrified","terror","afraid","scared","panic","horror","dread","nightmare","fearful","threatening","menacing"]),
    ("Disorientation & Confusion",    ["confused","disoriented","flustered","lost","bewildered","dazed","disordered","bizarre","surreal","strange","weird"]),
    ("Emotional Distress",            ["relieved","upset","distressed","anxious","overwhelmed","nervous","tense","worried","uneasy","exhausted"]),
    ("Named Characters & People",     ["john","bill","frank","jennifer","ashley","dorothy","matthew","brian","aunt","andrew","jane","conn"]),
    ("School & Academic Life",        ["school","class","exam","test","teacher","student","grade","homework","university","college","campus","lecture"]),
    ("Work & Professional Life",      ["work","office","boss","career","job","task","deadline","colleague","meeting","project","interview","business"]),
    ("Navigation & Urban Space",      ["street","road","building","hospital","church","school","river","field","mountain","city","town","bridge","park"]),
    ("Indoor Spaces & Architecture",  ["floor","wall","window","stairs","ceiling","corridor","hallway","basement","attic","door","room","bathroom"]),
    ("Water & Natural Landscape",     ["ocean","river","lake","flood","swim","wave","drown","shore","beach","stream","waterfall","rain","sea","woods","forest","mountain"]),
    ("Travel & Movement",             ["driving","drive","car","road","train","plane","bus","travel","journey","highway","vehicle","riding","trip"]),
    ("Body & Physical Sensation",     ["feet","hair","legs","hands","body","physical","inches","skin","touch","pain","sensation","limping","bleeding"]),
    ("Time & Measurement",            ["years","minutes","clock","hours","seconds","year","three","four","five","schedule","timing","late"]),
    ("Social Gathering & Talk",       ["talking","conversation","party","crowd","laugh","argue","gathering","chat","discussion","group","office","listening"]),
    ("Childhood & Family Memory",     ["childhood","memory","young","past","nostalgia","recall","episode","growing","youth","remember"]),
    ("Positive Emotion & Joy",        ["beautiful","wonderful","happy","joyful","delighted","pleased","impressed","lucky","honored","giggle","cheerful"]),
    ("Spiritual & Supernatural",      ["ghost","spirit","magic","demon","angel","supernatural","mystical","prophecy","vision","sacred","holy"]),
    ("Identity & Self-Reflection",    ["independence","clever","informal","affectionate","idyllic","femininity","identity","self","personal","character"]),
    ("Escape & Freedom",              ["escape","free","leave","away","exit","open","outside","ahead","quickly","fast","flee","break"]),
]

def generate_label(keywords, dominant_emotion):
    if not keywords:
        return "Unclassified Dream Theme"

    kw_set = set(w.lower() for w in keywords)

    # Score every theme — count how many trigger words match
    scores = []
    for theme_name, trigger_words in THEME_MAP:
        score = sum(1 for t in trigger_words if t in kw_set)
        scores.append((score, theme_name))

    scores.sort(reverse=True)
    best_score, best_theme = scores[0]

    # Need at least 1 match; prefer 2+
    if best_score >= 1:
        return best_theme

    # Pure fallback — use emotion + top 2 keywords
    top2 = " & ".join(keywords[:2]).title()
    emotion = dominant_emotion.title() if dominant_emotion not in ("unknown", "") else "Dream"
    return f"{emotion} Theme: {top2}"

# ── Load and process ──────────────────────────────────────────────────────────
with open("jsons/step7_8_results.json") as f:
    clusters = json.load(f)
print(f"Loaded {len(clusters)} clusters.")

results = []
label_counts = Counter()

for cluster in clusters:
    label = generate_label(cluster["keywords"], cluster["dominant_emotion"])
    cluster["topic_label"] = label
    label_counts[label] += 1
    results.append(cluster)

# ── Save ──────────────────────────────────────────────────────────────────────
with open("jsons/step9_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved step9_results.json with {len(results)} clusters.")

# ── Sanity check ──────────────────────────────────────────────────────────────
print("\nSample output (5 largest clusters):")
sorted_by_size = sorted(results, key=lambda x: -x["size"])[:5]
for r in sorted_by_size:
    print(f"  Cluster {r['cluster_id']:>3} | "
          f"size={r['size']:>6} | "
          f"{r['dominant_emotion']:>12} | "
          f"label={r['topic_label']}")

print("\nTheme distribution (all themes):")
for label, count in label_counts.most_common():
    print(f"  {count:>3} clusters — {label}")

print("\nStep 9 complete.")
