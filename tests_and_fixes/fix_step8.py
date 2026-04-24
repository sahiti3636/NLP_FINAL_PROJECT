import numpy as np, json
from collections import Counter, defaultdict
import re, math

labels = np.load('data_models/step7_cluster_labels.npy')
meta   = json.load(open('jsons/step6_metadata.json'))

from nltk.corpus import stopwords
nltk_stops = set(stopwords.words('english'))

dream_stops = {
    'dream','dreamed','dreaming','dreamt','woke','wake','waking','sleep',
    'sleeping','slept','night','bed','remember','remembered','felt','feel',
    'feeling','feelings','know','knew','think','thought','thinking',
    'things','thing','something','someone','somehow','somewhere','everything',
    'nothing','anything','anyone','everyone','got','get','getting','went',
    'going','come','came','coming','back','around','just','like','said','say',
    'saw','see','seen','look','looked','looking','told','tell','left','right',
    'found','find','took','take','started','start','stopped','stop','turned',
    'turn','walked','walk','ran','run','asked','ask','tried','try','wanted',
    'want','needed','need','seemed','seem','called','call','put','stood',
    'stand','heard','hear','began','begin','let','made','make','kept','keep',
    'place','time','way','day','people','man','woman','guy','girl','old',
    'new','big','little','small','long','house','room','door','car','away',
    'inside','outside','front','behind','suddenly','then','there','here',
    'next','still','also','even','much','many','lot','end','face','head',
    'hand','hands','eyes','body','voice','moment','later','soon','ago',
    'once','first','last','another','every','really','actually','maybe',
    'quite','very','never','always','sometimes','immediately','finally',
    'already','together','sat','sit','realized','realised','noticed',
    'would','could','should','might','must','shall','will','also','even',
    'well','good','kind','seem','seems','since','though','trying','taking',
    'saying','says','home','table','sitting','standing','walking','looking',
    'going','coming','getting','having','being','doing','making','telling',
    'seeing','knowing','thinking','feeling','wanting','trying','using',
    'told','went','came','took','made','said','knew','left','found','kept',
    'gave','give','shows','show','around','toward','towards','behind',
    'front','inside','outside','across','along','beside','between','without',
    'within','upon','onto','into','from','with','that','this','these',
    'those','them','they','their','your','mine','ours','hers','your',
    'really','quite','rather','pretty','little','large','huge','tiny',
    'happy','angry','scared','afraid','sure','okay','fine','nice','weird',
    'strange','different','same','other','else','again','back','away',
    'down','over','just','only','also','both','each','more','most','some',
    'such','than','then','when','where','while','after','before','until',
    'about','above','below','under','over','near','next','last','mother','father',
    'man','woman','girl','boy','person','people','friend','past'
}

ALL_STOPS = nltk_stops | dream_stops

def tokenize(text):
    if not isinstance(text, str):
        return []
    tokens = re.findall(r'[a-z]+', text.lower())
    return [w for w in tokens if len(w) >= 4 and w not in ALL_STOPS]

# ── Emotion: NRC 8-emotion dict format ───────────────────────────────────────
NRC_EMOTIONS = ['anger','anticipation','disgust','fear','joy','sadness','surprise','trust']
N_EMO = len(NRC_EMOTIONS)
UNIFORM_VAL = round(1.0 / N_EMO, 3)  # 0.125

def is_uniform_fallback(ev):
    if not ev or not isinstance(ev, dict):
        return True
    vals = [ev.get(e, 0) for e in NRC_EMOTIONS]
    arr  = np.array(vals)
    # uniform fallback = all values exactly equal (within tiny tolerance)
    return float(np.max(arr) - np.min(arr)) < 0.01

def ev_to_array(ev):
    return np.array([ev.get(e, 0.0) for e in NRC_EMOTIONS])

print('Processing segments...')
cluster_docs           = defaultdict(list)
cluster_emotion_sums   = defaultdict(lambda: np.zeros(N_EMO))
cluster_emotion_counts = defaultdict(int)

for idx, lbl in enumerate(labels):
    if lbl == -1:
        continue
    tokens = tokenize(meta[idx].get('text', ''))
    cluster_docs[lbl].extend(tokens)
    ev = meta[idx].get('emotion_vector')
    if not is_uniform_fallback(ev):
        cluster_emotion_sums[lbl]   += ev_to_array(ev)
        cluster_emotion_counts[lbl] += 1

print('Segments processed.')
print('Computing c-TF-IDF...')

cluster_tf = {}
for lbl, tokens in cluster_docs.items():
    total = len(tokens)
    if total == 0:
        cluster_tf[lbl] = {}
        continue
    tf = Counter(tokens)
    cluster_tf[lbl] = {w: c / total for w, c in tf.items()}

num_clusters = len(cluster_tf)
df = Counter()
for tf in cluster_tf.values():
    for word in tf:
        df[word] += 1

idf = {w: math.log(1 + num_clusters / (1 + f)) for w, f in df.items()}

TOP_K = 15
cluster_keywords = {}
for lbl in cluster_tf:
    scores = {w: cluster_tf[lbl][w] * idf[w] for w in cluster_tf[lbl]}
    top    = sorted(scores.items(), key=lambda x: -x[1])[:TOP_K]
    cluster_keywords[lbl] = [w for w, _ in top]

print(f'Keywords extracted for {len(cluster_keywords)} clusters.')

print('Computing dominant emotions...')
label_counts    = Counter(labels)
all_cluster_ids = [k for k in label_counts if k != -1]

cluster_dominant_emotion = {}
cluster_emotion_avg      = {}
for lbl in all_cluster_ids:
    count = cluster_emotion_counts[lbl]
    if count > 0:
        avg      = cluster_emotion_sums[lbl] / count
        dominant = NRC_EMOTIONS[int(np.argmax(avg))]
    else:
        avg      = np.zeros(N_EMO)
        dominant = 'unknown'
    cluster_dominant_emotion[lbl] = dominant
    cluster_emotion_avg[lbl]      = {NRC_EMOTIONS[i]: round(float(avg[i]), 4) for i in range(N_EMO)}

with open('jsons/step8_keywords.json', 'w') as f:
    json.dump({str(k): v for k, v in cluster_keywords.items()}, f, indent=2)
print('Saved step8_keywords.json')

results = []
for lbl in sorted(all_cluster_ids):
    results.append({
        'cluster_id':       int(lbl),
        'size':             int(label_counts[lbl]),
        'keywords':         cluster_keywords.get(lbl, []),
        'dominant_emotion': cluster_dominant_emotion.get(lbl, 'unknown'),
        'emotion_avg':      cluster_emotion_avg.get(lbl, {}),
        'topic_label':      ''
    })

with open('jsons/step7_8_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved step7_8_results.json with', len(results), 'clusters')

print()
print('Sanity check — top 5 LARGEST clusters:')
top5 = [lbl for lbl, _ in label_counts.most_common() if lbl != -1][:5]
for r in results:
    if r['cluster_id'] in top5:
        print(f'  Cluster {r["cluster_id"]:>3} | {r["dominant_emotion"]:>12} | {", ".join(r["keywords"][:6])}')
