"""
dream_pipeline_production.py
---
production pipeline using actual trained models
loads bilstm, vocabs, umap embeddings, cluster assignments
"""

import re
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF

# nrclex is optional — graceful fallback if missing
try:
    from nrclex import NRCLex
    _NRC_AVAILABLE = True
except ImportError:
    _NRC_AVAILABLE = False
    print("Warning: nrclex not available. Install with: pip install NRCLex")

from scipy.spatial.distance import cdist


# vocab class — must match step4_combined.py exactly
# pickle needs this resolvable from __main__ at load time
class Vocab:
    def __init__(self):
        self.PAD = "<PAD>"
        self.UNK = "<UNK>"
        self.token2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2token = {0: self.PAD, 1: self.UNK}
        self._next = 2

    def add(self, token: str):
        if token not in self.token2idx:
            self.token2idx[token] = self._next
            self.idx2token[self._next] = token
            self._next += 1

    def encode(self, token: str) -> int:
        return self.token2idx.get(token, self.token2idx[self.UNK])

    def __len__(self):
        return self._next


# model arch — keep in sync with step4_combined.py or everything breaks
class SharedBiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embed_dim, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.output_size = hidden_size * 2  # 512

    def forward(self, token_ids):
        x = self.dropout(self.embedding(token_ids))
        out, _ = self.bilstm(x)
        return self.dropout(out)


class NERHead(nn.Module):
    def __init__(self, input_size, num_ner_labels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.linear = nn.Linear(input_size // 2, num_ner_labels)
        self.crf = CRF(num_ner_labels, batch_first=True)

    def forward(self, encoder_out, ner_labels=None, mask=None):
        emissions = self.linear(self.proj(encoder_out))
        if ner_labels is not None:
            return -self.crf(emissions, ner_labels, mask=mask, reduction="mean")
        return self.crf.decode(emissions, mask=mask)


class SRLHead(nn.Module):
    def __init__(self, input_size, num_srl_labels, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, input_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(input_size // 2, input_size // 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(input_size // 4, num_srl_labels),
        )

    def forward(self, encoder_out):
        logits = self.mlp(encoder_out)
        return logits.argmax(dim=-1)


class DreamscapeMultiTaskBiLSTM(nn.Module):
    def __init__(self, vocab_size, num_ner_labels, num_srl_labels,
                 embed_dim=128, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.encoder = SharedBiLSTMEncoder(
            vocab_size, embed_dim, hidden_size, num_layers, dropout
        )
        self.ner_head = NERHead(self.encoder.output_size, num_ner_labels)
        self.srl_head = SRLHead(self.encoder.output_size, num_srl_labels, dropout)

    def forward(self, token_ids):
        mask = (token_ids != 0).bool()
        enc_out = self.encoder(token_ids)
        return {
            "ner_preds": self.ner_head(enc_out, mask=mask),
            "srl_preds": self.srl_head(enc_out),
            "encoder_out": enc_out,
        }


# tokenizer — identical to training, never diverge from this
# lowercase regex split — must stay in sync with step4
def simple_tokenize(text: str) -> List[str]:
    """same regex as training pipeline"""
    return re.findall(r"[a-z0-9]+", text.lower())


class DreamPipelineModel:
    def __init__(
        self,
        model_path="data_models/step4_bilstm.pt",
        vocab_path="data_models/step4_vocabs.pkl",
        embeddings_path="data_models/step6_enriched_embeddings.npy",
        cluster_labels_path="data_models/step7_cluster_labels.npy",
        theme_data_path="jsons/step9_results.json",
        emotion_data_path="jsons/step10_final_results.json",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading model and data...")
        
        # pickle resolves Vocab as __main__.Vocab — patch it so import works
        # regardless of which module is calling this
        import sys as _sys
        _main_mod = _sys.modules.get("__main__")
        if _main_mod is not None and not hasattr(_main_mod, "Vocab"):
            _main_mod.Vocab = Vocab

        with open(vocab_path, "rb") as f:
            vocabs = pickle.load(f)
        self.token_vocab = vocabs["token"]
        self.ner_vocab = vocabs["ner"]
        self.srl_vocab = vocabs["srl"]
        
        self.model = DreamscapeMultiTaskBiLSTM(
            vocab_size=len(self.token_vocab),
            num_ner_labels=len(self.ner_vocab),
            num_srl_labels=len(self.srl_vocab),
        ).to(self.device)
        
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        
        self.cluster_embeddings = np.load(embeddings_path)
        self.cluster_labels = np.load(cluster_labels_path)
        
        with open(theme_data_path, "r") as f:
            self.theme_data = json.load(f)
        
        with open(emotion_data_path, "r") as f:
            self.emotion_data = json.load(f)
        
        # build lookup by cluster_id for fast access later
        self.cluster_info = {}
        for theme in self.theme_data:
            cid = theme["cluster_id"]
            self.cluster_info[cid] = theme
        
        for emotion in self.emotion_data:
            cid = emotion["cluster_id"]
            if cid in self.cluster_info:
                self.cluster_info[cid]["dominant_emotion"] = emotion.get(
                    "dominant_emotion", "unknown"
                )
                self.cluster_info[cid]["emotion_avg"] = emotion.get(
                    "emotion_avg", {}
                )
        
        # fixed order — don't shuffle this ever
        self.NRC_EMOTIONS = [
            "anger", "anticipation", "disgust", "fear",
            "joy", "sadness", "surprise", "trust"
        ]
        
        print(f"Model loaded on {self.device}")
        print(f"Token vocab: {len(self.token_vocab)}")
        print(f"NER labels: {len(self.ner_vocab)}")
        print(f"SRL labels: {len(self.srl_vocab)}")
        print(f"Cluster embeddings: {self.cluster_embeddings.shape}")
        print(f"Unique clusters: {len(np.unique(self.cluster_labels))}")
        print(f"Theme metadata: {len(self.theme_data)} themes")


# run one forward pass, return NER tags, SRL roles, and mean pooled embedding
def run_inference(model_wrapper, tokens: List[str]) -> dict:
    """bilstm forward pass — get NER tags, SRL roles, mean embedding"""
    
    if not tokens:
        return {
            "ner_tags": [],
            "srl_roles": [],
            "encoder_embedding": np.zeros(512, dtype=np.float32),
            "entities": [],
            "semantic_relations": [],
        }
    
    token_ids = torch.tensor(
        [model_wrapper.token_vocab.encode(t) for t in tokens],
        dtype=torch.long
    ).unsqueeze(0).to(model_wrapper.device)
    
    with torch.no_grad():
        outputs = model_wrapper.model(token_ids)
    
    ner_pred_seq = outputs["ner_preds"][0]  # CRF returns list of seqs
    srl_pred_seq = outputs["srl_preds"][0].cpu().tolist()
    encoder_out = outputs["encoder_out"][0]  # (T, 512)
    
    # mean-pool over non-pad tokens only
    mask = (token_ids[0] != 0).float().unsqueeze(-1)
    summed = (encoder_out * mask).sum(dim=0)
    lengths = mask.sum(dim=0).clamp(min=1)
    mean_embedding = (summed / lengths).cpu().numpy().astype(np.float32)
    
    ner_tags = [
        model_wrapper.ner_vocab.idx2token.get(idx, "O")
        for idx in ner_pred_seq
    ]
    srl_roles = [
        model_wrapper.srl_vocab.idx2token.get(idx, "O")
        for idx in srl_pred_seq
    ]
    
    entities = extract_entities_from_bio(tokens, ner_tags)
    semantic_relations = extract_semantic_relations(tokens, srl_roles)
    
    return {
        "ner_tags": ner_tags,
        "srl_roles": srl_roles,
        "encoder_embedding": mean_embedding,
        "entities": entities,
        "semantic_relations": semantic_relations,
    }


# walk BIO tags and group B/I spans into entity dicts
def extract_entities_from_bio(tokens: List[str], bio_tags: List[str]) -> List[dict]:
    """pull entity spans out of BIO tags"""
    entities = []
    current_entity = None
    
    for i, (token, tag) in enumerate(zip(tokens, bio_tags)):
        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            entity_type = tag[2:]  # strip "B-"
            current_entity = {
                "text": token,
                "type": entity_type,
                "start": i,
                "end": i
            }
        elif tag.startswith("I-") and current_entity:
            current_entity["text"] += " " + token
            current_entity["end"] = i
        else:
            # outside or tag mismatch — flush current
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # don't drop the last one
    if current_entity:
        entities.append(current_entity)
    
    return entities


# convert SRL role sequence into agent-action-target triplets
def extract_semantic_relations(tokens: List[str], srl_roles: List[str]) -> List[dict]:
    """extract agent-action-target triplets from SRL output"""
    relations = []
    current_relation = {"agent": None, "action": None, "target": None}
    
    for i, (token, role) in enumerate(zip(tokens, srl_roles)):
        if role == "V":
            # flush previous relation when we hit a new verb
            if current_relation["action"]:
                relations.append(current_relation.copy())
                current_relation = {"agent": None, "action": None, "target": None}
            current_relation["action"] = token
        elif role == "ARG0":
            current_relation["agent"] = token if not current_relation["agent"] else current_relation["agent"] + " " + token
        elif role == "ARG1":
            current_relation["target"] = token if not current_relation["target"] else current_relation["target"] + " " + token
    
    if current_relation["action"]:
        relations.append(current_relation)
    
    return relations if relations else [{"agent": "Unknown", "action": "Unknown", "target": "Unknown"}]


# score tokens with NRC lexicon, normalize to 8-dim float dict
def compute_nrc_emotion_vector(tokens: List[str]) -> Dict[str, float]:
    """8-dim NRC emotion vector, normalized to sum to 1"""
    
    NRC_EMOTIONS = [
        "anger", "anticipation", "disgust", "fear",
        "joy", "sadness", "surprise", "trust"
    ]
    
    if not _NRC_AVAILABLE or not tokens:
        # uniform fallback — better than zeros downstream
        return {e: 1.0 / len(NRC_EMOTIONS) for e in NRC_EMOTIONS}
    
    # NRCLex v2: pass None as lexicon_file, then load_token_list
    # avoids the os.path.exists crash on long dream texts (errno 63)
    emotion_obj = NRCLex(None)
    emotion_obj.load_token_list(tokens)
    
    raw_scores = emotion_obj.raw_emotion_scores
    
    emotion_vector = {e: raw_scores.get(e, 0.0) for e in NRC_EMOTIONS}
    
    total = sum(emotion_vector.values())
    if total > 0:
        emotion_vector = {e: v / total for e, v in emotion_vector.items()}
    else:
        # no lexicon hits — uniform again
        emotion_vector = {e: 1.0 / len(NRC_EMOTIONS) for e in NRC_EMOTIONS}
    
    return emotion_vector


# concat bilstm + NER multi-hot + SRL multi-hot + emotion into one vector
def build_enriched_embedding(
    model_wrapper,
    encoder_embedding: np.ndarray,
    ner_tags: List[str],
    srl_roles: List[str],
    emotion_vector: Dict[str, float]
) -> np.ndarray:
    """
    concat: [bilstm (512) | NER multi-hot | SRL multi-hot | NRC emotion (8)]
    must match step6 exactly or cluster assignment will be garbage
    """
    
    NUM_NER = len(model_wrapper.ner_vocab)
    NUM_SRL = len(model_wrapper.srl_vocab)
    NUM_EMO = 8
    
    bilstm_vec = encoder_embedding  # already 512-d from mean-pooling
    
    # skip PAD=0, UNK=1 — no signal there
    ner_multihot = np.zeros(NUM_NER, dtype=np.float32)
    for tag in ner_tags:
        idx = model_wrapper.ner_vocab.encode(tag)
        if idx >= 2:
            ner_multihot[idx] = 1.0
    
    srl_multihot = np.zeros(NUM_SRL, dtype=np.float32)
    for role in srl_roles:
        idx = model_wrapper.srl_vocab.encode(role)
        if idx >= 2:
            srl_multihot[idx] = 1.0
    
    emo_vec = np.array([
        emotion_vector.get(e, 0.0)
        for e in model_wrapper.NRC_EMOTIONS
    ], dtype=np.float32)
    
    enriched = np.concatenate([
        bilstm_vec,
        ner_multihot,
        srl_multihot,
        emo_vec
    ])
    
    return enriched


# compare enriched embedding to cluster centroids, return closest match
def find_nearest_cluster(
    model_wrapper,
    enriched_embedding: np.ndarray
) -> Tuple[int, str, dict]:
    """nearest centroid by euclidean dist — returns (cluster_id, label, meta)"""
    
    unique_clusters = np.unique(model_wrapper.cluster_labels)
    cluster_centroids = []
    cluster_ids = []
    
    for cid in unique_clusters:
        if cid == -1:  # HDBSCAN noise — skip
            continue
        mask = model_wrapper.cluster_labels == cid
        centroid = model_wrapper.cluster_embeddings[mask].mean(axis=0)
        cluster_centroids.append(centroid)
        cluster_ids.append(int(cid))
    
    if not cluster_centroids:
        return -1, "Unknown", {}
    
    cluster_centroids = np.array(cluster_centroids)
    
    distances = cdist(
        enriched_embedding.reshape(1, -1),
        cluster_centroids,
        metric='euclidean'
    )[0]
    
    nearest_idx = np.argmin(distances)
    nearest_cluster_id = cluster_ids[nearest_idx]
    
    cluster_meta = model_wrapper.cluster_info.get(
        nearest_cluster_id,
        {"topic_label": "Unknown", "keywords": []}
    )
    
    return nearest_cluster_id, cluster_meta.get("topic_label", "Unknown"), cluster_meta


# end-to-end: text in, structured json out
def run_production_pipeline(dream_text: str, model_wrapper: DreamPipelineModel) -> dict:
    """
    full pipeline: tokenize → bilstm → nrc → enrich → cluster → format
    
    args:
        dream_text: raw dream narrative
        model_wrapper: loaded model + data
    
    returns:
        structured json with theme, emotion, entities, etc.
    """
    
    tokens = simple_tokenize(dream_text)
    
    if not tokens:
        return {
            "error": "No valid tokens found in input text",
            "raw_text": dream_text
        }
    
    inference_out = run_inference(model_wrapper, tokens)
    
    emotion_vector = compute_nrc_emotion_vector(tokens)
    
    enriched_emb = build_enriched_embedding(
        model_wrapper,
        inference_out["encoder_embedding"],
        inference_out["ner_tags"],
        inference_out["srl_roles"],
        emotion_vector
    )
    
    cluster_id, topic_label, cluster_meta = find_nearest_cluster(
        model_wrapper,
        enriched_emb
    )
    
    dominant_emotion = cluster_meta.get("dominant_emotion", "unknown")
    emotion_avg = cluster_meta.get("emotion_avg", {})
    
    keywords = cluster_meta.get("keywords", [])
    
    # group by entity type
    entities_by_type = {}
    for ent in inference_out["entities"]:
        etype = ent["type"]
        if etype not in entities_by_type:
            entities_by_type[etype] = []
        entities_by_type[etype].append(ent["text"])
    
    # priority order for key entities shown to user
    key_entities = []
    for etype in ["PER", "LOC", "OBJ", "SURREAL", "BODY"]:
        key_entities.extend(entities_by_type.get(etype, []))
    key_entities = list(dict.fromkeys(key_entities))  # dedup, preserve order
    
    result = {
        "raw_text": dream_text,
        "tokens": tokens,
        "cluster_id": int(cluster_id),
        "topic_cluster": topic_label,
        "dominant_emotion": dominant_emotion.capitalize(),
        "emotion_vector": emotion_vector,
        "cluster_emotion_avg": emotion_avg,
        "key_entities": key_entities if key_entities else ["Unknown"],
        "entities_by_type": entities_by_type,
        "semantic_relations": inference_out["semantic_relations"],
        "cluster_keywords": keywords,
        "ner_tags": inference_out["ner_tags"],
        "srl_roles": inference_out["srl_roles"],
        "enriched_embedding_dim": len(enriched_emb),
        "summary": (
            f"Dream classified as '{topic_label}' (cluster {cluster_id}) "
            f"with dominant emotion '{dominant_emotion}'. "
            f"Top cluster emotion score: {emotion_avg.get(dominant_emotion.lower(), 0.0):.2f}."
        )
    }
    
    return result


# CLI entry point — reads from argv or stdin, prints + saves result
def main():
    import sys
    
    print("="*60)
    print("  Dreamscape Mapper — Production Pipeline")
    print("="*60)
    
    model_wrapper = DreamPipelineModel()
    
    print("\n" + "="*60)
    print("  Ready to analyze dreams!")
    print("="*60 + "\n")
    
    if len(sys.argv) > 1:
        dream_input = " ".join(sys.argv[1:])
    else:
        dream_input = input("Enter dream narrative: ").strip()
    
    if not dream_input:
        print("Error: Empty input")
        return
    
    print("\nProcessing...")
    result = run_production_pipeline(dream_input, model_wrapper)
    
    print("\n" + "="*60)
    print("  ANALYSIS RESULTS")
    print("="*60)
    print(json.dumps(result, indent=2))
    
    output_file = "dream_analysis_output.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()