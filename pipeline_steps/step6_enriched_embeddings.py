# step 6: enriched embedding construction
#
# inputs:
#   dream_annotations.json         — segments from steps 1-3
#   step4_bilstm.pt                — trained NER + SRL weights
#   step4_vocabs.pkl               — token / NER / SRL vocabs
#
# outputs:
#   step6_enriched_embeddings.npy  — (N, D) float32 matrix
#   step6_metadata.json            — text + labels per row
#
# vector layout (D = 512 + NER_size + SRL_size + 8):
#   [0        : 512           ]  bilstm mean-pooled embedding
#   [512      : 512+NER_size  ]  NER entity-type presence (multi-hot)
#   [512+NER  : 512+NER+SRL   ]  SRL role presence        (multi-hot)
#   [512+NER+SRL : end        ]  NRC emotion vector       (8-dim)
#
# NER_size and SRL_size come from the loaded vocab — never hardcode these

import json
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence

from step4_combined import (
    DreamscapeMultiTaskBiLSTM,
    Vocab,
    NRC_EMOTIONS,
    simple_tokenize,
)

DIM_BILSTM  = 512
DIM_EMOTION = len(NRC_EMOTIONS)   # 8


# load step4 model + vocabs from disk, set to eval mode
def load_model(device):
    print("Loading Step 4 model and vocabs...")
    with open("data_models/step4_vocabs.pkl", "rb") as f:
        vocabs = pickle.load(f)
    token_vocab = vocabs["token"]
    ner_vocab   = vocabs["ner"]
    srl_vocab   = vocabs["srl"]

    model = DreamscapeMultiTaskBiLSTM(
        vocab_size     = len(token_vocab),
        num_ner_labels = len(ner_vocab),
        num_srl_labels = len(srl_vocab),
    ).to(device)
    model.load_state_dict(
        torch.load("data_models/step4_bilstm.pt", map_location=device, weights_only=True)
    )
    model.eval()

    print(f"  Token vocab  : {len(token_vocab)}")
    print(f"  NER labels   : {len(ner_vocab)}  (incl. <PAD>, <UNK>)")
    print(f"  SRL labels   : {len(srl_vocab)}  (incl. <PAD>, <UNK>)")
    return model, token_vocab, ner_vocab, srl_vocab


# mean-pool encoder output over non-PAD tokens, returns (B, 512) numpy array
def build_bilstm_embedding(encoder, padded, device):
    """mean-pool bilstm hidden states over non-PAD tokens — returns (B, 512)"""
    with torch.no_grad():
        enc_out = encoder(padded)                          # (B, T, 512)
    mask    = (padded != 0).float().unsqueeze(-1)          # (B, T, 1)
    summed  = (enc_out * mask).sum(dim=1)                  # (B, 512)
    lengths = mask.sum(dim=1).clamp(min=1)                 # (B, 1)
    return (summed / lengths).cpu().numpy().astype(np.float32)


# binary vector over NER vocab indicating which entity types appeared
def build_ner_multihot(ner_pred_seq, num_ner_labels):
    """
    multi-hot over vocab size (read at runtime, not hardcoded)
    skips PAD=0 and UNK=1 — no entity signal there
    """
    vec = np.zeros(num_ner_labels, dtype=np.float32)
    for idx in ner_pred_seq:
        if idx >= 2:    # skip PAD=0, UNK=1
            vec[idx] = 1.0
    return vec


# binary vector over SRL vocab indicating which roles appeared
def build_srl_multihot(srl_pred_seq, num_srl_labels):
    """same as NER multi-hot — skips PAD=0 and UNK=1"""
    vec = np.zeros(num_srl_labels, dtype=np.float32)
    for idx in srl_pred_seq:
        if idx >= 2:
            vec[idx] = 1.0
    return vec


# pull NRC scores out of dict in fixed order, fallback to uniform
def build_emotion_vector(emotion_dict):
    """
    8-dim NRC vector in fixed NRC_EMOTIONS order
    falls back to uniform if keys are missing
    """
    vec = np.array(
        [emotion_dict.get(emo, 1.0 / DIM_EMOTION) for emo in NRC_EMOTIONS],
        dtype=np.float32,
    )
    return vec


# why we guard against empty segments:
# some segments in dream_annotations.json have empty token lists
# (very short sentences that simple_tokenize reduces to []).
# pad_sequence on an empty tensor makes CRF's mask[:,0] = False → ValueError.
# fix: filter empties before batching, store zero-vector placeholder so
# output row count still matches len(segments)

# zero placeholder for empty segments so matrix row count stays correct
def make_zero_vector(dim):
    return np.zeros(dim, dtype=np.float32)


def build_enriched_embeddings(
    data_path   = "jsons/dream_annotations.json",
    output_emb  = "data_models/step6_enriched_embeddings.npy",
    output_meta = "jsons/step6_metadata.json",
    batch_size  = 128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dream_annotations.json...")
    with open(data_path, "r") as f:
        segments = json.load(f)
    print(f"  Segments loaded: {len(segments)}")

    model, token_vocab, ner_vocab, srl_vocab = load_model(device)
    encoder = model.encoder

    # dims from vocab — never hardcode these
    NUM_NER = len(ner_vocab)
    NUM_SRL = len(srl_vocab)
    DIM_TOTAL = DIM_BILSTM + NUM_NER + NUM_SRL + DIM_EMOTION

    print(f"\n{'='*55}")
    print(f"  STEP 6 — Enriched Embedding Construction")
    print(f"  Device : {device}")
    print(f"  Vector : {DIM_TOTAL} dims  "
          f"(BiLSTM {DIM_BILSTM} + NER {NUM_NER} + "
          f"SRL {NUM_SRL} + Emotion {DIM_EMOTION})")
    print(f"{'='*55}\n")

    all_token_lists = []
    for seg in segments:
        toks = seg.get("tokens") or simple_tokenize(seg.get("text", ""))
        all_token_lists.append(toks)

    empty_count = sum(1 for t in all_token_lists if len(t) == 0)
    if empty_count:
        print(f"  Warning: {empty_count} empty segments found — "
              f"will be stored as zero vectors and skipped by CRF.")

    all_enriched = []
    all_metadata = []
    skipped      = 0

    print(f"Building enriched embeddings in batches of {batch_size}...")
    for batch_start in tqdm(range(0, len(segments), batch_size)):
        batch_end  = min(batch_start + batch_size, len(segments))
        batch_segs = segments[batch_start:batch_end]
        batch_toks = all_token_lists[batch_start:batch_end]

        valid_indices  = [i for i, t in enumerate(batch_toks) if len(t) > 0]
        empty_indices  = [i for i, t in enumerate(batch_toks) if len(t) == 0]

        batch_enriched = [None] * len(batch_segs)
        batch_meta     = [None] * len(batch_segs)

        # empty segments get zero vectors — keeps matrix row count correct
        for i in empty_indices:
            seg = batch_segs[i]
            batch_enriched[i] = make_zero_vector(DIM_TOTAL)
            batch_meta[i] = {
                "text":           seg.get("text", ""),
                "tokens":         [],
                "entities":       seg.get("entities", []),
                "emotion_vector": seg.get("emotion_vector", {}),
                "coref":          seg.get("coref", []),
                "ner_tags":       [],
                "srl_roles":      [],
                "empty":          True,
            }
            skipped += 1

        if valid_indices:
            valid_toks = [batch_toks[i] for i in valid_indices]
            valid_segs = [batch_segs[i] for i in valid_indices]

            token_id_tensors = [
                torch.tensor(
                    [token_vocab.encode(t) for t in toks], dtype=torch.long
                )
                for toks in valid_toks
            ]
            padded = pad_sequence(
                token_id_tensors, batch_first=True, padding_value=0
            ).to(device)

            bilstm_embs = build_bilstm_embedding(encoder, padded, device)

            with torch.no_grad():
                outputs = model(padded)
            ner_preds_batch = outputs["ner_preds"]   # list of lists (CRF)
            srl_preds_batch = outputs["srl_preds"]   # (B, T) tensor

            for j, (seg, toks) in enumerate(zip(valid_segs, valid_toks)):
                ner_seq = ner_preds_batch[j]
                srl_seq = srl_preds_batch[j].cpu().tolist()

                ner_vec  = build_ner_multihot(ner_seq, NUM_NER)
                srl_vec  = build_srl_multihot(srl_seq, NUM_SRL)
                emo_vec  = build_emotion_vector(seg.get("emotion_vector", {}))

                enriched = np.concatenate([
                    bilstm_embs[j],
                    ner_vec,
                    srl_vec,
                    emo_vec,
                ]).astype(np.float32)

                orig_i = valid_indices[j]
                batch_enriched[orig_i] = enriched
                batch_meta[orig_i] = {
                    "text":           seg.get("text", ""),
                    "tokens":         toks,
                    "entities":       seg.get("entities", []),
                    "emotion_vector": seg.get("emotion_vector", {}),
                    "coref":          seg.get("coref", []),
                    "ner_tags":  [ner_vocab.idx2token.get(idx, "O")
                                  for idx in ner_seq],
                    "srl_roles": [srl_vocab.idx2token.get(idx, "O")
                                  for idx in srl_seq],
                    "empty":     False,
                }

        all_enriched.extend(batch_enriched)
        all_metadata.extend(batch_meta)

    embedding_matrix = np.stack(all_enriched, axis=0)
    np.save(output_emb, embedding_matrix)

    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False)

    print(f"\nDone.")
    print(f"  Embedding matrix : {output_emb}  "
          f"shape={embedding_matrix.shape}")
    print(f"  Metadata file    : {output_meta}  "
          f"({len(all_metadata)} records)")
    print(f"  Empty segments   : {skipped} (stored as zero vectors)")

    # sanity check — spot check first 3 non-empty
    print("\nSample enriched vectors (first 3 non-empty segments):")
    shown = 0
    for i, m in enumerate(all_metadata):
        if m.get("empty"):
            continue
        vec = embedding_matrix[i]
        ner_active = [ner_vocab.idx2token.get(j, "?")
                      for j in range(NUM_NER) if vec[512 + j] > 0]
        srl_active = [srl_vocab.idx2token.get(j, "?")
                      for j in range(NUM_SRL)
                      if vec[512 + NUM_NER + j] > 0]
        top_emo = sorted(
            zip(NRC_EMOTIONS, vec[512 + NUM_NER + NUM_SRL:].tolist()),
            key=lambda x: -x[1]
        )[:3]
        print(f"\n  [{i}] {m['text'][:70]}")
        print(f"       BiLSTM norm : {np.linalg.norm(vec[:512]):.4f}")
        print(f"       NER active  : {ner_active}")
        print(f"       SRL active  : {srl_active}")
        print(f"       Top emotions: {top_emo}")
        shown += 1
        if shown == 3:
            break

    print(f"\n{'='*55}")
    print("Step 6 complete.")
    print("\nOutputs for Step 7 (UMAP + HDBSCAN):")
    print(f"  step6_enriched_embeddings.npy  — shape {embedding_matrix.shape}")
    print(f"  step6_metadata.json            — {len(all_metadata)} records")
    print(f"\nVector layout:")
    print(f"  [0        : 512          ]  BiLSTM contextual embedding")
    print(f"  [512      : {512+NUM_NER:<4}         ]  "
          f"NER entity presence  ({NUM_NER}-dim multi-hot)")
    print(f"  [{512+NUM_NER:<4}      : {512+NUM_NER+NUM_SRL:<4}         ]  "
          f"SRL role presence    ({NUM_SRL}-dim multi-hot)")
    print(f"  [{512+NUM_NER+NUM_SRL:<4}      : {DIM_TOTAL:<4}         ]  "
          f"NRC emotion          ({DIM_EMOTION}-dim normalised)")

    return embedding_matrix, all_metadata


if __name__ == "__main__":
    build_enriched_embeddings(
        data_path   = "jsons/dream_annotations.json",
        output_emb  = "data_models/step6_enriched_embeddings.npy",
        output_meta = "jsons/step6_metadata.json",
        batch_size  = 128,
    )
