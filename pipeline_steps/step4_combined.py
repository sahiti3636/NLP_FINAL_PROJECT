# step 4: multi-task bilstm (NER + SRL)
# step 5 (emotion) is handled by NRC in steps 1-3 now, not here
#
# outputs:
#   step4_bilstm.pt   — NER + SRL weights
#   step4_vocabs.pkl  — token / NER / SRL vocabs
#
# install deps:
#   pip install torch torchcrf

import json
import re
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchcrf import CRF

# matches steps1_3_pipeline.py — fixed order, never change
NRC_EMOTIONS = ["anger", "anticipation", "disgust", "fear",
                "joy", "sadness", "surprise", "trust"]
NUM_EMOTIONS = len(NRC_EMOTIONS)   # 8


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


# lowercase regex tokenizer — identical to steps1_3, must never diverge
def simple_tokenize(text: str):
    """identical to steps1_3_pipeline.py — must never diverge"""
    return re.findall(r"[a-z0-9]+", text.lower())


# init all three vocabs and add fixed NER/SRL tag sets before seeing data
def build_vocabs(annotated_segments):
    token_vocab, ner_vocab, srl_vocab = Vocab(), Vocab(), Vocab()

    for tag in [
        "O",
        "B-PER", "I-PER",
        "B-LOC", "I-LOC",
        "B-GPE", "I-GPE",
        "B-ORG", "I-ORG",
        "B-OBJ", "I-OBJ",
        "B-SURREAL", "I-SURREAL",
        "B-EVT", "I-EVT",
        "B-MISC", "I-MISC",
    ]:
        ner_vocab.add(tag)

    for role in [
        "O", "V",
        "ARG0", "ARG1", "ARG2", "ARG3", "ARG4",
        "ARGM-LOC", "ARGM-TMP", "ARGM-MNR",
        "ARGM-CAU", "ARGM-NEG", "ARGM-MOD",
        "ARGM-DIR", "ARGM-ADV",
    ]:
        srl_vocab.add(role)

    for seg in annotated_segments:
        for tok in seg.get("tokens", []):
            token_vocab.add(tok)

    return token_vocab, ner_vocab, srl_vocab


LABEL_MAP = {
    "PERSON": "PER", "LOCATION": "LOC", "GPE": "GPE", "FAC": "LOC",
    "ORG": "ORG", "NORP": "GPE", "PRODUCT": "OBJ", "WORK_OF_ART": "MISC",
    "EVENT": "EVT", "MISC": "MISC", "OBJ": "OBJ",
    "SURREAL": "SURREAL", "PER": "PER", "LOC": "LOC",
}


# align entity spans to token positions, produce BIO label id sequence
def align_ner_labels(tokens, entities, ner_vocab):
    label_seq = ["O"] * len(tokens)
    for ent_text, ent_label in entities:
        ent_tokens = ent_text.lower().split()
        n          = len(ent_tokens)
        normalised = LABEL_MAP.get(ent_label.upper(), "MISC")
        b_tag, i_tag = f"B-{normalised}", f"I-{normalised}"
        if b_tag not in ner_vocab.token2idx:
            b_tag, i_tag = "B-MISC", "I-MISC"
        for i in range(len(tokens) - n + 1):
            if tokens[i: i + n] == ent_tokens:
                label_seq[i] = b_tag
                for j in range(1, n):
                    if i + j < len(label_seq):
                        label_seq[i + j] = i_tag
                break
    return [ner_vocab.encode(l) for l in label_seq]


# align SRL verb+arg spans to token positions, produce role id sequence
def align_srl_labels(tokens, relations, srl_vocab):
    label_seq = ["O"] * len(tokens)
    for rel in relations:
        verb       = rel.get("verb", "").lower()
        verb_index = rel.get("verb_index", None)
        if verb_index is not None and verb_index < len(tokens):
            label_seq[verb_index] = "V"
        else:
            for i, tok in enumerate(tokens):
                if tok == verb:
                    label_seq[i] = "V"
                    break
        for arg in rel.get("args", []):
            role      = arg.get("role", "").upper()
            if role not in srl_vocab.token2idx:
                continue
            span_text = arg.get("span", "").lower().split()
            n         = len(span_text)
            for i in range(len(tokens) - n + 1):
                if tokens[i: i + n] == span_text:
                    for j in range(n):
                        if label_seq[i + j] == "O":
                            label_seq[i + j] = role
                    break
    return [srl_vocab.encode(l) for l in label_seq]


class DreamSegmentDataset(Dataset):
    def __init__(self, annotated_segments, token_vocab, ner_vocab, srl_vocab):
        self.samples = []
        for seg in annotated_segments:
            tokens = seg.get("tokens", [])
            if not tokens:
                continue
            token_ids = torch.tensor([token_vocab.encode(t) for t in tokens], dtype=torch.long)
            ner_ids   = torch.tensor(align_ner_labels(tokens, seg.get("entities", []), ner_vocab), dtype=torch.long)
            srl_ids   = torch.tensor(align_srl_labels(tokens, seg.get("relations", []), srl_vocab), dtype=torch.long)
            self.samples.append((token_ids, ner_ids, srl_ids))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# pad variable-length sequences in a batch to the same length
def collate_fn(batch):
    token_ids, ner_labels, srl_labels = zip(*batch)
    return (
        pad_sequence(token_ids,  batch_first=True, padding_value=0),
        pad_sequence(ner_labels, batch_first=True, padding_value=0),
        pad_sequence(srl_labels, batch_first=True, padding_value=0),
    )


class SharedBiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm      = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                                   batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout     = nn.Dropout(dropout)
        self.output_size = hidden_size * 2   # 512

    def forward(self, token_ids):
        x = self.dropout(self.embedding(token_ids))
        out, _ = self.bilstm(x)
        return self.dropout(out)


class NERHead(nn.Module):
    def __init__(self, input_size, num_ner_labels):
        super().__init__()
        self.proj   = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.linear = nn.Linear(input_size // 2, num_ner_labels)
        self.crf    = CRF(num_ner_labels, batch_first=True)

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
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, encoder_out, srl_labels=None):
        logits = self.mlp(encoder_out)
        if srl_labels is not None:
            B, T, C = logits.shape
            return self.loss_fn(logits.view(B * T, C), srl_labels.view(B * T))
        return logits.argmax(dim=-1)


class DreamscapeMultiTaskBiLSTM(nn.Module):
    def __init__(self, vocab_size, num_ner_labels, num_srl_labels,
                 embed_dim=128, hidden_size=256, num_layers=2, dropout=0.3,
                 ner_weight=1.0, srl_weight=0.8):
        super().__init__()
        self.encoder    = SharedBiLSTMEncoder(vocab_size, embed_dim, hidden_size, num_layers, dropout)
        self.ner_head   = NERHead(self.encoder.output_size, num_ner_labels)
        self.srl_head   = SRLHead(self.encoder.output_size, num_srl_labels, dropout)
        self.ner_weight = ner_weight
        self.srl_weight = srl_weight

    def forward(self, token_ids, ner_labels=None, srl_labels=None):
        mask    = (token_ids != 0).bool()
        enc_out = self.encoder(token_ids)
        if ner_labels is not None and srl_labels is not None:
            ner_loss = self.ner_head(enc_out, ner_labels, mask=mask)
            srl_loss = self.srl_head(enc_out, srl_labels)
            total    = self.ner_weight * ner_loss + self.srl_weight * srl_loss
            return {"total": total, "ner": ner_loss, "srl": srl_loss}
        return {
            "ner_preds":   self.ner_head(enc_out, mask=mask),
            "srl_preds":   self.srl_head(enc_out),
            "encoder_out": enc_out,
        }


# one train or eval pass over a dataloader — returns avg losses
def run_epoch(model, loader, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total = ner_sum = srl_sum = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for token_ids, ner_labels, srl_labels in loader:
            token_ids  = token_ids.to(device)
            ner_labels = ner_labels.to(device)
            srl_labels = srl_labels.to(device)
            if train:
                optimizer.zero_grad()
            losses = model(token_ids, ner_labels, srl_labels)
            if train:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            total   += losses["total"].item()
            ner_sum += losses["ner"].item()
            srl_sum += losses["srl"].item()
    n = len(loader)
    return total / n, ner_sum / n, srl_sum / n


# full step 4 training run — builds vocabs, trains model, saves weights + vocabs
def train_step4(data_path="jsons/dream_annotations.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  STEP 4 — Multi-Task BiLSTM (NER + SRL)")
    print(f"  Device: {device}")
    print(f"{'='*55}")

    with open(data_path, "r") as f:
        results = json.load(f)

    token_vocab, ner_vocab, srl_vocab = build_vocabs(results)
    dataset = DreamSegmentDataset(results, token_vocab, ner_vocab, srl_vocab)

    train_size = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = DreamscapeMultiTaskBiLSTM(
        vocab_size     = len(token_vocab),
        num_ner_labels = len(ner_vocab),
        num_srl_labels = len(srl_vocab),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val, patience_counter = float("inf"), 0
    for epoch in range(1, 51):
        tr_total, tr_ner, tr_srl = run_epoch(model, train_loader, optimizer, device, train=True)
        va_total, va_ner, va_srl = run_epoch(model, val_loader,   optimizer, device, train=False)
        scheduler.step(va_total)
        print(f"Epoch {epoch:>2}/50 | "
              f"Train {tr_total:.4f} (NER {tr_ner:.4f} SRL {tr_srl:.4f}) | "
              f"Val {va_total:.4f} (NER {va_ner:.4f} SRL {va_srl:.4f})")
        if va_total < best_val:
            best_val, patience_counter = va_total, 0
            torch.save(model.state_dict(), "data_models/step4_bilstm.pt")
            print("   Saved best model → step4_bilstm.pt")
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print("   Early stopping.")
                break

    with open("data_models/step4_vocabs.pkl", "wb") as f:
        pickle.dump({"token": token_vocab, "ner": ner_vocab, "srl": srl_vocab}, f)
    print("Step 4 complete. Artefacts: step4_bilstm.pt | step4_vocabs.pkl")
    return token_vocab, ner_vocab, srl_vocab


if __name__ == "__main__":
    train_step4("jsons/dream_annotations.json")
    print("\nDone. Next: Step 6 — Enriched Embedding Construction.")
    print("Inputs for Step 6:")
    print("  step4_bilstm.pt   — encoder for base embeddings")
    print("  step4_vocabs.pkl  — token/NER/SRL vocabs")
    print("  dream_annotations.json — contains emotion_vector (8-dim NRC) per segment")
