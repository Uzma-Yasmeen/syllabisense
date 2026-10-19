"""
Turn (topic, page) text pairs into numeric features for the ANN.

Each text is encoded with a pretrained sentence embedding model (BAAI/bge-small-en-v1.5,
384 dimensions, mean pooling). The model is used as a frozen feature extractor - no fine-tuning -
so the trainable part is the ANN in train_ann.py.

Features per example (1154 numbers):
  topic embedding            384
  page embedding             384
  topic * page (elementwise)  384   -> lets the network see where the two agree
  cosine similarity            1
  word overlap fraction        1
  page length (scaled)         1
Output: dataset/embed_<split>.npz
"""
import csv
import hashlib
import json
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")
MODEL = "BAAI/bge-small-en-v1.5"
BATCH = 32
MAXLEN = 256

tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModel.from_pretrained(MODEL).eval()
torch.set_num_threads(8)


@torch.no_grad()
def encode(texts):
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i in range(0, len(texts), BATCH):
        b = texts[i:i + BATCH]
        enc = tok(b, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        h = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out[i:i + len(b)] = torch.nn.functional.normalize(v, dim=-1).numpy()
        if i % (BATCH * 40) == 0:
            print(f"    {i}/{len(texts)}", flush=True)
    return out


def main():
    splits = {s: list(csv.DictReader(open(f"dataset/pairs_{s}.csv", encoding="utf8")))
              for s in ("train", "val", "test")}
    # encode every distinct text once
    texts, index = [], {}
    for rows in splits.values():
        for r in rows:
            for t in (r["topic"], r["text"]):
                k = hashlib.md5(t.encode()).hexdigest()
                if k not in index:
                    index[k] = len(texts)
                    texts.append(t)
    print(f"distinct texts to encode: {len(texts)}")
    E = encode(texts)

    def vec(t):
        return E[index[hashlib.md5(t.encode()).hexdigest()]]

    for name, rows in splits.items():
        a = np.stack([vec(r["topic"]) for r in rows])
        b = np.stack([vec(r["text"]) for r in rows])
        cos = (a * b).sum(1, keepdims=True)
        ov, ln = [], []
        for r in rows:
            tw = {w for w in r["topic"].lower().split() if len(w) > 3}
            pw = r["text"].lower()
            ov.append([sum(1 for w in tw if w in pw) / max(1, len(tw))])
            ln.append([min(len(r["text"]), 1500) / 1500])
        X = np.hstack([a, b, a * b, cos, np.array(ov, np.float32), np.array(ln, np.float32)]).astype(np.float32)
        y = np.array([int(r["label"]) for r in rows], np.int8)
        np.savez_compressed(f"dataset/embed_{name}.npz", X=X, y=y,
                            domain=np.array([r["domain"] for r in rows]),
                            kind=np.array([r["kind"] for r in rows]))
        print(f"  {name}: X={X.shape} positives={int(y.sum())}/{len(y)}")
    json.dump({"model": MODEL, "dims": int(X.shape[1]), "max_tokens": MAXLEN},
              open("dataset/embed_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
