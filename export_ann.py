"""
Export the evidence ANN's weights so the application can run it without TensorFlow.

The network is three dense layers - 387 to 256 to 64 to 1, with ReLU twice and a sigmoid at the
end, and dropout which does nothing at prediction time. That is a handful of matrix
multiplications. Keeping TensorFlow installed to perform them costs 316 MB of memory, several
seconds of start-up, and a hard dependency on a Python version TensorFlow happens to publish
wheels for - which is what broke the first deployment.

    python export_ann.py

Writes results_tfidf/evidence_ann.npz, read by evidence.py.
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.stdout.reconfigure(encoding="utf-8")
SRC = "results_tfidf/evidence_ann.keras"
OUT = "results_tfidf/evidence_ann.npz"


def main():
    model = tf.keras.models.load_model(SRC)
    dense = [l for l in model.layers if l.get_weights()]
    arrays, meta = {}, []
    for i, layer in enumerate(dense):
        w, b = layer.get_weights()
        arrays[f"W{i}"] = w.astype(np.float32)
        arrays[f"b{i}"] = b.astype(np.float32)
        act = getattr(layer, "activation", None)
        meta.append({"name": layer.name, "shape": list(w.shape),
                     "activation": act.__name__ if act else "linear"})
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(OUT, **arrays)

    params = sum(int(np.prod(arrays[f"W{i}"].shape)) + arrays[f"b{i}"].size
                 for i in range(len(dense)))
    print(f"exported {len(dense)} layers, {params:,} parameters")
    for m in meta:
        print(f"  {m['name']:<12} {m['shape'][0]:>4} -> {m['shape'][1]:<4} {m['activation']}")
    print(f"\n  {SRC}: {os.path.getsize(SRC) / 1024:.0f} KB")
    print(f"  {OUT}: {os.path.getsize(OUT) / 1024:.0f} KB")

    # the export is only useful if it predicts exactly what the original does
    from evidence import EvidenceANN
    net = EvidenceANN(OUT)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, dense[0].get_weights()[0].shape[0])).astype(np.float32)
    a = model.predict(X, verbose=0).ravel()
    b = net(X)
    print(f"\n  agreement on 500 random inputs:")
    print(f"    largest difference : {np.abs(a - b).max():.3e}")
    print(f"    identical to 6 dp  : {np.allclose(a, b, atol=1e-6)}")


if __name__ == "__main__":
    main()
