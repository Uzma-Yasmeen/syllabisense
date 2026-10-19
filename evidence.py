"""
The evidence ANN, in numpy.

Identical arithmetic to the trained Keras model - the same weights, the same activations - but
without importing TensorFlow. The network is 115,841 parameters; running it needs three matrix
multiplications, not a deep learning framework.

Why it matters: TensorFlow is 316 MB of the application's memory and most of its start-up time,
and it only publishes wheels for some Python versions, which is enough to stop a deployment dead.
numpy is installed everywhere.

    from evidence import load_evidence_model
    net = load_evidence_model()          # falls back to Keras if the export is missing
    probabilities = net(X)               # X: (n, 387) float32
"""
import json
import os

import numpy as np

EXPORT = "models/evidence_ann.npz"
KERAS = "models/evidence_ann.keras"


def _relu(x):
    return np.maximum(x, 0.0, out=x)


def _sigmoid(x):
    # the stable form: exp() of a large positive number overflows, of a large negative one does not
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


ACTIVATIONS = {"relu": _relu, "sigmoid": _sigmoid, "linear": lambda x: x}


class EvidenceANN:
    """The exported network. Call it with a (n, 387) array to get n probabilities."""

    def __init__(self, path=EXPORT):
        d = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(d["meta"]))
        self.layers = [(d[f"W{i}"], d[f"b{i}"], ACTIVATIONS[m["activation"]])
                       for i, m in enumerate(self.meta)]
        self.input_dim = int(self.meta[0]["shape"][0])
        self.n_parameters = sum(int(W.size + b.size) for W, b, _ in self.layers)

    def __call__(self, X):
        h = np.ascontiguousarray(X, dtype=np.float32)
        for W, b, act in self.layers:
            h = act(h @ W + b)
        return h.ravel()

    # so it can stand in for a Keras model without the caller knowing
    def predict(self, X, verbose=0):
        return self(X).reshape(-1, 1)

    def __repr__(self):
        shape = " -> ".join([str(self.input_dim)]
                            + [str(m["shape"][1]) for m in self.meta])
        return f"EvidenceANN({shape}, {self.n_parameters:,} parameters, numpy)"


def load_evidence_model(export=EXPORT, keras=KERAS):
    """The numpy network when it has been exported, otherwise the Keras one.

    Keeping the fallback means a checkout that has only ever trained the model still runs; the
    export is an optimisation, not a new source of truth.
    """
    if os.path.exists(export):
        return EvidenceANN(export)
    import tensorflow as tf
    return tf.keras.models.load_model(keras)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    net = load_evidence_model()
    print(net)
    x = np.zeros((3, net.input_dim), np.float32)
    print("output on three zero vectors:", net(x))
