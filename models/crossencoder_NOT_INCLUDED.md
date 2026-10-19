# Cross-encoder weights are not in this repository

`model.safetensors` is 87 MB. The application never loads it — the transformer was trained as a
measured comparison against the evidence ANN, not as a component of the system.

To reproduce it (about an hour on a CPU):

    python train_crossencoder.py --epochs 2 --max-train 9000

Its measured results, graphs and confusion matrices are in `results/crossencoder/`.
