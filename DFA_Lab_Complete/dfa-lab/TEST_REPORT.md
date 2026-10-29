# Local verification report

Verified on 2026-10-29 with Python 3.12.14, NumPy 2.3.5, and Flask 3.1.2 on Windows.

## Training and evaluation

Real training completed for 15 epochs; the lowest validation-loss checkpoint was exported and reloaded before evaluation.

| Evaluation | Count | Accuracy |
|---|---:|---:|
| Held-out in-range strings | 900 | 100.00% |
| Held-out longer strings | 1,000 | 100.00% |
| Exhaustive lengths 0–10 (overlaps training) | 2,047 | 100.00% |

## Automated verification

All eight unittest checks passed:

1. Exact DFA vs. an independent suffix definition on every binary string of lengths 0–10, including full state traces.
2. Invalid-input validation, including non-string types, whitespace, Unicode lookalikes, and excessive length.
3. Analytical RNN gradients vs. numerical finite differences for every parameter of a small model.
4. Export/reload equality and batch-padding invariance, including empty and 256-character strings.
5. Dataset labels, row counts, uniqueness, and disjoint splits.
6. Flask page/assets/metrics routes, prediction responses, invalid JSON, invalid types, length boundaries, and oversized requests.
7. Missing-model behavior: exact validation stays available and neural result is null.
8. Recomputed held-out metrics match the saved report and exceed the 95% test threshold.

JavaScript syntax validation passed. Notebook JSON and executable-cell Python syntax were checked; its embedded DFA, RNN, and training source exactly match the locally executed modules. The notebook was not executed inside Google's hosted Colab service; the hosted installation, plotting, and browser-download steps remain unverified there.

## Browser verification

The real Flask server was opened in a browser. Confirmed initial accepted prediction for 1010, rejected prediction for 1011, empty-string rejection, invalid-input feedback for 10x, state stepping, and automatic playback. Inspected the rendered desktop layout. The local test server is stopped after verification; use the README to start your own instance.

The neural score for the empty string is close to the 0.5 decision boundary. It is correctly classified in this checkpoint, but this is an instructive weak prediction, not robust formal assurance. Exact DFA results are authoritative.

## Reproduce

From the extracted project, install requirements and run:

```bash
python -m unittest discover -s tests -v
```

See README for virtual-environment commands. The ZIP excludes temporary dependencies and Python caches.
