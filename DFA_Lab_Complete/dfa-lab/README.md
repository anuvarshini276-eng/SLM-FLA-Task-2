# DFA Lab — learn the pattern, verify every step

A complete beginner FLA project: synthetic labeled data → a small character-level recurrent sequence classifier → evaluation against an exact DFA → exported weights → a Flask API and interactive website.

## Run the included trained model

Install Python 3.10–3.13. Extract the ZIP and open a terminal **inside `dfa-lab`** (the folder containing `app.py`). No GPU, account, API key, or model download is needed. Installing dependencies needs internet once.

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

macOS / Linux:
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Open **http://127.0.0.1:5000**. Stop with Ctrl+C. The app binds to your computer only; Flask's development server is for this local demo, not public production hosting. No build step is needed. Keep `templates/`, `static/`, and `artifacts/` with `app.py`.

## What language does it recognize?

`L = { w ∈ {0,1}* | w ends in 0 }`, equivalent to the regular expression `(0|1)*0` with full-string matching.

The DFA is `M = (Q, Σ, δ, q0, F)`:

- `Q = {q0, q1}`; `Σ = {0,1}`; start state `q0`; accepting set `F = {q1}`.
- `q0` means no symbol has been read, or the last symbol was `1`.
- `q1` means the last symbol was `0`.

| Current state | Read 0 | Read 1 |
|---|---|---|
| q0 | q1 | q0 |
| q1 | q1 | q0 |

For `1010`: `q0 → q0 → q1 → q0 → q1`, so accept. `1011` is rejected. The empty string `ε` leaves the state at `q0`, so reject. On the website leave the field empty to represent ε; do not type the Greek character. Invalid characters are input errors, not rejected strings in the binary language. Spaces are intentionally not trimmed.

Why is the DFA exact? Initially q0 correctly represents an empty prefix. After each symbol, the transition sets q1 exactly when that symbol is 0. By induction the final state is q1 exactly when the input ends in 0. This argument covers every finite binary string; the website imposes a 256-character limit for a readable demo.

## Neural prediction versus exact validation

The neural component is a **character-level sequence classifier**, not a generative language model or pretrained LLM. It reads the full sequence and learns to estimate acceptance. Calling it an SLM is loose terminology; “tiny recurrent sequence predictor” is more accurate.

Each bit becomes a two-element one-hot vector. A 12-unit tanh RNN updates `h_t = tanh(x_t W + h_(t-1) U + b)` and outputs `sigmoid(h_final v + c)`. All **193 parameters** are learned using binary cross-entropy, backpropagation through time, gradient clipping, and Adam. Padding freezes hidden states; empty strings use the zero initial state. The threshold is 0.5. There is no hard-coded suffix check inside neural inference.

The DFA is used to generate labels and evaluate predictions, never to replace a model prediction. Both paths are returned separately. The website labels the neural output as `P(accept)`; this score is not calibrated confidence or proof. Even 100% accuracy on sampled tests is not a guarantee on all possible strings. If the two answers disagree, the DFA is authoritative. This language is intentionally easy: its final character is sufficient, and a neural model is unnecessary for practical validation. The purpose is to compare learning with formal computation.

## Dataset and evaluation

CSV columns: `string,label,length`; label 1 = accept, 0 = reject. The empty CSV string is valid data, not a missing record (with pandas use `keep_default_na=False`).

| Split | Rows | Lengths | Purpose |
|---|---:|---|---|
| train | 4,201 | 0–24 | Fit model; includes ε once |
| validation | 900 | 1–24 | Choose lowest validation-loss checkpoint |
| test | 900 | 1–24 | Held-out in-range evaluation |
| long_test | 1,000 | 25–128 | Held-out length generalization |

Seeds and sorted deduplication make generation repeatable. Every string is unique across all four splits. Lengths are initially sampled uniformly, but deduplication changes the final distribution; this is not a uniform sample over all binary strings. The splits are random, not exactly label-balanced. Validation and test sets do not include ε, so empty-input behavior is checked separately. An additional exhaustive test of all 2,047 strings of lengths 0–10 is reported; it overlaps training and must not be advertised as held out.

`artifacts/metrics.json` contains accuracy, confusion matrices, up to 20 mismatches per evaluation, and training history. Rows of each matrix are true labels; columns are predictions; order is reject, accept. Metrics are generated from the exported checkpoint.

## Retrain locally

With your virtual environment's Python (replace `python` below with `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux):

```bash
python train.py --epochs 15
python -m unittest discover -s tests -v
python app.py
```

Training regenerates the CSVs and overwrites `artifacts/model.npz` and `artifacts/metrics.json`. Stop and restart Flask after replacing weights. Training is CPU-friendly and should finish in a few minutes; timing depends on your machine. Seeds are fixed, but numerical results may vary slightly across NumPy/platform versions.

## Train in Google Colab

1. Open https://colab.research.google.com/ and choose **File → Upload notebook**.
2. Upload `DFA_Training_Colab.ipynb` from this project.
3. Use a CPU runtime; select **Runtime → Run all**. The notebook is self-contained: it writes the same training files, generates data, trains, evaluates, plots loss, and tests export/reload equivalence.
4. The final cell downloads `dfa-trained-artifacts.zip`. Extract its `artifacts` folder into your local `dfa-lab` folder, replacing the existing folder.
5. Restart `app.py`. Keep model and metrics from the same run together.

Colab is for training; run the Flask website locally. The notebook's download cell is specific to Colab; outside Colab the generated ZIP is simply retained on disk.

## Files

```text
app.py                     Flask website + API
dfa.py                     Input validation and exact transitions
model.py                   NumPy RNN, gradients, export/import
train.py                   Dataset generation, training, evaluation
DFA_Training_Colab.ipynb    Self-contained training notebook
data/*.csv                 Generated, labeled split datasets
artifacts/model.npz         Actual trained numeric weights
artifacts/metrics.json      Actual exported-model evaluation
templates/index.html       Website markup and DFA diagram
static/style.css            Responsive styling, no remote assets
static/app.js               API integration and trace playback
tests/test_project.py      DFA/API/data/gradient/export tests
TEST_REPORT.md             Local verification results
requirements.txt           Pinned direct dependencies
```

## API

`POST /api/predict` with `Content-Type: application/json`:

```json
{"string":"1010"}
```

Response includes `exact.accepted`, `exact.final_state`, full `exact.trace`, `neural.accepted`, `neural.accept_probability`, `neural.agrees`, and `outside_training_lengths`. Missing weights produce `neural: null` plus a clear message; the exact validator still works. Invalid JSON, missing fields, non-string values, nonbinary characters, and strings over 256 characters return HTTP 400. Bodies over 4 KB return 413. `GET /api/metrics` returns saved evaluation statistics.

## Demo / viva walkthrough

1. State the language and explain the two-state transition table.
2. Show where the synthetic labels come from and why split overlap is prevented.
3. Explain one-hot encoding, hidden state, sigmoid, and held-out evaluation.
4. Try `1010`, `1011`, and ε. Use Step/Play and inspect the full trace.
5. Try a 25+ character string to demonstrate the training-length notice.
6. Explain why neural agreement is evidence, whereas the DFA's transition invariant is a proof.

## Troubleshooting

- `No module named flask/numpy`: install requirements using the same Python executable used to launch the app.
- Port 5000 busy: stop the other server, or run `python -m flask --app app:create_app run --port 5001` and visit that port.
- Model unavailable: run training, ensure the weights are in `artifacts/model.npz`, and restart Flask.
- Do not open `index.html` directly; the page requires the Flask API.
- The visual trace begins at step 0 even though result cards already show the final answer.

## Reference documentation

- Flask installation: https://flask.palletsprojects.com/en/stable/installation/
- NumPy array loading: https://numpy.org/doc/stable/reference/generated/numpy.load.html (weights are numeric arrays loaded with `allow_pickle=False`).

This project is provided for educational use; see `LICENSE`.
