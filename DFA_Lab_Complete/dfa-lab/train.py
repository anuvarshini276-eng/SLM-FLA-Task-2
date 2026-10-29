"""Generate unique labeled examples, train with Adam, evaluate, and export."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from dfa import run_dfa
from model import RNN

ROOT = Path(__file__).resolve().parent


def generate(seed=42):
    rng = np.random.default_rng(seed)
    seen = set()
    while len(seen) < 6000:
        length = int(rng.integers(1, 25))
        seen.add(''.join(map(str, rng.integers(0, 2, length))))
    strings = sorted(seen)  # Stable even across Python processes.
    rng.shuffle(strings)
    splits = {'train': [''] + strings[:4200], 'validation': strings[4200:5100],
              'test': strings[5100:]}
    ood = set()
    while len(ood) < 1000:
        ood.add(''.join(map(str, rng.integers(0, 2, int(rng.integers(25, 129))))))
    splits['long_test'] = sorted(ood)
    (ROOT / 'data').mkdir(exist_ok=True)
    for split, values in splits.items():
        with (ROOT / 'data' / f'{split}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['string', 'label', 'length'])
            writer.writerows((s, int(run_dfa(s)['accepted']), len(s)) for s in values)
    return splits


def evaluate(model, strings):
    y = np.array([run_dfa(s)['accepted'] for s in strings], dtype=int)
    p = np.concatenate([model.forward(strings[i:i+128]) for i in range(0, len(strings), 128)])
    pred = (p >= .5).astype(int)
    cm = [[int(((y == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)]
    return {'count': len(strings), 'accuracy': float((pred == y).mean()),
            'confusion_matrix': cm, 'matrix_order': 'rows=true, columns=predicted; reject, accept',
            'mismatches': [{'string': s, 'exact': int(a), 'predicted': int(b)}
                           for s, a, b in zip(strings, y, pred) if a != b][:20]}


def main(epochs=15):
    splits = generate()
    model = RNN()
    rng = np.random.default_rng(43)
    m = {k: np.zeros_like(v) for k, v in model.p.items()}
    v = {k: np.zeros_like(value) for k, value in model.p.items()}
    step, best_loss, history = 0, float('inf'), []
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    for epoch in range(1, epochs+1):
        order = rng.permutation(len(splits['train']))
        losses = []
        for start in range(0, len(order), 128):
            batch = [splits['train'][i] for i in order[start:start+128]]
            labels = np.array([run_dfa(s)['accepted'] for s in batch], dtype=float)
            loss, grad = model.gradients(batch, labels)
            losses.append(loss)
            step += 1
            norm = np.sqrt(sum(np.sum(g*g) for g in grad.values()))
            for k in model.p:
                g = grad[k] * min(1., 1. / (norm + 1e-9))
                m[k] = .9*m[k] + .1*g
                v[k] = .999*v[k] + .001*g*g
                model.p[k] -= .01 * (m[k]/(1-.9**step)) / (np.sqrt(v[k]/(1-.999**step)) + 1e-8)
        val = splits['validation']
        p = model.forward(val)
        y = np.array([run_dfa(s)['accepted'] for s in val])
        val_loss = float(-np.mean(y*np.log(p+1e-9)+(1-y)*np.log(1-p+1e-9)))
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation_loss': val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            model.save(ROOT / 'artifacts' / 'model.npz')
        print(f'Epoch {epoch:02d}: train loss {np.mean(losses):.5f}, validation loss {val_loss:.5f}', flush=True)
    model = RNN.load(ROOT / 'artifacts' / 'model.npz')
    exhaustive = [''.join(s) for length in range(11) for s in itertools.product('01', repeat=length)]
    report = {'seed': 42, 'architecture': 'character RNN, 12 tanh hidden units, sigmoid output',
              'parameters': sum(p.size for p in model.p.values()), 'threshold': .5,
              'training_lengths': [0, 24], 'epochs': epochs, 'history': history,
              'test': evaluate(model, splits['test']), 'long_test': evaluate(model, splits['long_test']),
              'exhaustive_0_to_10': evaluate(model, exhaustive),
              'note': 'Exhaustive set overlaps training. Test and long_test are held out. Accuracy is empirical, not proof.'}
    (ROOT / 'artifacts' / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('test', 'long_test', 'exhaustive_0_to_10')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    main(args.epochs)
