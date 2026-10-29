"""Tiny character RNN. All parameters are learned, including recurrent weights.

h_t = tanh(x_t W + h_(t-1) U + b); P(accept) = sigmoid(h_final v + c).
Padding freezes hidden states, so it never becomes a character.
"""
import numpy as np


class RNN:
    def __init__(self, hidden=12, seed=42):
        rng = np.random.default_rng(seed)
        self.p = dict(W=rng.normal(0, .25, (2, hidden)),
                      U=rng.normal(0, .15, (hidden, hidden)),
                      b=np.zeros(hidden), v=rng.normal(0, .25, hidden), c=np.zeros(1))

    def forward(self, strings, cache=False):
        n = len(strings)
        h = np.zeros((n, len(self.p['b'])))
        history = []
        for t in range(max(map(len, strings), default=0)):
            active = np.array([t < len(s) for s in strings])[:, None]
            x = np.zeros((n, 2))
            for i, s in enumerate(strings):
                if t < len(s):
                    x[i, int(s[t])] = 1
            candidate = np.tanh(x @ self.p['W'] + h @ self.p['U'] + self.p['b'])
            if cache:
                history.append((x, h, candidate, active))
            h = np.where(active, candidate, h)
        z = h @ self.p['v'] + self.p['c'][0]
        prob = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
        return (prob, h, history) if cache else prob

    def gradients(self, strings, labels):
        prob, h, history = self.forward(strings, cache=True)
        dz = (prob - labels) / len(labels)
        grad = {k: np.zeros_like(v) for k, v in self.p.items()}
        grad['v'] = h.T @ dz
        grad['c'][0] = dz.sum()
        dh = dz[:, None] * self.p['v']
        for x, previous, candidate, active in reversed(history):
            da = dh * active * (1 - candidate ** 2)
            grad['W'] += x.T @ da
            grad['U'] += previous.T @ da
            grad['b'] += da.sum(axis=0)
            dh = da @ self.p['U'].T + dh * (~active)
        loss = -np.mean(labels * np.log(prob + 1e-9) + (1-labels) * np.log(1-prob + 1e-9))
        return float(loss), grad

    def save(self, path):
        np.savez_compressed(path, **self.p)

    @classmethod
    def load(cls, path):
        model = cls()
        with np.load(path, allow_pickle=False) as data:
            model.p = {k: data[k].copy() for k in model.p}
        return model
