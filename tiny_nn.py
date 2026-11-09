"""
tiny_nn.py
A minimal, dependency-free (numpy only) 3-layer MLP with manually
derived forward/backward passes and an Adam optimizer, used as the
Q-network for train_navigation.py.

Deliberately NOT built on PyTorch/TensorFlow: those are multi-gigabyte
dependencies that are often unavailable (or undesirable) on the kind
of embedded boards used for real robot hardware (e.g. a Raspberry Pi
or a microcontroller-adjacent SBC). A ~15-input, 8-output, 2-hidden-
layer network like this one needs only basic matrix multiplies, so a
handful of numpy arrays is all that's required both to train it here
and to run it later on constrained hardware.
"""

import numpy as np


class QNetwork:
    def __init__(self, in_dim, hidden_dim, out_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), size=(in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.W3 = rng.normal(0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, out_dim))
        self.b3 = np.zeros(out_dim)
        self._param_names = ["W1", "b1", "W2", "b2", "W3", "b3"]
        self._init_adam()

    def forward(self, x):
        """x: (batch, in_dim). Returns (q_values, cache_for_backward)."""
        z1 = x @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.tanh(z2)
        z3 = a2 @ self.W3 + self.b3
        return z3, (x, a1, a2)

    def backward(self, cache, dQ):
        """dQ: (batch, out_dim) — gradient of the loss w.r.t. the output."""
        x, a1, a2 = cache
        dW3 = a2.T @ dQ
        db3 = dQ.sum(axis=0)
        da2 = dQ @ self.W3.T
        dz2 = da2 * (1 - a2 ** 2)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - a1 ** 2)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def _init_adam(self):
        self.m = {n: np.zeros_like(getattr(self, n)) for n in self._param_names}
        self.v = {n: np.zeros_like(getattr(self, n)) for n in self._param_names}
        self.t = 0

    def adam_step(self, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, clip=5.0):
        self.t += 1
        for name, g in grads.items():
            g = np.clip(g, -clip, clip)
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * g
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * (g ** 2)
            m_hat = self.m[name] / (1 - beta1 ** self.t)
            v_hat = self.v[name] / (1 - beta2 ** self.t)
            param = getattr(self, name)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, name, param)

    def copy_from(self, other):
        for name in self._param_names:
            setattr(self, name, getattr(other, name).copy())

    def q_values(self, obs):
        """obs: 1D array. Returns 1D array of Q-values, no grad tracking."""
        q, _ = self.forward(obs.reshape(1, -1))
        return q[0]

    def save(self, path):
        np.savez(path, **{n: getattr(self, n) for n in self._param_names})

    def load(self, path):
        data = np.load(path)
        for name in self._param_names:
            setattr(self, name, data[name])
