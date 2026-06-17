import numpy as np
from collections import defaultdict


class Logger:
    '''Lightweight time-series logger. Call .log(t, **kwargs) each step;
    .save() / .arrays() to retrieve. Keys are dynamic.'''
    def __init__(self):
        self._data = defaultdict(list)

    def log(self, t, **kwargs):
        self._data['t'].append(t)
        for k, v in kwargs.items():
            self._data[k].append(np.asarray(v).copy())

    def arrays(self):
        return {k: np.array(v) for k, v in self._data.items()}

    def save(self, path):
        np.savez(path, **self.arrays())