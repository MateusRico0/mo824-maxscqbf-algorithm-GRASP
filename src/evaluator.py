from typing import Iterable, Set
import numpy as np

class Evaluator:
    def __init__(self, Q: np.ndarray):
        self.Q = 0.5 * (Q + Q.T)

    def value(self, chosen: Iterable[int]) -> float:
        chosen = list(chosen)
        if not chosen:
            return 0.0
        idx = np.array(chosen, dtype=int)
        sub = self.Q[np.ix_(idx, idx)]
        return float(sub.sum())

    def delta_add(self, chosen_set: Set[int], i: int) -> float:
        if i in chosen_set:
            return 0.0
        row = self.Q[i]
        s = sum(row[j] for j in chosen_set)
        return float(self.Q[i, i] + 2.0 * s)

    def delta_remove(self, chosen_set: Set[int], i: int) -> float:
        if i not in chosen_set:
            return 0.0
        row = self.Q[i]
        s = sum(row[j] for j in chosen_set if j != i)
        return float(- (self.Q[i, i] + 2.0 * s))
