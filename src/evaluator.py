from typing import Iterable, Set
import numpy as np

class Evaluator:
    def __init__(self, Q: np.ndarray):
        self.Q = 0.5 * (Q + Q.T)

    def value(self, chosen: Iterable[int]) -> float:
        idx = np.fromiter(chosen, dtype=int)
        if idx.size == 0:
            return 0.0
        idx.sort()
        v = float(np.sum(self.Q[idx, idx]))
        for a_pos, i in enumerate(idx[:-1]):
            j_block = idx[a_pos + 1:]
            v += float(np.sum(self.Q[i, j_block]))
        return v

    def delta_add(self, chosen_set: Set[int], i: int) -> float:
        if i in chosen_set:
            return 0.0
        s = self.Q[i, i]
        for j in chosen_set:
            if i < j:
                s += self.Q[i, j]
            elif j < i:
                s += self.Q[j, i]
        return float(s)

    def delta_remove(self, chosen_set: Set[int], i: int) -> float:
        if i not in chosen_set:
            return 0.0
        s = -self.Q[i, i]
        for j in chosen_set:
            if j == i:
                continue
            if i < j:
                s -= self.Q[i, j]
            else:
                s -= self.Q[j, i]
        return float(s)
