from typing import List, Set
import numpy as np

class SCMaxQBF:
    def __init__(self, Q: np.ndarray, sets: List[Set[int]]):
        assert Q.shape[0] == Q.shape[1], "Q must be square"
        n = Q.shape[0]
        assert len(sets) == n, "sets must have length n"
        self.Q = Q
        self.sets = sets
        self.n = n
        self.U = set()
        for s in sets:
            self.U |= s

    def is_feasible(self, chosen: Set[int]) -> bool:
        if not chosen:
            return False
        covered = set()
        for j in chosen:
            if 0 <= j < self.n:
                covered |= self.sets[j]
        return covered == self.U

    def feasible_after_removal(self, chosen: Set[int], i: int) -> bool:
        if i not in chosen:
            return True
        covered = set()
        for j in chosen:
            if j == i:
                continue
            covered |= self.sets[j]
        return self.U.issubset(covered)

        
