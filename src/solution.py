from dataclasses import dataclass, field
from typing import List, Set

@dataclass
class Solution:
    n: int
    chosen: Set[int] = field(default_factory=set)
    value: float = 0.0

    def clone(self) -> "Solution":
        return Solution(n=self.n, chosen=set(self.chosen), value=self.value)

    def __len__(self):
        return len(self.chosen)

    def as_vector(self) -> List[int]:
        x = [0] * self.n
        for i in self.chosen:
            x[i] = 1
        return x
