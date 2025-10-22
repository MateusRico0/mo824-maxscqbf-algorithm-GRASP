from typing import Tuple, List, Set
import numpy as np

def read_sc_max_qbf(path: str) -> Tuple[int, List[Set[int]], np.ndarray]:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise ValueError(f"Empty file: {path}")

    n = int(float(lines[0].split()[0]))

    sizes = list(map(lambda x: int(float(x)), lines[1].split()))
    if len(sizes) != n:
        raise ValueError(f"Expected {n} set sizes on line 2, got {len(sizes)} in {path}")

    sets: List[Set[int]] = []
    idx = 2
    for i in range(n):
        toks = lines[idx].split()
        idx += 1
        if len(toks) != sizes[i]:
            raise ValueError(f"S{i+1} expects {sizes[i]} elements, found {len(toks)} in {path}")
        elems_1based = [int(float(t)) for t in toks]
        sets.append(set(e - 1 for e in elems_1based))

    tail_nums: List[float] = []
    for ln in lines[idx:]:
        tail_nums.extend(float(t) for t in ln.split())
    expected = n * (n + 1) // 2
    if len(tail_nums) < expected:
        raise ValueError(f"Not enough Q values: got {len(tail_nums)}, expected {expected}")

    Q = np.zeros((n, n), dtype=float)
    k = 0
    for i in range(n):
        for j in range(i, n):
            v = tail_nums[k]; k += 1
            Q[i, j] = v
            Q[j, i] = v  

    return n, sets, Q