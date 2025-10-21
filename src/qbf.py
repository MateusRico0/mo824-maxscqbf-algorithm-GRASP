from typing import Tuple, List, Set
import numpy as np

def read_sc_max_qbf(path: str) -> Tuple[int, List[Set[int]], np.ndarray]:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise ValueError(f"Empty file: {path}")

    first_toks = lines[0].split()
    n = int(float(first_toks[0]))

    if len(lines) < 1 + n:
        raise ValueError(f"Expected at least {1+n} lines, found {len(lines)} in {path}")

    cov_lines = lines[1:1+n]
    sets: List[Set[int]] = []
    for ln in cov_lines:
        toks = ln.split()
        elems = []
        for t in toks:
            try:
                v = int(float(t))
                elems.append(v - 1)  
            except:
                pass
        sets.append(set(elems))

    tail = " ".join(lines[1+n:])
    if not tail.strip():
        raise ValueError(f"No Q data found after coverage lines in {path}")

    nums: List[float] = []
    for t in tail.split():
        try:
            nums.append(float(t))
        except:
            pass

    expected = n*(n+1)//2
    if len(nums) < expected:
        raise ValueError(
            f"Not enough Q values in {path}: got {len(nums)}, expected {expected}"
        )
    nums = nums[:expected]

    Q = np.zeros((n, n), dtype=float)
    idx = 0
    for i in range(n):
        row_len = n - i
        for k in range(row_len):
            j = i + k
            val = nums[idx]
            idx += 1
            Q[i, j] = val
            Q[j, i] = val

    return n, sets, Q