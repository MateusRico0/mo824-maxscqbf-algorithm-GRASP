from dataclasses import dataclass
from typing import Set, Optional
import random, time
import numpy as np

from src.evaluator import Evaluator
from src.sc_model import SCMaxQBF

@dataclass
class GRASPConfig:
    alpha: float = 0.3
    time_limit: float = 30.0
    ls_mode: str = "best"        
    seed: Optional[int] = None
    lambda_balance: float = 0.5
    max_iterations: Optional[int] = None

class GRASP_SC_MAX_QBF:
    def __init__(self, model: SCMaxQBF, cfg: GRASPConfig):
        self.model = model
        self.cfg = cfg
        if cfg.seed is not None:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
        self.evaluator = Evaluator(model.Q)

    def construct(self) -> Set[int]:
        n = self.model.n
        chosen: Set[int] = set()
        covered: Set[int] = set()

        while covered != self.model.U:
            gains = []
            for i in range(n):
                if i in chosen:
                    continue
                cov_gain = len(self.model.sets[i] - covered)
                dv = self.evaluator.delta_add(chosen, i)
                score = (1.0 - self.cfg.lambda_balance) * cov_gain + self.cfg.lambda_balance * max(0.0, dv)
                gains.append((i, cov_gain, dv, score))
            if not gains:
                break
            if any(g[1] > 0 for g in gains):
                gains = [g for g in gains if g[1] > 0]
            scores = [g[3] for g in gains]
            smax, smin = max(scores), min(scores)
            thr = smax - self.cfg.alpha * (smax - smin)
            rcl = [g for g in gains if g[3] >= thr]
            i, _, _, _ = random.choice(rcl)
            chosen.add(i)
            covered |= self.model.sets[i]

        while True:
            best_i, best_dv = None, 0.0
            for i in range(n):
                if i in chosen:
                    continue
                dv = self.evaluator.delta_add(chosen, i)
                if dv > best_dv:
                    best_dv, best_i = dv, i
            if best_i is None or best_dv <= 1e-9:
                break
            chosen.add(best_i)

        return chosen

    def local_search(self, chosen: Set[int]) -> Set[int]:
        def coverage_ok(S: Set[int]) -> bool:
            cov = set()
            for i in S:
                cov |= self.model.sets[i]
            return cov == self.model.U

        current = set(chosen)
        best_val = self.evaluator.value(current)

        while True:
            neighborhood = []

            for i in range(self.model.n):
                if i in current: 
                    continue
                dv = self.evaluator.delta_add(current, i)
                if dv > 1e-9:
                    neighborhood.append(("add", i, dv))

            for j in list(current):
                if not coverage_ok(current - {j}):
                    continue
                dv = self.evaluator.delta_remove(current, j)
                if dv > 1e-9:  
                    neighborhood.append(("rem", j, -dv))

            for i in range(self.model.n):
                if i in current:
                    continue
                for j in list(current):
                    cand = (current - {j}) | {i}
                    if not coverage_ok(cand):
                        continue
                    dv = self.evaluator.delta_add(current - {j}, i) + self.evaluator.delta_remove(current, j)
                    if dv > 1e-9:
                        neighborhood.append(("swap", (i, j), dv))

            if not neighborhood:
                break

            if self.cfg.ls_mode == "first":
                random.shuffle(neighborhood)
                mv = neighborhood[0]
            else:
                mv = max(neighborhood, key=lambda t: t[2])

            kind, idx, dv = mv
            if dv <= 1e-9:
                break

            if kind == "add":
                current.add(idx)
            elif kind == "rem":
                current.remove(idx)
            else:
                i, j = idx
                current.remove(j)
                current.add(i)

            best_val += dv

        return current

    def run(self):
        start = time.time()
        best_S: Set[int] = set()
        best_val = float("-inf")
        ttt = []
        it = 0
        time_best_sol = None
        iter_best_sol = None

        while time.time() - start < self.cfg.time_limit:
            if self.cfg.max_iterations and it >= self.cfg.max_iterations:
                break
            it += 1
            S0 = self.construct()
            S1 = self.local_search(S0)
            val = self.evaluator.value(S1)
            if val > best_val:
                best_val = val
                best_S = set(S1)
                time_best_sol = time.time() - start
                iter_best_sol = it
            ttt.append((time.time() - start, best_val, it))

        total_time = time.time() - start
        total_iterations = it
        return best_S, best_val, ttt, total_time, total_iterations, time_best_sol, iter_best_sol
