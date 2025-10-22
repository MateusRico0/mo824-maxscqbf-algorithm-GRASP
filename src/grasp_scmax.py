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
    def __init__(self, model: SCMaxQBF, evaluator: Evaluator, cfg: GRASPConfig):
        self.model = model
        self.evaluator = evaluator
        self.cfg = cfg
        if cfg.seed is not None:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)

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

    def _feasible_remove(self, S: Set[int], i: int) -> bool:
        return self.model.feasible_after_removal(S, i)

    def local_search(self, S0: Set[int]) -> Set[int]:
        S = set(S0)
        improved = True

        while improved:
            improved = False

            if self.cfg.ls_mode == "first":
                best_delta = 0.0
                for i in range(self.model.n):
                    if i in S: 
                        continue
                    d = self.evaluator.delta_add(S, i)
                    if d > 1e-12:
                        S.add(i)
                        improved = True
                        break
                if improved:
                    continue

                for i in list(S):
                    if not self._feasible_remove(S, i):
                        continue
                    d = self.evaluator.delta_remove(S, i)
                    if d > 1e-12:
                        S.remove(i)
                        improved = True
                        break

            else:  
                best_i, best_d, best_op = None, 0.0, None

                for i in range(self.model.n):
                    if i in S:
                        continue
                    d = self.evaluator.delta_add(S, i)
                    if d > best_d + 1e-12:
                        best_i, best_d, best_op = i, d, "add"

                for i in S:
                    if not self._feasible_remove(S, i):
                        continue
                    d = self.evaluator.delta_remove(S, i)
                    if d > best_d + 1e-12:
                        best_i, best_d, best_op = i, d, "rem"

                if best_op is not None:
                    if best_op == "add":
                        S.add(best_i)
                    else:
                        S.remove(best_i)
                    improved = True

        return S

    def run(self):
        start = time.time()
        best_S: Set[int] = set()
        best_val = float("-inf")
        ttt = []

        it = 0
        time_best_sol = 0.0
        iter_best_sol = 0
        while True:
            if self.cfg.max_iterations is not None and it >= self.cfg.max_iterations:
                break
            if time.time() - start >= self.cfg.time_limit:
                break

            it += 1
            S0 = self.construct()        
            assert self.model.is_feasible(S0), "Construction failed: not a valid set cover(Line 143 grasp_scmax)" # Optional, maybe remove it
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
