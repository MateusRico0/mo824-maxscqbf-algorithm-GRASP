import sys, os
sys.path.append(os.path.dirname(__file__))

import os, csv, json
from typing import Dict, Tuple, List

from src.qbf import read_sc_max_qbf
from src.sc_model import SCMaxQBF
from src.grasp_scmax import GRASP_SC_MAX_QBF, GRASPConfig

INSTANCES_DIR   = "./instances"          
SEEDS           = list(range(2))        
ALPHA           = 0.3
TIME_LIMIT      = 30.0
MAX_ITER        = 1
LS_MODE         = "best"                 # "first" or "best"
LAMBDA_BALANCE  = 0.4
OUTDIR          = "./results"            
CONFIG_LABEL    = "GRASP_RANDOM_GREEDY"  

TTT_PP_HEADER = [
    "instance","config","seed",
    "total_iterations","total_time",
    "best_cost","time_best_sol","iterations_best_sol"
]
GUROBI_PP_HEADER = ["instance","best_cost","total_time","optimal"]


def run_one(instance_path: str, seed: int) -> Tuple[List, float, float]:
    n, sets, Q = read_sc_max_qbf(instance_path)
    model = SCMaxQBF(Q, sets)
    cfg = GRASPConfig(
        alpha=ALPHA,
        time_limit=TIME_LIMIT,
        ls_mode=LS_MODE,
        seed=seed,
        lambda_balance=LAMBDA_BALANCE,
        max_iterations=MAX_ITER
    )
    grasp = GRASP_SC_MAX_QBF(model, cfg)

    best_S, best_val, _ttt_traj, total_time, total_iterations, time_best_sol, iter_best_sol = grasp.run()

    if time_best_sol is None:
        time_best_sol = total_time
    if iter_best_sol is None:
        iter_best_sol = total_iterations

    row = [
        os.path.basename(instance_path),
        CONFIG_LABEL,
        seed,
        total_iterations,
        f"{float(total_time):.6f}",
        f"{float(best_val):.6f}",
        f"{float(time_best_sol):.6f}",
        int(iter_best_sol),
    ]

    x = [1 if i in best_S else 0 for i in range(n)]
    sol_path = os.path.join(OUTDIR, f"{os.path.basename(instance_path)}.seed{seed}.best_solution.json")
    with open(sol_path, "w") as fsol:
        json.dump({
            "instance": os.path.basename(instance_path),
            "n": n,
            "best_value": float(best_val),
            "seed": seed,
            "chosen_indices": sorted(list(best_S)),
            "x": x
        }, fsol, indent=2)

    return row, float(best_val), float(total_time)


def main():
    if not INSTANCES_DIR or not os.path.isdir(INSTANCES_DIR):
        raise SystemExit(f"INSTANCES_DIR not found: {INSTANCES_DIR}")
    instances = [os.path.join(INSTANCES_DIR, fn)
                 for fn in os.listdir(INSTANCES_DIR)
                 if fn.lower().endswith(".txt")]
    if not instances:
        raise SystemExit(f"No .txt instances found in {INSTANCES_DIR}")
    instances.sort()

    os.makedirs(OUTDIR, exist_ok=True)

    ttt_path  = os.path.join(OUTDIR, "ttt.csv")
    pp_path   = os.path.join(OUTDIR, "performance_profile.csv")
    grb_path  = os.path.join(OUTDIR, "gurobi_performance_profile.csv")

    with open(ttt_path, "w", newline="") as f_ttt:
        csv.writer(f_ttt).writerow(TTT_PP_HEADER)
    with open(pp_path, "w", newline="") as f_pp:
        csv.writer(f_pp).writerow(TTT_PP_HEADER)
    with open(grb_path, "w", newline="") as f_grb:
        csv.writer(f_grb).writerow(GUROBI_PP_HEADER)

    agg_total_iterations = 0
    agg_total_time = 0.0
    agg_best_cost_overall = None
    agg_time_best_overall = None
    agg_iter_best_overall = None

    best_by_instance: Dict[str, Tuple[float, float]] = {} 

    with open(ttt_path, "a", newline="") as f_ttt:
        w_ttt = csv.writer(f_ttt)

        for inst in instances:
            inst_name = os.path.basename(inst)
            best_cost_this_inst = None
            total_time_for_best = None

            for seed in SEEDS:
                row, best_val, total_time = run_one(inst, seed)

                w_ttt.writerow(row)

                agg_total_iterations += int(row[3])
                agg_total_time += float(row[4])

                if (agg_best_cost_overall is None) or (float(row[5]) > agg_best_cost_overall):
                    agg_best_cost_overall = float(row[5])
                    agg_time_best_overall = float(row[6])
                    agg_iter_best_overall = int(row[7])

                if (best_cost_this_inst is None) or (best_val > best_cost_this_inst):
                    best_cost_this_inst = best_val
                    total_time_for_best = total_time

            if best_cost_this_inst is not None:
                best_by_instance[inst_name] = (best_cost_this_inst, total_time_for_best if total_time_for_best is not None else 0.0)

    pp_row = [
        "ALL",
        CONFIG_LABEL,
        "ALL",
        agg_total_iterations,
        f"{agg_total_time:.6f}",
        f"{(agg_best_cost_overall if agg_best_cost_overall is not None else 0.0):.6f}",
        f"{(agg_time_best_overall if agg_time_best_overall is not None else 0.0):.6f}",
        int(agg_iter_best_overall if agg_iter_best_overall is not None else 0),
    ]
    with open(pp_path, "a", newline="") as f_pp:
        csv.writer(f_pp).writerow(pp_row)

    with open(grb_path, "a", newline="") as f_grb:
        w_grb = csv.writer(f_grb)
        for inst_name in sorted(best_by_instance.keys()):
            best_cost, tot_time = best_by_instance[inst_name]
            w_grb.writerow([inst_name, f"{best_cost:.6f}", f"{tot_time:.6f}", ""])

    print("Wrote:")
    print(" -", ttt_path, "(all instances & seeds)")
    print(" -", pp_path,  "(single summary row)")
    print(" -", grb_path, "(one row per instance)")

if __name__ == "__main__":
    main()
