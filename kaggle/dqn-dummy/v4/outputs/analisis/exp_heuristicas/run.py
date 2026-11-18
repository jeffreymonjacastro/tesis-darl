"""Experimento sin entrenamiento: ¿actuar antes con A3 mejora el retorno del episodio?

Reutiliza las celdas de definiciones del notebook v4 (datos, pipeline, drift,
monitor y entorno) y compara políticas fijas en 10 replays de B-test.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
WORK = ROOT / ".tmp/exp_heuristicas/work"
WORK.mkdir(parents=True, exist_ok=True)
WORK.mkdir(exist_ok=True)
os.chdir(ROOT)
os.environ["DARL_DUMMY_OUTPUT"] = str(WORK)  # evita sobrescribir kaggle/dqn-dummy/v4/outputs

nb = json.loads((ROOT / "kaggle/dqn-dummy/v4/input/main.ipynb").read_text(encoding="utf-8"))
code = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
# Celdas de código 0..12 = celdas 1..25 del notebook: definiciones, sin entrenar ni ejecutar.
g = {"__name__": "v4"}
t0 = time.time()
for src in code[:13]:
    exec(compile(src, "v4_cell", "exec"), g)
print(f"Definiciones v4 cargadas en {time.time()-t0:.0f}s", flush=True)

BedEnv, make_replay, inject, auc = g["BedEnv"], g["make_replay"], g["inject"], g["auc"]
B_PARTS, SEED, ACTION_SECONDS = g["B_PARTS"], g["SEED"], g["ACTION_SECONDS"]
COST_LAMBDA, COST_UNIT = g["COST_LAMBDA"], g["COST_UNIT"]
COV_RATE, CONCEPT_RATE = 0.25, 0.15  # velocidades calibradas en la corrida v4
# Costos de la corrida v4 en Kaggle (run_summary.json), no del hardware local.
v4_summary = json.loads((ROOT / "kaggle/dqn-dummy/v4/outputs/run_summary.json").read_text(encoding="utf-8"))
FIXED_COST = np.array([v4_summary["action_cost_units"][a] for a in ("A1", "A2", "A3", "A4")])
N_REPLAYS = int(os.environ.get("N_REPLAYS", 20))
print("Costo fijo por acción:", FIXED_COST.round(4), flush=True)


class Policy:
    """Stateful daily policy; sees only the monitor metrics of the current day."""

    def __init__(self, name):
        self.name = name
        self.ref_auc = None   # AUC observada tras la última actualización del modelo
        self.last = 0
        self.day = 0

    def choose(self, m, env):
        self.day += 1
        a = np.nan_to_num(m["auc"], nan=1.0)
        if self.ref_auc is None or self.last in (2, 3):
            self.ref_auc = a
        cov = m["ks"] >= .30 or m["psi"] >= .25
        if self.name == "A1":
            act = 0
        elif self.name == "heuristica_085":
            act = 3 if cov and a < .85 else 1 if cov else 2 if a < .85 else 0
        elif self.name == "heuristica_090":
            act = 3 if cov and a < .90 else 1 if cov else 2 if a < .90 else 0
        elif self.name == "heuristica_caida":
            low = a < self.ref_auc - .03 or a < .85
            act = 3 if cov and low else 1 if cov else 2 if low else 0
        elif self.name == "periodica_A3_3d":
            act = 1 if cov else 2 if self.day % 3 == 0 else 0
        elif self.name == "oracle_miope":
            act = oracle_action(env)
        else:
            raise ValueError(self.name)
        self.last = act
        return act


def oracle_action(env):
    """Best next-day utility with fixed costs; uses future labels (reference only)."""
    day = env.day
    update = env.parts[day].loc[env.parts[day].role.eq("update")]
    nxt = env.parts[day + 1].loc[env.parts[day + 1].role.eq("eval")]
    utils = []
    for a in range(4):
        new, _ = env.pipe.act(a, update)
        utils.append(auc(nxt, new) - FIXED_COST[a])
    return int(np.nanargmax(utils))


POLICIES = ["A1", "heuristica_085", "heuristica_090", "heuristica_caida", "periodica_A3_3d", "oracle_miope"]
SCENARIOS = ["stable", "covariate", "concept", "combined"]
rows = []
for k in range(N_REPLAYS):
    base = make_replay(B_PARTS["test"], days=16, seed=SEED + k)
    for scenario in SCENARIOS:
        schedule = inject(base, scenario, COV_RATE, CONCEPT_RATE)
        for name in POLICIES:
            env, pol = BedEnv(schedule), Policy(name)
            env.reset()
            while env.day < env.days:
                _, _, done, _, _ = env.step(pol.choose(env.current[1], env))
                if done:
                    break
            acts = pd.DataFrame(env.actions)
            ex = acts.executed.to_numpy()
            rows.append({"replay": k, "scenario": scenario, "policy": name,
                         "return": float(acts.auc_next.sum() - FIXED_COST[ex].sum()),
                         "mean_auc_next": float(acts.auc_next.mean()), "cost": float(FIXED_COST[ex].sum()),
                         **{f"n_A{i+1}": int((ex == i).sum()) for i in range(4)},
                         "first_A3_or_A4": int(acts.day[np.isin(ex, (2, 3))].min()) if np.isin(ex, (2, 3)).any() else -1})
    pd.DataFrame(rows).to_csv(WORK / "resultados_replays.csv", index=False)
    print(f"replay {k+1}/{N_REPLAYS} listo ({time.time()-t0:.0f}s)", flush=True)
print("FIN", flush=True)
