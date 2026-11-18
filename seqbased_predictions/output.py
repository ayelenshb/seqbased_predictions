"""Result files (CSV and JSON)."""
import datetime
import json
import platform
import re

import numpy as np
import pandas as pd


def file_name(name):
    """Variant name usable in file names (e.g. 'Literature (alphabet)' -> 'Literature_alphabet')."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def columns_text(columns):
    if columns and isinstance(columns[0], (list, tuple)):
        return ";".join(f"{a}-{b}" for a, b in columns)
    return ";".join(str(c) for c in columns)


def write_positions(path, sel, data):
    weights = np.round(np.asarray(sel.weights, dtype=np.float64), 6)
    if sel.mode == "pairs":
        df = pd.DataFrame({"position_1": [a for a, _ in sel.columns], "position_2": [b for _, b in sel.columns],
                           "weight": weights})
    else:
        df = pd.DataFrame({"position": sel.columns, "weight": weights})
        if sel.individual_rho is not None:
            df["individual_rho"] = [sel.individual_rho[k] for k in sel.idx]
            df["rank"] = [sel.order.index(int(k)) + 1 for k in sel.idx]
    df.sort_values(list(df.columns[:1 if sel.mode == "singles" else 2]), inplace=True)
    df.to_csv(path, index=False)


def write_predictions(path, method_replicates, controls, data):
    rows = []
    def add(group, reps):
        for rep in reps:
            for s, o, p in zip(rep.val, rep.observed, rep.predicted):
                rows.append((group, rep.replicate, data.ids[s], o, p))
    add("method", method_replicates)
    for c in controls:
        add(f"control_{c.control}", c.replicates)
    pd.DataFrame(rows, columns=["set", "replicate", "sequence", "observed", "predicted"]).to_csv(path, index=False)


def write_controls(path, controls):
    pd.DataFrame([{"control": c.control, "positions": columns_text(c.columns), "r2_mean": c.r2_mean,
                   "r2_sd": c.r2_sd, "rmse": c.rmse, "slope": c.slope, "intercept": c.intercept}
                  for c in controls]).to_csv(path, index=False)


def write_individual_rho(path, sel, data):
    """Individual and cumulative ρ of each column (variants sorted by position)."""
    rank = {int(k): r + 1 for r, k in enumerate(sel.order)}
    df = pd.DataFrame({"position": data.col_list, "individual_rho": sel.individual_rho,
                       "rank": [rank[k] for k in range(data.npos)]})
    df["cumulative_rho"] = [sel.cumulative_rho[o - 1] for o in df["rank"]]
    df["selected"] = df["rank"] <= len(sel.idx)
    df.to_csv(path, index=False)


def versions():
    import Bio, matplotlib, numba, scipy
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "numba": numba.__version__, "biopython": Bio.__version__, "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__, "system": platform.platform()}


def write_config(path, case, extra):
    data = {"case": case.as_dict(), "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "versions": versions(), **extra}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
