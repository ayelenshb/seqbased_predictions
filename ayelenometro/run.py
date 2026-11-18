"""Full run of a case."""
import os
import time

import numpy as np
import pandas as pd

from . import output, plots
from .alphabets import create_alphabet
from .config import VARIANTS, Case
from .controls import empirical_p, permutation_controls, position_controls
from .data import alphabet_data, identity_data, universe_columns
from .labels import check_against_fasta, load_labels, save_labels
from .prediction import make_partitions, metrics, predict
from .variants import fixed_selection, select

# The seed of each stage is derived from a text key (see Case.stage_seed). Those keys keep the
# text of the original (Spanish) version of the code, so that the results do not change:
#   "alfabeto" (alphabet), "particiones" (partitions), "mcmc:<variant>",
#   "controles:<variant>" (controls) and "permutacion:<variant>:<k>" (permutations),
# where the literature selections use their original names.
_SEED_NAMES = {"Literature (identity)": "Literatura (identidad)",
               "Literature (alphabet)": "Literatura (alfabeto)"}


def _seed_name(name):
    return _SEED_NAMES.get(name, name)


def _alphabet(case, ids, sequences, phen, verbose):
    path = case.alphabet_path
    if os.path.exists(path):
        lab = load_labels(path)
        check_against_fasta(lab, ids, sequences, phen, path)
        if verbose:
            print(f"  alphabet: using the existing one ({path})")
        return lab, False
    if verbose:
        print(f"  alphabet: {path} does not exist; creating it ({sequences.shape[1]} columns)")
    lab = create_alphabet(ids, sequences, phen, case.integer_seed("alfabeto"),
                          case.alphabet_iterations, case.alphabet_alpha, case.name, verbose)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_labels(path, lab)
    return lab, True


def _evaluate_selection(sel, data, case, partitions, folder, verbose):
    """Prediction, controls, files and plots of a selection. Returns the summary row."""
    t0 = time.time()
    fname = output.file_name(sel.name)
    reps = predict(sel.vector, data, partitions)
    r2m, r2s, rmse = metrics(reps)

    ctrl = position_controls(sel, data, partitions, case.n_controls, case.rng(f"controles:{_seed_name(sel.name)}"))
    perm = []
    if case.n_permutations and sel.method != "fixed":
        perm = permutation_controls(sel.name, _seed_name(sel.name), sel.encoding, sel.mode, sel.method, data, case,
                                    partitions, case.n_permutations, case.stage_seed)

    output.write_positions(os.path.join(folder, f"positions_{fname}.csv"), sel, data)
    output.write_predictions(os.path.join(folder, f"predictions_{fname}.csv"), reps, ctrl, data)
    if ctrl:
        output.write_controls(os.path.join(folder, f"controls_{fname}.csv"), ctrl)
    if perm:
        output.write_controls(os.path.join(folder, f"permutations_{fname}.csv"), perm)
    if sel.individual_rho is not None and sel.mode == "singles":
        output.write_individual_rho(os.path.join(folder, f"individual_rho_{fname}.csv"), sel, data)

    p_ctrl = empirical_p(r2m, ctrl)
    title = f"{case.name} · {sel.name}"
    plots.observed_vs_predicted(reps, ctrl, case.phenotype, title, os.path.join(folder, f"obs_vs_pred_{fname}.png"))
    if sel.cumulative_rho is not None:
        plots.cumulative_rho(sel, title, os.path.join(folder, f"cumulative_rho_{fname}.png"))
    if sel.individual_rho is not None and sel.mode == "singles":
        plots.individual_rho(sel, data, title, os.path.join(folder, f"individual_rho_{fname}.png"))

    row = {
        "variant": sel.name, "encoding": sel.encoding, "mode": sel.mode, "method": sel.method,
        "n_positions": len(sel.distinct_positions),
        "n_pairs": len(sel.columns) if sel.mode == "pairs" else np.nan,
        "pct_positions": 100 * len(sel.distinct_positions) / data.L,
        "rho": sel.rho, "r2_mean": r2m, "r2_sd": r2s, "rmse": rmse,
        "r2_controls_mean": np.mean([c.r2_mean for c in ctrl]) if ctrl else np.nan,
        "r2_controls_sd": np.std([c.r2_mean for c in ctrl]) if ctrl else np.nan,
        "p_controls": p_ctrl, "n_controls": len(ctrl),
    }
    if case.n_permutations:
        row.update({"r2_permutations_mean": np.mean([c.r2_mean for c in perm]) if perm else np.nan,
                    "r2_permutations_sd": np.std([c.r2_mean for c in perm]) if perm else np.nan,
                    "p_permutations": empirical_p(r2m, perm), "n_permutations": len(perm)})
    row["time_s"] = round(time.time() - t0, 1)
    row["positions"] = output.columns_text(sorted(sel.columns))
    row["distinct_positions"] = (";".join(str(p) for p in sel.distinct_positions)
                                 if sel.mode == "pairs" else np.nan)
    if verbose:
        print(f"    r² = {r2m:.3f} ± {r2s:.3f} | RMSE = {rmse:.3f} | positions = {row['n_positions']} "
              f"({row['pct_positions']:.1f} %) | controls: r² = {row['r2_controls_mean']:.3f}, p = {p_ctrl:.3f}")
    return row


def run_case(case: Case, verbose=True):
    """Runs all the variants of a case and writes the results in <results_dir>/<name>/."""
    t0 = time.time()
    if verbose:
        print(f"\n==== {case.name} ====")
    ids, sequences, phen = case.validate()
    os.makedirs(case.folder, exist_ok=True)

    universe = universe_columns(sequences, case.max_gaps)
    data = {"identity": identity_data(ids, sequences, phen, universe)}
    alphabet_created = None
    if case.uses_alphabet():
        lab, alphabet_created = _alphabet(case, ids, sequences, phen, verbose)
        data["alphabet"] = alphabet_data(lab, universe, ids)
    if verbose:
        print(f"  {len(ids)} sequences, {sequences.shape[1]} columns, {len(universe)} with <= {case.max_gaps:.0%} gaps "
              f"(varying: identity {data['identity'].npos}"
              + (f", alphabet {data['alphabet'].npos}" if "alphabet" in data else "") + ")")

    partitions = make_partitions(len(ids), case.replicates, case.train_fraction, case.rng("particiones"))

    rows = []
    for name in case.variants:
        encoding, mode, method = VARIANTS[name]
        if verbose:
            print(f"  -- {name} ({encoding}, {mode}, {method})")
        sel = select(name, encoding, mode, method, data[encoding], case,
                     rng_mcmc=case.rng(f"mcmc:{name}"), verbose=verbose)
        rows.append(_evaluate_selection(sel, data[encoding], case, partitions, case.folder, verbose))

    if case.literature:
        for encoding, d in data.items():
            name = f"Literature ({encoding})"
            sel = fixed_selection(name, encoding, case.literature, d)
            skipped = sorted(set(case.literature) - set(sel.columns))
            if verbose:
                print(f"  -- {name}: {len(sel.columns)} positions"
                      + (f" (skipped because of gaps or no variation: {skipped})" if skipped else ""))
            if len(sel.idx):
                rows.append(_evaluate_selection(sel, d, case, partitions, case.folder, verbose))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(case.folder, "summary.csv"), index=False)
    output.write_config(os.path.join(case.folder, "config_used.json"), case, {
        "columns_used": [int(c) + 1 for c in universe],
        "alphabet": {"path": case.alphabet_path, "created_in_this_run": alphabet_created} if case.uses_alphabet() else None,
        "duration_s": round(time.time() - t0, 1),
    })
    if verbose:
        print(f"  done in {time.time() - t0:.0f} s -> {case.folder}")
    return df


def run_cases(cases, verbose=True):
    """Runs several cases; each one is independent (its own seed)."""
    return {case.name: run_case(case, verbose) for case in cases}
