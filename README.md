## Installation

```
pip install -r requirements.txt
```

## Usage

Open `ayelenometro.ipynb`, fill in the configuration cell and run the run cell.
The included example (`HIV_FPV.fasta`: HIV-1 protease variants with their percent inhibition by
fosamprenavir, FPV) runs from that notebook.

It can also be used from Python:

```python
from ayelenometro import Case, run_case
case = Case(fasta="HIV_FPV.fasta", name="HIV_FPV", phenotype="FPV-IP (%)",
            phenotype_index=-1, create_alphabet=True)
summary = run_case(case)
```

## Inputs

- **Alignment** in FASTA format. The phenotype goes in the ID of each sequence, separated by `_`
  (for example `>Human_0.8095`); `phenotype_index` indicates which field it is (`-1` = the last one).
- **Literature** (optional): list of positions in alignment numbering (1-based).

## `Case` options

| Option | Default | Description |
|---|---|---|
| `fasta`, `name`, `phenotype`, `phenotype_index` | (required) | alignment, case name, phenotype name and phenotype field (see above) |
| `create_alphabet` | `False` | if the case alphabet does not exist: `True` creates it, `False` stops the run. If it exists, it is used. |
| `alphabet` | `<results_dir>/<name>/alphabet_<name>.npz` | path to the alphabet |
| `literature` | `None` | literature positions |
| `max_gaps` | `0.5` | maximum fraction of gaps for a column to be used (all variants) |
| `seed` | `42` | case seed; all stages derive from it |
| `lambda_reg` | `1.0` | λ of the Monte Carlo regularization |
| `variants` | `SS, MC-SS, MC-E, AC-MC-SS, AC-MC-E` | also available: `E`, `AC-SS`, `AC-E` |
| `replicates`, `train_fraction` | `10`, `0.9` | training/validation replicates |
| `n_controls` | `100` | random position sets per variant |
| `n_permutations` | `0` | controls with permuted phenotypes (they repeat the selection; costly) |
| `alphabet_iterations`, `alphabet_alpha` | `1000`, `0.1` | alphabet Monte Carlo |
| `mcmc` | `{}` | options of the selection Monte Carlo (see `ayelenometro.mcmc.mcmc`) |
| `results_dir` | `results` | output folder |

## Variants

| Variant | Divergence between sequences | Selection |
|---|---|---|
| SS | identity | positions ranked by individual ρ, prefix with the highest cumulative ρ |
| MC-SS | identity | positions, Monte Carlo |
| MC-E | identity | position pairs (both differ), Monte Carlo |
| AC-MC-SS | alphabet | positions, Monte Carlo |
| AC-MC-E | alphabet | position pairs, Monte Carlo |

## Controls

For each variant (and for the literature), `n_controls` random position sets of the same size
are evaluated, drawn from the varying columns, with the same training/validation partitions. In
the pair variants each selected position is replaced by a random one, so that the number of
pairs, the number of distinct positions and the pair structure are preserved. The Monte Carlo
weights are randomly reassigned.

## Outputs (`<results_dir>/<name>/`)

| File | Contents |
|---|---|
| `summary.csv` | one row per variant: positions, % of the alignment, ρ, r² (mean ± SD over replicates), RMSE, control r², p-value and the selected positions (and, for pairs, the distinct positions) |
| `positions_<variant>.csv` | selected positions (or pairs), in alignment numbering, with their weight |
| `predictions_<variant>.csv` | observed and predicted value of each validation sequence, per replicate, for the method and each control |
| `controls_<variant>.csv` | positions and r² of each control |
| `individual_rho_SS.csv` | individual and cumulative ρ of each position (SS) |
| `obs_vs_pred_<variant>.png` | observed vs predicted, with the control regression lines |
| `cumulative_rho_SS.png`, `individual_rho_SS.png` | SS selection |
| `alphabet_<name>.npz` | case alphabet (group of each residue per sequence and column) |
| `config_used.json` | configuration, columns used, library versions and date |

r² is the square of Pearson's r between observed and predicted phenotypes (mean over
replicates); RMSE pools all predictions from all replicates.
