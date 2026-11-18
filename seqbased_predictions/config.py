"""Configuration of a case and validation of the inputs."""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import os
import zlib

import numpy as np

from .data import read_fasta, phenotypes_from_ids

# Variants: name -> (encoding, mode, method)
#   encoding: "identity" (Hamming distance) or "alphabet" (amino acid groups)
#   mode: "singles" (positions) or "pairs" (position pairs, epistasis)
#   method: "ordering" (ordered selection) or "mcmc" (Monte Carlo)
VARIANTS = {
    "SS":       ("identity", "singles", "ordering"),
    "MC-SS":    ("identity", "singles", "mcmc"),
    "MC-E":     ("identity", "pairs",   "mcmc"),
    "AC-MC-SS": ("alphabet", "singles", "mcmc"),
    "AC-MC-E":  ("alphabet", "pairs",   "mcmc"),
    # optional (not part of the manuscript)
    "E":        ("identity", "pairs",   "ordering"),
    "AC-SS":    ("alphabet", "singles", "ordering"),
    "AC-E":     ("alphabet", "pairs",   "ordering"),
}
DEFAULT_VARIANTS = ["SS", "MC-SS", "MC-E", "AC-MC-SS", "AC-MC-E"]


@dataclass
class Case:
    """
    Configuration of a case.

    Required
      fasta            multiple sequence alignment in FASTA; the phenotype goes in the ID of each sequence
      name             case name (used in the output files)
      phenotype        phenotype name (for the plots)
      phenotype_index  position of the phenotype in the ID split by '_' (e.g. -1 = the last field)

    Optional
      create_alphabet  if the case alphabet does not exist: True creates it, False stops the run.
                       If it already exists, it is used (not created again).
      alphabet         path to the alphabet (.npz); default <results_dir>/<name>/alphabet_<name>.npz
      literature       list of literature positions, in alignment numbering (1-based)
      max_gaps         maximum fraction of gaps for a column to be used (all variants)
      seed             case seed (all stages derive from it)
      lambda_reg       λ of the MCMC regularization
      variants         variants to run (see VARIANTS)
      replicates       training/validation replicates
      train_fraction   fraction of sequences used for training
      n_controls       random position sets per variant
      n_permutations   controls with permuted phenotypes per variant (0 = not run;
                       each one repeats the selection, so it is costly)
      alphabet_iterations, alphabet_alpha   parameters of the alphabet Monte Carlo
      mcmc             extra options for the MCMC (see mcmc.mcmc)
      results_dir
    """
    fasta: str
    name: str
    phenotype: str
    phenotype_index: int
    create_alphabet: bool = False
    alphabet: Optional[str] = None
    literature: Optional[List[int]] = None
    max_gaps: float = 0.5
    seed: int = 42
    lambda_reg: float = 1.0
    variants: List[str] = field(default_factory=lambda: list(DEFAULT_VARIANTS))
    replicates: int = 10
    train_fraction: float = 0.9
    n_controls: int = 100
    n_permutations: int = 0
    alphabet_iterations: int = 1000
    alphabet_alpha: float = 0.1
    mcmc: Dict = field(default_factory=dict)
    results_dir: str = "results"

    # ------------------------------------------------------------------ paths
    @property
    def folder(self):
        return os.path.join(self.results_dir, self.name)

    @property
    def alphabet_path(self):
        return self.alphabet or os.path.join(self.folder, f"alphabet_{self.name}.npz")

    def uses_alphabet(self):
        return any(VARIANTS[v][0] == "alphabet" for v in self.variants)

    def as_dict(self):
        return asdict(self)

    # ------------------------------------------------------------------ seeds
    def stage_seed(self, stage: str) -> np.random.SeedSequence:
        """
        Independent seed for each stage of the case (alphabet, partitions, each variant, its
        controls...). Adding or removing a variant does not change the others.
        """
        return np.random.SeedSequence([int(self.seed), zlib.crc32(stage.encode("utf-8"))])

    def rng(self, stage: str) -> np.random.Generator:
        return np.random.default_rng(self.stage_seed(stage))

    def integer_seed(self, stage: str) -> int:
        return int(self.stage_seed(stage).generate_state(1)[0])

    # ------------------------------------------------------------------ validation
    def validate(self):
        """Checks the configuration and the inputs before running. Raises ValueError with a clear message."""
        errors = []
        for attr in ("fasta", "name", "phenotype"):
            if not isinstance(getattr(self, attr), str) or not getattr(self, attr).strip():
                errors.append(f"'{attr}' must be a non-empty string")
        if not isinstance(self.phenotype_index, (int, np.integer)):
            errors.append("'phenotype_index' must be an integer")
        unknown = [v for v in self.variants if v not in VARIANTS]
        if unknown:
            errors.append(f"unknown variants: {unknown}. Available: {list(VARIANTS)}")
        if not 0 <= self.max_gaps <= 1:
            errors.append("'max_gaps' must be between 0 and 1")
        if not 0 < self.train_fraction < 1:
            errors.append("'train_fraction' must be between 0 and 1")
        if self.replicates < 1 or self.n_controls < 0 or self.n_permutations < 0:
            errors.append("'replicates' must be >= 1; 'n_controls' and 'n_permutations' >= 0")
        if errors:
            raise ValueError(f"Configuration of {self.name!r}:\n  - " + "\n  - ".join(errors))

        if not os.path.exists(self.fasta):
            raise ValueError(f"[{self.name}] FASTA not found: {self.fasta}")
        ids, sequences = read_fasta(self.fasta)
        if len(ids) < 4:
            raise ValueError(f"[{self.name}] the FASTA has {len(ids)} sequences; at least 4 are needed")
        try:
            phen = phenotypes_from_ids(ids, self.phenotype_index)
        except (ValueError, IndexError) as e:
            raise ValueError(f"[{self.name}] could not read the phenotype (field {self.phenotype_index} of the ID "
                             f"split by '_'), e.g. in '{ids[0]}': {e}") from None
        if not np.all(np.isfinite(phen)):
            raise ValueError(f"[{self.name}] some phenotypes are not finite numbers")
        # if len(set(ids)) != len(ids):
        #     raise ValueError(f"[{self.name}] there are repeated IDs in the FASTA")
        L = sequences.shape[1]
        if self.literature is not None:
            outside = [p for p in self.literature if not (isinstance(p, (int, np.integer)) and 1 <= p <= L)]
            if outside:
                raise ValueError(f"[{self.name}] literature positions outside the alignment (1..{L}): {outside}")
        if self.uses_alphabet() and not os.path.exists(self.alphabet_path) and not self.create_alphabet:
            raise ValueError(f"[{self.name}] the alphabet {self.alphabet_path} does not exist and create_alphabet=False. "
                             "Set create_alphabet=True to create it (it may take a while) or give its path in 'alphabet'.")
        return ids, sequences, phen
