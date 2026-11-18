"""
Ayelenómetro: identification of the positions that modulate a quantitative phenotype from a
multiple sequence alignment, and phenotype prediction.

Usage:
    from ayelenometro import Case, run_case
    case = Case(fasta="HIV_FPV.fasta", name="HIV_FPV", phenotype="FPV-IP (%)", phenotype_index=-1,
                create_alphabet=True)
    summary = run_case(case)
"""
from .config import Case, VARIANTS, DEFAULT_VARIANTS
from .run import run_case, run_cases

__version__ = "1.0.0"
__all__ = ["Case", "VARIANTS", "DEFAULT_VARIANTS", "run_cases", "run_case"]
