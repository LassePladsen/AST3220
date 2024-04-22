"""
Big bang nucleosynthesis module for Project 2 of UiO AST3320 Cosmology I
Project 2 link: 
https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/prosjekt-2-ast3220-2024.pdf

Classes:
bbn.BBN
background.Background
reaction_rates.ReactionRates
constants.SI
constants.CGS

Constants:
FIG_DIR (directory of stored figures)
"""

from pathlib import Path

# Directory to save figures
FIG_DIR = Path(__file__).parents[1] / "figures"

# Create this directory if it doesn't already exist
if not FIG_DIR.exists():
    FIG_DIR.mkdir()
