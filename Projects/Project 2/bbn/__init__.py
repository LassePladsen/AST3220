"""
Big bang nucleosynthesis module for Project 2 of UiO AST3320 Cosmology I
Project 2 link: 
https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/prosjekt-2-ast3220-2024.pdf
"""

# Import submodules
from bbn import BBN
from background import Background
from reaction_rates import ReactionRates
from constants import SI, CGS

# Variable imported from bbn submodule:
# FIG_DIR (directory of stored figures)