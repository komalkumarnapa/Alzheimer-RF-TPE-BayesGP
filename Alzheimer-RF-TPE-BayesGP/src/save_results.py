"""
Save Results
"""

import os

RESULTS = "../results"

FIGURES = "../figures"

os.makedirs(RESULTS,exist_ok=True)

os.makedirs(FIGURES,exist_ok=True)

print("Results directory created")

print("Figures directory created")