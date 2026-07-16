"""
Configuration File
"""

import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_JOBS = -1

CSV_PATH = "../data/alzheimers_disease_data.csv"
TARGET_COL = "Diagnosis"

TEST_SIZE = 0.30

BOOTSTRAP_ITER = 1000

N_TRIALS_TPE = 40
N_CALLS_BAYES = 35

CLASS_NAMES = ["Non-AD", "AD"]

RESULT_DIR = "../results"
FIGURE_DIR = "../figures"