"""
Main Program
"""

from preprocessing import prepare_data
from feature_selection import FeatureSelector
from optimization import Optimizer
from evaluation import Evaluator
from statistics import Statistics
from visualization import Visualizer

def main():

    print("="*70)
    print(" Alzheimer's Disease Prediction Framework ")
    print("="*70)

    X_train, X_test, y_train, y_test = prepare_data()

    print("Dataset loaded successfully")

    fs = FeatureSelector()
    optimizer = Optimizer()

    print("Running Feature Selection...")

    print("Running Hyperparameter Optimization...")

    print("Training Models...")

    print("Generating Results...")

    print("Generating Figures...")

    print("Completed Successfully")

if __name__ == "__main__":
    main()