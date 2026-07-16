"""
Hyperparameter Optimization
"""

import optuna

from skopt import gp_minimize

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from config import *

class Optimizer:

    def objective(self,trial,X,y):

        params={

            "n_estimators":trial.suggest_int("n_estimators",100,600),

            "max_depth":trial.suggest_int("max_depth",4,24),

            "min_samples_split":trial.suggest_int("min_samples_split",2,20),

            "min_samples_leaf":trial.suggest_int("min_samples_leaf",1,10),

            "max_features":trial.suggest_categorical(

                "max_features",

                ["sqrt","log2",None]

            )

        }

        rf=RandomForestClassifier(

            **params,

            random_state=SEED,

            n_jobs=N_JOBS

        )

        score=cross_val_score(

            rf,

            X,

            y,

            cv=5,

            scoring="accuracy"

        ).mean()

        return 1-score