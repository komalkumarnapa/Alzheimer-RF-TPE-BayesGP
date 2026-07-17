"""
Feature Selection Module
"""

from boruta import BorutaPy

from sklearn.feature_selection import RFECV

from sklearn.ensemble import RandomForestClassifier

from config import *

from utils import ReliefFSelector

class FeatureSelector:

    def relief(self):

        return ReliefFSelector()

    def boruta(self):

        return BorutaPy(

            estimator=RandomForestClassifier(

                n_estimators=600,

                random_state=SEED,

                n_jobs=N_JOBS

            ),

            n_estimators='auto'

        )

    def rfecv(self):

        return RFECV(

            estimator=RandomForestClassifier(

                random_state=SEED,

                n_jobs=N_JOBS

            ),

            cv=10,

            scoring="accuracy"

        )
