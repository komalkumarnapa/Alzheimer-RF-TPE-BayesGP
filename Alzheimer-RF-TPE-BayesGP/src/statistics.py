"""
Statistical Validation
"""

from statsmodels.stats.contingency_tables import mcnemar

class Statistics:

    @staticmethod

    def mcnemar(table):

        return mcnemar(table)