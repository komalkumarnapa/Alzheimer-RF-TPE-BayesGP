"""
Evaluation Metrics
"""

from sklearn.metrics import *

class Evaluator:

    @staticmethod
    def evaluate(model,X_test,y_test):

        pred=model.predict(X_test)

        prob=model.predict_proba(X_test)[:,1]

        return {

            "Accuracy":accuracy_score(y_test,pred),

            "Precision":precision_score(y_test,pred),

            "Recall":recall_score(y_test,pred),

            "F1":f1_score(y_test,pred),

            "AUC":roc_auc_score(y_test,prob)

        }