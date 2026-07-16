"""
Data preprocessing module
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE

from config import *

def load_dataset():

    df = pd.read_csv(CSV_PATH)

    y = df[TARGET_COL]

    X = df.drop(columns=[TARGET_COL])

    return X,y


def preprocess(X_train,X_test):

    num_cols = X_train.select_dtypes(include=np.number).columns

    cat_cols = [c for c in X_train.columns if c not in num_cols]

    numeric = Pipeline([

        ("imputer",SimpleImputer(strategy="median")),

        ("scaler",MinMaxScaler())

    ])

    categorical = Pipeline([

        ("imputer",SimpleImputer(strategy="most_frequent")),

        ("encoder",OrdinalEncoder(handle_unknown="use_encoded_value",
                                  unknown_value=-1)),

        ("scaler",MinMaxScaler())

    ])

    preprocessor = ColumnTransformer([

        ("num",numeric,num_cols),

        ("cat",categorical,cat_cols)

    ])

    X_train = preprocessor.fit_transform(X_train)

    X_test = preprocessor.transform(X_test)

    return X_train,X_test


def apply_smote(X,y):

    smote = SMOTE(random_state=SEED)

    return smote.fit_resample(X,y)


def prepare_data():

    X,y = load_dataset()

    X_train,X_test,y_train,y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        stratify=y,

        random_state=SEED

    )

    X_train,X_test = preprocess(X_train,X_test)

    X_train,y_train = apply_smote(X_train,y_train)

    return X_train,X_test,y_train,y_test