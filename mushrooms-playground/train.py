# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
import datetime
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
from collections import Counter

from xgboost.callback import TrainingCallback

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef


class EvalLogger(TrainingCallback):
    def __init__(self, period, epochs):
        super().__init__()
        self.period = period
        self.epochs = epochs
        self.__start = datetime.datetime.now()

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        if epoch % self.period == 0:
            eta = (datetime.datetime.now() - self.__start) * self.epochs / (1 + epoch)
            print(f"Epoch {epoch}, eta {eta}")


# run with 'poetry run python mushrooms-playground/train.py'
if __name__ == "__main__":
    original_train = pd.read_csv("data/mushrooms-playground/train.csv")
    original_test = pd.read_csv("data/mushrooms-playground/test.csv")

    print("Loaded datasets")

    # Data cleaning

    cleaned_train = original_train.dropna(subset=['cap-diameter'])
    cleaned_test = original_test

    columns_to_clean = ['cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed', 'gill-attachment',
                        'gill-spacing', 'gill-color', 'stem-root', 'stem-surface', 'stem-color', 'veil-type',
                        'veil-color', 'has-ring', 'ring-type', 'spore-print-color', 'habitat', 'season']

    counters = {c: Counter(cleaned_train[c]) for c in columns_to_clean}
    thresholds = {'cap-shape': 41,
                  'cap-surface': 8,
                  'cap-color': 10,
                  'does-bruise-or-bleed': 547082,
                  'gill-attachment': 5,
                  'gill-spacing': 3,
                  'gill-color': 9,
                  'stem-root': 2,
                  'stem-surface': 7,
                  'stem-color': 6,
                  'veil-type': 2,
                  'veil-color': 3,
                  'has-ring': 747982,  # maybe 2
                  'ring-type': 9,
                  'spore-print-color': 3,
                  'habitat': 7,
                  'season': 0
                  }

    for c, threshold in thresholds.items():
        counter = {k: v for k, v in counters[c].items() if v >= threshold}
        cleaned_train = cleaned_train[cleaned_train[c].isin(counter)]

    print("Cleaned datasets")


    # Prepare datasets

    def sample_cols(df):
        df["cap-diameter"] = np.sqrt(df["cap-diameter"])
        df = df[["cap-diameter", "stem-height", "stem-width"] + list(thresholds.keys())]
        for c in thresholds.keys():
            # df.loc[:, c] = df[c].map(counters[c])
            df.loc[:, c] = df[c].map({k: i for i, k in enumerate(counters[c].keys())})
        df = df.astype({c: float for c in thresholds.keys()})
        return df


    train_x = cleaned_train.drop(columns=["id", "class"])

    train_y = cleaned_train[["class"]]
    train_y.loc[:, "class"] = train_y["class"].map({'e': 0, 'p': 1})
    train_y = train_y.astype({"class": float})

    test_ids = original_test["id"]
    test_x = original_test.drop(columns=["id"])

    train_x = sample_cols(train_x)
    test_x = sample_cols(test_x)

    print(train_x.shape, train_y.shape, test_x.shape)


    def mcc(y_pred, dtrain):
        y_true = dtrain.get_label()
        y_pred = np.round(y_pred)  # Convert probabilities to binary predictions
        mcc = matthews_corrcoef(y_true, y_pred)
        return 'MCC', mcc


    def train(dataset, params, epochs):
        start = dt.datetime.now()
        logger = EvalLogger(100, epochs)
        bst = xgb.train(params, dataset, epochs, callbacks=[logger])
        bst.save_model("model")
        end = dt.datetime.now()
        print("Training done in ", end - start)
        return bst


    def predict(bst, test_df):
        preds = bst.predict(test_df)
        return [1 if prob > 0.5 else 0 for prob in preds]


    # Validation

    """
    validation_x, test_validation_x, validation_y, test_validation_y = (
        train_test_split(train_x, train_y, test_size=0.2, random_state=239723))

    validation_dtrain = xgb.DMatrix(validation_x, label=validation_y, enable_categorical=True)
    validation_dtest = xgb.DMatrix(test_validation_x, label=test_validation_y, enable_categorical=True)
    
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'max_depth': 4,                  # Maximum depth of the trees
        'eta': 0.1,                      # Learning rate
        'subsample': 0.7,                # Subsample ratio of the training instances
        'colsample_bytree': 1.0,         # Subsample ratio of columns when constructing each tree
        'eval_metric': 'logloss',        # Evaluation metric for binary classification
        'seed': 32830283                 # Random seed for reproducibility
    }
    print('Test started')
    bst = train(validation_dtrain, params, 1000)
    
    print(classification_report(test_validation_y, predict(bst, validation_dtest)))
    """

    # Full training

    dtrain = xgb.DMatrix(train_x, label=train_y, enable_categorical=True)
    dtest = xgb.DMatrix(test_x, enable_categorical=True)

    params = {
        'n_estimators': 1000,
        'enable_categorical': True,
        'max_depth': 9,
        'eta': 0.005,
        'subsample': 0.8,
        'colsample_bytree': 1.0,
        'min_child_weight': 12,
        'objective': 'binary:logistic',  # Binary classification
        'eval_metric': 'logloss',  # Evaluation metric for binary classification
        'seed': 32830283,  # Random seed for reproducibility
    }
    print('training started')
    bst = train(dtrain, params, 40000)

    print(classification_report(train_y, predict(bst, dtrain)))

    # Predict test

    y_pred = predict(bst, dtest)

    result = pd.DataFrame([[i, y] for i, y in zip(test_ids, y_pred)], columns=["id", "class"])
    result = result.set_index('id')
    result.loc[:, "class"] = result["class"].map({0: 'e', 1: 'p'})

    result.to_csv("data/mushrooms-playground/submission.csv")
    print("Saved")
