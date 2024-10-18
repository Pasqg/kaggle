import datetime
import pandas as pd
import datetime as dt
from collections import Counter
import re

from xgboost.callback import TrainingCallback

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)

import xgboost as xgb

from sklearn.model_selection import GridSearchCV


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


def extract_cylinders(description):
    if 'V6' in description:
        return 6
    if 'V4' in description:
        return 4
    if 'V8' in description:
        return 8
    matches = re.search(r"(\d+) cylinder|V(\d+)", description)
    if matches is not None and len(matches.groups()) > 0:
        return float(matches[1]) if matches[1] is not None else float('NaN')

    return float('NaN')


def extract_volume(description):
    matches = re.search(r"(\d+l)|(\d+\.\d+l)", description)
    if matches is not None and len(matches.groups()) > 0:
        group = matches.group(0)[:-1]
        return float(group) if group is not None else float('NaN')

    return float('NaN')


def extract_hps(description):
    matches = re.search(r"(\d+hp)|(\d+\.\d+hp)", description)
    if matches is not None and len(matches.groups()) > 0:
        group = matches.group(0)[:-2]
        return float(group) if group is not None else float('NaN')

    return float('NaN')


# The engine.md type can be split in various features, like additional fuel type, size of the engine.md, power
def feature_extraction(counters, simple_features, complex_features):
    features = {f: {} for f in simple_features + complex_features}
    for description in counters['engine'].keys():
        desc = description.lower()
        for simple_feature in simple_features:
            features[simple_feature][description] = simple_feature in desc
        features['cylinders'][description] = extract_cylinders(desc)
        features['volume'][description] = extract_volume(desc)
        features['hps'][description] = extract_hps(desc)
    return features


def create_feature_counters(df, columns):
    return {c: Counter(df[c]) for c in columns}


# run with 'poetry run python mushrooms-playground/train.py'
if __name__ == "__main__":
    original_train = pd.read_csv("data/used-cars-prices/train.csv")
    original_test = pd.read_csv("data/used-cars-prices/test.csv")

    print("Loaded datasets")

    # Data cleaning
    train_y = original_train[["price"]]
    test_ids = original_test["id"]
    cleaned_train = original_train.drop(columns=['price'])
    cleaned_test = original_test
    concatenated_df = pd.concat([cleaned_train, cleaned_test])

    # Extract features from engine description
    columns_to_enumerate = ['brand', 'model', 'model_year', 'milage', 'fuel_type', 'transmission', 'ext_col',
                            'int_col', 'accident', 'clean_title']
    counters = create_feature_counters(concatenated_df, columns_to_enumerate + ['engine'])

    simple_features = ['gasoline', 'diesel', 'electric_motor', 'electric_fuel', 'hybrid', 'turbo', 'flex']
    complex_features = ['volume', 'cylinders', 'hps']

    feature_maps = feature_extraction(counters, simple_features, complex_features)
    for feature, map_values in feature_maps.items():
        cleaned_train.loc[:, feature] = cleaned_train['engine'].map(map_values)
        cleaned_test.loc[:, feature] = cleaned_test['engine'].map(map_values)

    numerical_columns = {'model_year', 'milage'}.union(set(complex_features))
    boolean_columns = set(simple_features)
    columns_to_enumerate = list(
        set(columns_to_enumerate + list(feature_maps.keys())) - numerical_columns - boolean_columns)
    counters = {c: Counter(cleaned_train[c]) for c in columns_to_enumerate}


    def remove_cols(df):
        for c in counters.keys():
            df.loc[:, c] = df[c].map({k: i for i, k in enumerate(counters[c].keys())})
        df = df[columns_to_enumerate + list(numerical_columns) + list(boolean_columns)]

        for c in numerical_columns:
            df.loc[:, c] = df[c].fillna(-1.0)

        df = df.astype({c: float for c in list(numerical_columns) + columns_to_enumerate})

        return df


    cleaned_train = remove_cols(cleaned_train)
    cleaned_test = remove_cols(cleaned_test)

    print("Cleaned datasets")

    with open("head.txt", "w") as file:
        file.write(str(cleaned_train.head(30)))

    # Prepare for XGBoost
    train_y = train_y.astype({"price": float})

    train_x = cleaned_train
    test_x = cleaned_test

    print("Test cols", cleaned_test.columns)
    print("Train cols", cleaned_train.columns)

    print(train_x.shape, train_y.shape, test_x.shape)


    def train(dataset, params, epochs, name="model"):
        start = dt.datetime.now()
        logger = EvalLogger(100, epochs)
        bst = xgb.train(params, dataset, epochs, callbacks=[logger])
        bst.save_model("models/used-cars-prices.xgboost")
        end = dt.datetime.now()
        print("Training done in ", end - start)
        return bst


    def predict(bst, test_df):
        preds = bst.predict(test_df)
        return preds


    # Cross Validation
    dtrain_x = xgb.DMatrix(train_x, enable_categorical=True)
    dtrain_y = xgb.DMatrix(train_y, enable_categorical=True)

    params = {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'eta': 0.1,
        'subsample': 0.7,
        'colsample_bytree': 1.0,
        'eval_metric': 'logloss',
        'seed': 32830283
    }
    print('Test started')

    param_grid = [
        {"max_depth": [3, 4, 5],
         "n_estimators": [670],
         "learning_rate": [0.01]},
    ]
    regressor = xgb.XGBRegressor(eval_metric='rmse')
    search = GridSearchCV(regressor, param_grid, cv=5, verbose=20).fit(train_x, train_y)
    best_params = search.best_params_

    print("The best hyperparameters are ", best_params)

    # Full training
    regressor = xgb.XGBRegressor(learning_rate=best_params["learning_rate"],
                                 n_estimators=best_params["n_estimators"],
                                 max_depth=best_params["max_depth"],
                                 eval_metric='rmsle')

    regressor.fit(train_x, train_y, verbose=5)
    predictions = regressor.predict(test_x)

    result = pd.DataFrame([[i, y] for i, y in zip(test_ids, predictions)], columns=["id", "price"])
    result = result.set_index('id')

    result.to_csv("data/used-cars-prices/submission.csv")
    print("Saved")
