import pandas as pd
import numpy as np
import random
import pyarrow
import os
import time
import joblib
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from Program.legacy import egesz_folyamat
from xgboost import XGBRFClassifier


def time_log(start):
    end_time = time.time()
    elapsed_time = end_time - start
    print(f"Elapsed time: {elapsed_time:.2f} seconds.")

n = 7
filter_size = 11
ac_components_number = 9
BINS = 128

q_values = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

sample_size = 2000
unit_pieces = sample_size // (len(q_values) - 1)

for q in q_values:
    start_time = time.time()
    columns = ['flag'] + [f'v{i}' for i in range(ac_components_number * BINS)]
    training_df = pd.DataFrame(columns=columns)

    # Load single compression data
    df = pd.read_parquet(f'datas/single/single_q{q}_teach.parquet')
    single_df = df.head(sample_size)

    time_log(start_time)
    print(f"Single compression data for Q {q} loaded")

    random.seed(42)
    num_combinations = len(q_values) - 1
    all_indices = random.sample(range(sample_size + 1, 2 * sample_size), num_combinations * unit_pieces)
    # 1, sample_size

    assigned_indices = {}
    fact_q_pairs = [fact for fact in q_values if fact != q]
    all_selected_rows = []

    # Load double compression data
    for i, fact in enumerate(fact_q_pairs):
        indices = all_indices[i * unit_pieces:(i + 1) * unit_pieces]
        assigned_indices[(fact, q)] = indices

        df_part = pd.read_parquet(f'datas/double/double_q{fact}_q{q}_teach.parquet')
        selected_rows = df_part.iloc[indices]
        all_selected_rows.append(selected_rows)

    combined_df = pd.concat(all_selected_rows, ignore_index=True)

    # Combine single and double compression data
    training_df = pd.concat([single_df, combined_df], ignore_index=True)

    time_log(start_time)
    print(f"Double compression data for Q {q} loaded")

    y_train = training_df['flag']  # Target variable (flag column)
    x_train = training_df.drop(columns=['flag'])  # Features (excluding flag column)

    param = {'kernel': ['rbf'],
             'C': np.arange(1, 13, 1),
             'gamma': ['scale']}

    print("Grid search for SVM started...")

    SVModel = SVC()
    GridS = GridSearchCV(SVModel, param, cv=5, n_jobs=-1, verbose=1)
    GridS.fit(x_train, y_train)

    print(GridS.best_params_)

    print(confusion_matrix(y_train, GridS.predict(x_train)))
    print(accuracy_score(y_train, GridS.predict(x_train)))

    """
    param = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.5, 0.7, 1.0],
        'colsample_bynode': [0.5, 0.7, 1.0]
    }

    print("Gridsearch elkezdődött...")

    XGBModel = XGBRFClassifier()
    GridS = GridSearchCV(XGBModel, param, cv=5, n_jobs=-1, verbose=1)
    GridS.fit(x_train, y_train)


    print(GridS.best_params_)

    print(confusion_matrix(y_train, GridS.predict(x_train)))
    print(accuracy_score(y_train, GridS.predict(x_train)))
    """

    # Save model
    joblib.dump(GridS, f'models/3_2/q{q}_model_3_2.joblib')
    print(f"Model saved: q{q}")
    time_log(start_time)
