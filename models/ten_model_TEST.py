import os
import pyarrow
import pandas as pd
import time
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from Program.legacy import egesz_folyamat


def time_log(start):
    end_time = time.time()
    elapsed_time = end_time - start
    print(f"Elapsed time: {elapsed_time:.2f} seconds.")

start = time.time()

n = 7
filter_size = 11
ac_components_number = 9
BINS = 128

q_values = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

sample_size = 1000

result = pd.DataFrame(index=q_values, columns=q_values)

for q in q_values:
    model = joblib.load(f'models/2_2/q{q}_model_2_2.joblib')

    single_testing_df = pd.read_parquet(f'datas/single/single_q{q}_test.parquet')
    single_testing_df = single_testing_df.head(sample_size)

    for q1 in q_values:
        if q1 == q:
            continue

        double_testing_df = pd.read_parquet(f'datas/double/double_q{q1}_q{q}_test.parquet')
        double_testing_df = double_testing_df.head(sample_size)

        columns = ['flag'] + [f'v{i}' for i in range(ac_components_number * BINS)]
        testing_df = pd.DataFrame(columns=columns)
        testing_df = pd.concat([single_testing_df, double_testing_df], axis=0, ignore_index=True)

        y_train = testing_df['flag']  # Target variable
        x_train = testing_df.drop(columns=['flag'])  # Features

        print(f"{q1}-{q}:")
        print(confusion_matrix(y_train, model.predict(x_train)))
        score = accuracy_score(y_train, model.predict(x_train))
        print(score)

        result.at[q, q1] = score

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(result)
