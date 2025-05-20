import os
import pyarrow
import pandas as pd
from Program import legacy
import time


def time_log(start):
    end_time = time.time()
    elapsed_time = end_time - start
    print(f"Elapsed time: {elapsed_time:.2f} seconds.")

start_time = time.time()

BINS = 128

n = 7
filter_size = 11

sample_size = 4000
ac_components_number = 9

q_values = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

print("Processing started")

for q1 in q_values:
    for q2 in [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]:
        if q1 == q2:
            continue

        columns = ['flag'] + [f'v{i}' for i in range(ac_components_number * BINS)]
        df = pd.DataFrame(columns=columns)

        for i in range(1, sample_size + 1):
            legacy.egesz_folyamat(f'test_images/images/double_compressed_jpegs/{q1}_{q2}/{i}_q{q1}_q{q2}.jpg',
                                  df, True, n, filter_size, ac_components_number)

        df.to_parquet(f"datas/double/double_q{q1}_q{q2}_teach.parquet", index=True)
        print(f"teach_double_q{q1}_q{q2}.parquet created")

        columns = ['flag'] + [f'v{i}' for i in range(ac_components_number * BINS)]
        df2 = pd.DataFrame(columns=columns)


        for i in range(9000, 10001):
            legacy.egesz_folyamat(f'test_images/images/double_compressed_jpegs/{q1}_{q2}/{i}_q{q1}_q{q2}.jpg',
                                  df2, True, n, filter_size, ac_components_number)

        df2.to_parquet(f"datas/double/double_q{q1}_q{q2}_test.parquet", index=True)
        print(f"test_double_q{q1}_q{q2}.parquet created")
        time_log(start_time)

