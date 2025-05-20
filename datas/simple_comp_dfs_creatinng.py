import pyarrow
import pandas as pd
from double_compressed_jpeg_detection import complete_the_process 
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

for q in q_values:
    print(f"Processing training data for Q: {q}")
    columns = ['flag'] + [f'v{i}' for i in range(ac_components_number * BINS)]
    df = pd.DataFrame(columns=columns)

    for i in range(1, sample_size + 1):
        complete_the_process(f'test_images/images/normal_jpegs/{i}_q{q}.jpg',
                       df, False, n, filter_size, ac_components_number)

    df.to_parquet(f"./datas/single/single_q{q}_teach.parquet", index=True)
    print(f"{sample_size}_single_q{q}.parquet created")
    time_log(start_time)


for q in q_values:
    print(f"Processing test data for Q: {q}")
    columns = ['flag'] + [f'v{i}' for i in range(ac_components_number * BINS)]
    df = pd.DataFrame(columns=columns)

    for i in range(9000, 10001):
        complete_the_process(f'test_images/images/normal_jpegs/{i}_q{q}.jpg',
                       df, False, n, filter_size, ac_components_number)

    df.to_parquet(f"datas/single/single_q{q}_test.parquet", index=True)
    print(f"{sample_size}_single_q{q}.parquet created")
    time_log(start_time)

