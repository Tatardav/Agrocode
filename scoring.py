import os
from os.path import join

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def calculate_score(test_dataset_path, submission_path):
    test_dataset = pd.read_csv(test_dataset_path)
    submission = pd.read_csv(submission_path)

    merged_Y = test_dataset.merge(
        submission, left_on=['animal_id', 'lactation'], right_on=['animal_id', 'lactation'], how='outer'
    )
    mean_squared_errors = []

    median_value = np.nanmedian(submission[[f'milk_yield_{i}' for i in range(3, 11)]].values)

    for index, row in merged_Y.iterrows():
        # yapf: disable
        arr_real = (
            row[
                [f'milk_yield_{i}_x' for i in range(3, 11)]
            ].fillna(method='ffill')
             .fillna(method='bfill')
        )
        arr_predict = (
            row[
                [f'milk_yield_{i}_y' for i in range(3, 11)]
            ].fillna(method='ffill')
             .fillna(method='bfill')
             .fillna(value=median_value)
             .fillna(value=0)
        )
        # yapf: enable

        mean_squared_errors.append(mean_squared_error(arr_real, arr_predict))

    rmse_score = np.sqrt(np.mean(mean_squared_errors))

    return rmse_score


if __name__ == '__main__':
    # print('full score', calculate_score(join('private', 'y_test.csv'), join('data', 'submission.csv')))
    print(
        'public score',
        calculate_score(join('private', 'y_test_public.csv'), join('../milk_forecasting/data', 'submission.csv'))
    )
    print(
        'private score',
        calculate_score(join('private', 'y_test_private.csv'), join('../milk_forecasting/data', 'submission_private.csv'))
    )
