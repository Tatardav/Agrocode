
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
from typing import Dict, List, Tuple, Optional
import calendar
from datetime import datetime, timedelta
from dataclasses import dataclass
import xgboost as xgb
import tqdm
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
import re




def fit():
    train_dataset = pd.read_csv('train.csv')
    pedigree=pd.read_csv('pedigree.csv')
    train_dataset = train_dataset.merge(pedigree, on=['animal_id'], how='left')
    train_dataset['father_id'].fillna('ID_0000000001', inplace=True)
    train_dataset['mother_id'].fillna('ID_0000000001', inplace=True)
    train_dataset.dropna(inplace=True)
    train_dataset.calving_date = pd.to_datetime(train_dataset.calving_date).map(datetime.toordinal)
    train_dataset.birth_date = pd.to_datetime(train_dataset.birth_date).map(datetime.toordinal)
    train_dataset['father_id'] = train_dataset['father_id'].apply(lambda x: re.sub(r'ID_(\d+)', r'\1', x))
    train_dataset['father_id']=train_dataset['father_id'].astype('float')
    train_dataset['mother_id'] = train_dataset['mother_id'].apply(lambda x: re.sub(r'ID_(\d+)', r'\1', x))
    train_dataset['mother_id']=train_dataset['mother_id'].astype('float')
    train_dataset['month'] = pd.to_datetime(train_dataset.calving_date).dt.month
    regressor = [
        LGBMRegressor(
            learning_rate=0.05,
            tree_learner='serial',
            n_estimators=180,
            max_depth=16,
            random_state=42,
            n_jobs=10,
            device_type='cpu',  # or CPU
            gpu_platform_id=0,
            gpu_device_id=0,
            max_bin=63,  # Максимальное количество бинов, в которые будут группироваться значения признаков
            # - см.  https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html
            gpu_use_dp=False,  # По возможности старайтесь использовать обучение с одинарной точностью
            # ( ), потому что большинство графических процессоров
            # (особенно потребительские графические процессоры NVIDIA) имеют низкую
            # производительность при двойной точности.
            #num_leaves=31,
            boosting='gbdt',  # gbdt быстрее существенно чем dart на gpu
        ) for _ in range(8)
    ]
    
    for i in range(8):
        if i <= 1:
            X = train_dataset.loc[:,['lactation', 'calving_date', 'farm', 'farmgroup', 'birth_date', f'milk_yield_{1 + i}', f'milk_yield_{2 + i}', 'mother_id', 'month']]
            y = train_dataset[f'milk_yield_{i + 3}']
            regressor[i].fit(X.values, y.values)

        else:
            X = train_dataset.loc[:,['lactation', 'calving_date', 'farm', 'farmgroup', 'birth_date', 'milk_yield_1', 'milk_yield_2', 'mother_id', 'month']]
            y = train_dataset[f'milk_yield_{i + 3}']
            regressor[i].fit(X.values, y.values)

    return regressor


def predict(regressor, test_dataset_path):
    test_dataset = pd.read_csv(test_dataset_path)
    pedigree=pd.read_csv('pedigree.csv')
    
    
    test_dataset = test_dataset.merge(pedigree, on=['animal_id'], how='left')
    
    test_dataset['father_id'].fillna('ID_0000000001', inplace=True)
    
    test_dataset['mother_id'].fillna('ID_0000000001', inplace=True)
    test_dataset.dropna(inplace=True)
    test_dataset.calving_date = pd.to_datetime(test_dataset.calving_date).map(datetime.toordinal)
    test_dataset.birth_date = pd.to_datetime(test_dataset.birth_date).map(datetime.toordinal)
    test_dataset['father_id'] = test_dataset['father_id'].apply(lambda x: re.sub(r'ID_(\d+)', r'\1', x))
    test_dataset['father_id']=test_dataset['father_id'].astype('float')
    test_dataset['mother_id'] = test_dataset['mother_id'].apply(lambda x: re.sub(r'ID_(\d+)', r'\1', x))
    test_dataset['mother_id']=test_dataset['mother_id'].astype('float')
    test_dataset['month'] = pd.to_datetime(test_dataset.calving_date).dt.month
    submission = test_dataset.loc[:, 'animal_id':'lactation']
 
    for i in range(8):
        if i <= 1:
            X = test_dataset.loc[:,['lactation', 'calving_date', 'farm', 'farmgroup', 'birth_date', f'milk_yield_{1 + i}', f'milk_yield_{2 + i}', 'mother_id', 'month']]
            preds = regressor[i].predict(X)
            test_dataset[f'milk_yield_{i + 3}'] = preds
            submission[f'milk_yield_{i + 3}'] = preds
        else:
            X = test_dataset.loc[:,['lactation', 'calving_date', 'farm', 'farmgroup', 'birth_date', 'milk_yield_1', 'milk_yield_2', 'mother_id', 'month']]
            preds = regressor[i].predict(X)
            test_dataset[f'milk_yield_{i + 3}'] = preds
            submission[f'milk_yield_{i + 3}'] = preds

    return submission


if __name__ == '__main__':
    _model = fit()

    _submission = predict(_model, os.path.join('data', 'X_test_public.csv'))
    _submission.to_csv(os.path.join('data', 'submission.csv'), sep=',', index=False)

    # _submission_private = predict(_model, os.path.join('private', 'X_test_private.csv'))
    # _submission_private.to_csv(os.path.join('data', 'submission_private.csv'), sep=',', index=False)