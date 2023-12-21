"""
Бейзлайн задачи прогнозирования удоев.
В качестве прогнозного значения каждого из 8 удоев подставляется медианное значение известных контрольных
удоев животного.
"""

import os
import json
from typing import Any

import tqdm
import numpy as np
import pandas as pd


def get_animals_milk_yield_median(animal_id: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """
    Получить медианное значение по всем контрольным дойкам животного

    @param animal_id: идентификатор животного
    @param train_df: обучающая выборка
    @param test_df: тестовая выборка

    @returns: медианное значение всех контрольных доек животного
    """

    milk_yield_values = []

    # Собрать 10 известных значений контрольных доек из обучающей выборки
    df_train = train_df[train_df['animal_id'] == animal_id]
    for index, row in df_train.iterrows():
        for i in range(1, 11):
            milk_yield_values.append(row[f'milk_yield_{i}'])

    # Собрать 2 известных значения контрольных доек из тестовой выборки
    df_test = test_df[test_df['animal_id'] == animal_id]
    for index, row in df_test.iterrows():
        for i in range(1, 3):
            milk_yield_values.append(row[f'milk_yield_{i}'])

    median = float(np.nanmedian(milk_yield_values))
    return median


def fit() -> Any:
    """
    Обучить модель прогнозирования
    @return: Обученная модель прогнозирования
    """
    if os.path.exists('model-baseline.json'):
        with open('model-baseline.json', 'r') as f:
            return json.load(f)

    train_dataset = pd.read_csv(os.path.join('data', 'train.csv'))
    test_dataset = pd.read_csv(os.path.join('data', 'X_test_public.csv'))
    # pedigree = pd.read_csv(os.path.join('dataset', 'pedigree.csv'))

    # Получить медианные значения контрольных удоев по каждому животному из тестовой выборки
    print('Вычисление медианных значений для каждого временного ряда в выборке')
    medians_of_animals = {
        animal_id: get_animals_milk_yield_median(animal_id, train_dataset, test_dataset)
        for animal_id in tqdm.tqdm(list(sorted(set(test_dataset['animal_id']))))
    }

    with open('model-baseline.json', 'w') as f:
        json.dump(medians_of_animals, f)

    return medians_of_animals


def predict(model: Any, test_dataset_path: str) -> pd.DataFrame:
    """
    Построить прогноз с помощью модели прогнозирования для датасета, заданного по имени файла

    @param model: Обученная ранее модель прогнозирования
    @param test_dataset_path: Путь к тестовому датасету
    @return: Датафрейм с построенным прогнозом, заданного формата
    """

    test_dataset = pd.read_csv(test_dataset_path)
    submission = pd.DataFrame(test_dataset[['animal_id', 'lactation']])

    def get_predict_value(row):
        value = model.get(row['animal_id'], None)
        if value is not None and not np.isnan(value):
            return value

        return np.nanmedian([row['milk_yield_1'], row['milk_yield_2']])

    submission['milk_yield_3'] = test_dataset.apply(get_predict_value, axis=1)
    submission['milk_yield_4'] = submission['milk_yield_3']
    submission['milk_yield_5'] = submission['milk_yield_3']
    submission['milk_yield_6'] = submission['milk_yield_3']
    submission['milk_yield_7'] = submission['milk_yield_3']
    submission['milk_yield_8'] = submission['milk_yield_3']
    submission['milk_yield_9'] = submission['milk_yield_3']
    submission['milk_yield_10'] = submission['milk_yield_3']
    return submission


if __name__ == '__main__':
    _model = fit()

    _submission = predict(_model, os.path.join('data', 'X_test_public.csv'))
    _submission.to_csv(os.path.join('data', 'submission.csv'), sep=',', index=False)

    # _submission_private = predict(_model, os.path.join('private', 'X_test_private.csv'))
    # _submission_private.to_csv(os.path.join('data', 'submission_private.csv'), sep=',', index=False)
