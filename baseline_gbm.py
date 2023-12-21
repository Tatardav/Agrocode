import os
from typing import Dict, List, Tuple, Optional
import calendar
from datetime import datetime, timedelta
from dataclasses import dataclass

import tqdm
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


PUBLIC_TEST_DATASET_NAME = 'X_test_public.csv'


@dataclass
class AnimalMetadata:
    """Метаданные животного, включая временной ряд"""

    # Временной ряд известных контрольных удоев
    ts: pd.DataFrame
    # Временной ряд контрольных удоев животного, ресемплированный по границе 30 дней с заполненными пропусками
    ts_resampled: pd.DataFrame
    # Ферма
    farm: int
    # Группа ферм
    farmgroup: int
    # Дата рождения животного
    birth_date: datetime
    # Идентификатор животного
    animal_id: str
    # Номер лактации
    lactation: int
    # Медианная оценка удоев
    median: float


def round_calving_date(calving_date: pd.Timestamp) -> pd.Timestamp:
    """Выровнять точку временного ряда по ближайшей границе месяца"""

    num_days = calendar.monthrange(calving_date.year, calving_date.month)[1]
    clawing_day = calving_date.day
    if clawing_day > num_days / 2:
        rounded_clawing_date = (calving_date.replace(day=1) + timedelta(days=32)).replace(day=1)
    else:
        rounded_clawing_date = calving_date.replace(day=1)

    return rounded_clawing_date


def make_animal_metadata(animal_id: str, train_dataset: pd.DataFrame, test_dataset: pd.DataFrame) -> AnimalMetadata:
    """Получить временной ряд с метаданными для животного"""

    ts_dst = []
    birth_date = None
    farm = None
    farmgroup = None
    lactation = None

    df_train = train_dataset[train_dataset['animal_id'] == animal_id]
    df_test = test_dataset[test_dataset['animal_id'] == animal_id]

    for index, row in df_train.iterrows():
        birth_date = row['birth_date']
        farm = row['farm']
        farmgroup = row['farmgroup']
        lactation = row['lactation']

        # Выравнивание точки временного ряда по границе месяца
        rounded_clawing_date = round_calving_date(row['calving_date'])

        for i in range(1, 11):
            ts_dst.append((rounded_clawing_date + timedelta(days=30 * i), row[f'milk_yield_{i}']))

    for index, row in df_test.iterrows():
        birth_date = row['birth_date']
        farm = row['farm']
        farmgroup = row['farmgroup']
        lactation = row['lactation']

        rounded_clawing_date = round_calving_date(row['calving_date'])

        for i in range(1, 3):
            ts_dst.append((rounded_clawing_date + timedelta(days=30 * i), row[f'milk_yield_{i}']))

    ts = pd.DataFrame(ts_dst, columns=['ts', 'y'])
    ts.sort_values('ts', inplace=True)
    ts.set_index('ts', inplace=True)

    # Ресемплирование ряда по границе 30 дней с заполнением пропусков
    # yapf: disable
    ts_resampled = (
        ts.interpolate(method='linear', limit_direction='forward', axis=0)
          .resample('30d')
          .mean()
          .interpolate(method='linear', limit_direction='forward', axis=0)
    )
    # yapf: enable
    median = float(np.nanmedian(ts[['y']]))

    return AnimalMetadata(
        ts=ts,
        ts_resampled=ts_resampled,
        farm=farm,
        farmgroup=farmgroup,
        birth_date=birth_date,
        animal_id=animal_id,
        lactation=lactation,
        median=median,
    )


def get_avg_milk_yield(animal_id: str, animals_metadata_dict: Dict[str, AnimalMetadata]) -> Optional[float]:
    """
    Получить средний удой коровы

    @param animal_id: Идентификатор животного
    @param animals_metadata_dict: Словарь метаданных животного
    @return: Средний удой животного
    """

    if animal_id is None:
        return None

    if animal_id in animals_metadata_dict:
        return animals_metadata_dict[animal_id].ts.mean()['y']

    return None


def get_pedigree_mother(animal_id, pedigree_by_animal_id) -> Optional[str]:
    """Получить мать коровы"""

    if animal_id is None:
        return None
    try:
        mother_father_row = pedigree_by_animal_id.loc[animal_id]
    except KeyError:
        mother_father_row = None

    if mother_father_row is not None:
        return mother_father_row['mother_id']
    return None


def get_pedigree_grandmother_by_fater(animal_id: str, pedigree_by_animal_id: pd.DataFrame) -> Optional[str]:
    """Получить бабку коровы по отцу"""

    if animal_id is None:
        return None
    try:
        mother_father_row = pedigree_by_animal_id.loc[animal_id]
    except KeyError:
        mother_father_row = None

    if mother_father_row is not None:
        father_id = mother_father_row['father_id']
        return get_pedigree_mother(father_id, pedigree_by_animal_id)

    return None


def get_pedigree_milk_yield(
    animal_id: str,
    animals_metadata_dict: Dict[str, AnimalMetadata],
    pedigree_by_animal_id: pd.DataFrame,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Получить средние удои матери и двух бабок"""

    mother_id = get_pedigree_mother(animal_id, pedigree_by_animal_id)

    grandmother_by_mother_id = get_pedigree_mother(
        get_pedigree_mother(animal_id, pedigree_by_animal_id),
        pedigree_by_animal_id,
    )
    grandmother_by_father_id = get_pedigree_grandmother_by_fater(animal_id, pedigree_by_animal_id)

    return (
        get_avg_milk_yield(mother_id, animals_metadata_dict),
        get_avg_milk_yield(grandmother_by_mother_id, animals_metadata_dict),
        get_avg_milk_yield(grandmother_by_father_id, animals_metadata_dict),
    )


def prepare_x_train(cow_timeseries_by_ids, estimates_for_animals_train) -> np.ndarray:
    """
    Подготовить обучающую выборку X

    Обучающие X включают:
    - значение возраста на момент начала первой точки
    - средний удой матери и двух бабок
    - вектор идентификаторов группы ферм
    - две точки, смещаемые на единицу по ресемплированному ряду как авторегрессионный лаг

    Авторегрессионный лаг состоит из двух точек с шагом 1, т.к. строится универсальная модель, учитывающая животных,
    отсуствтующих в обучающей выборке.

    Для более точного прогнозирования для животных, присутствующих в обучающей выборке - можно обучить модели
    с большим размером авторегрессионной выборки.

    :param cow_timeseries_by_ids:
    :param estimates_for_animals_train:
    :return: Подготовленный Numpy Array для передачи на вход модели градиентного бустинга
    """

    x_len = 0
    for cow_id, animal_obj in tqdm.tqdm(cow_timeseries_by_ids.items()):
        df_len = len(animal_obj.ts_resampled)
        for i in range(0, df_len - 9, 1):
            x_len += 1

    len_columns = len(X_COLUMNS)
    x_np = np.ndarray(shape=[x_len, len_columns])

    item_idx = 0
    for cow_id, animal_obj in tqdm.tqdm(cow_timeseries_by_ids.items()):
        farmgroup = animal_obj.farmgroup
        cow_birth_date = animal_obj.birth_date

        x_arr = np.zeros(len_columns)
        x_arr[FARMGROUP_COLUMNS_BY_IDS[int(farmgroup)]] = 1

        avg_mother, avg_grandmother_by_mother, avg_grandmother_by_father = estimates_for_animals_train.get(
            cow_id,
            [np.NaN, np.NaN, np.NaN]
        )

        x_arr[IDX_AVG_MOTHER] = avg_mother
        x_arr[IDX_AVG_GRANDMOTHER_BY_MOTHER] = avg_grandmother_by_mother
        x_arr[IDX_AVG_GRANDMOTHER_BY_FATHER] = avg_grandmother_by_father

        df_len = len(animal_obj.ts_resampled)
        ts = animal_obj.ts_resampled
        for i in range(0, df_len - 9, 1):
            row1 = ts.iloc[i]
            row2 = ts.iloc[i + 1]
            x_arr[IDX_AGE] = (row1.name - cow_birth_date).days
            x_arr[IDX_X1] = row1['y']
            x_arr[IDX_X2] = row2['y']
            x_arr[IDX_TIMESTAMP] = row1.name.timestamp()
            for j in range(8):
                # print(j, IDX_Y3 + j, i + j + 2)
                x_arr[IDX_Y3 + j] = ts.iloc[i + j + 2]['y']

            x_np[item_idx] = x_arr

        item_idx += 1

    return x_np


def convert_x_np_to_df(x_np):
    x = pd.DataFrame(x_np, columns=X_COLUMNS)
    x = x.sort_values(['timestamp'] + FARMGROUP_COLUMNS)
    x = x.reset_index()
    # сделаем разбиение train-test-split таким образом, чтобы в тестовую часть не попадало будущее
    del x['index']

    y = x[['Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10']]

    for i in range(3, 11):
        del x[f'Y{i}']

    del x['timestamp']
    return x, y


def convert_x_np_to_df_test(x_np):
    x = pd.DataFrame(x_np, columns=X_COLUMNS_TEST)
    # сделаем разбиение train-test-split таким образом, чтобы в тестовую часть не попадало будущее
    del x['timestamp']
    del x['animal_id']
    del x['lactation']

    return x


def prepare_x_test(
    cow_timeseries_by_ids_test,
    estimates_for_animals_train,
    factorized_test_animals,
    factorized_test_animals_grouper_by_id
):
    x_len = len(cow_timeseries_by_ids_test.items())
    len_columns = len(X_COLUMNS_TEST)
    x_np = np.ndarray(shape=[x_len, len_columns])

    item_idx = 0
    for cow_id, animal_obj in tqdm.tqdm(cow_timeseries_by_ids_test.items()):
        farmgroup = animal_obj.farmgroup
        cow_birth_date = animal_obj.birth_date

        x_arr = np.zeros(len_columns)
        x_arr[FARMGROUP_COLUMNS_BY_IDS[int(farmgroup)]] = 1

        avg_mother, avg_grandmother_by_mother, avg_grandmother_by_father = estimates_for_animals_train.get(
            cow_id,
            [np.NaN, np.NaN, np.NaN]
        )

        x_arr[IDX_AVG_MOTHER] = avg_mother
        x_arr[IDX_AVG_GRANDMOTHER_BY_MOTHER] = avg_grandmother_by_mother
        x_arr[IDX_AVG_GRANDMOTHER_BY_FATHER] = avg_grandmother_by_father

        ts = animal_obj.ts
        row1 = ts.iloc[0]
        row2 = ts.iloc[1]
        x_arr[IDX_AGE] = (row1.name - cow_birth_date).days
        x_arr[IDX_X1] = row1['y']
        x_arr[IDX_X2] = row2['y']
        x_arr[IDX_TIMESTAMP] = row1.name.timestamp()
        x_arr[IDX_ANIMAL_ID] = factorized_test_animals.iloc[[
            factorized_test_animals_grouper_by_id.groups[animal_obj.animal_id]
        ][0]]['idx'].iloc[0]
        x_arr[IDX_LACTATION] = animal_obj.lactation

        x_np[item_idx] = x_arr

        item_idx += 1

    return x_np


def load_train_dataset() -> pd.DataFrame:
    train_dataset = pd.read_csv(os.path.join('data', 'train.csv'))
    train_dataset['calving_date'] = pd.to_datetime(train_dataset['calving_date'])
    train_dataset['birth_date'] = pd.to_datetime(train_dataset['birth_date'])
    return train_dataset


def load_test_dataset(dataset_path: str) -> pd.DataFrame:
    test_dataset = pd.read_csv(dataset_path)
    test_dataset['calving_date'] = pd.to_datetime(test_dataset['calving_date'])
    test_dataset['birth_date'] = pd.to_datetime(test_dataset['birth_date'])
    return test_dataset


def load_pedigree() -> pd.DataFrame:
    return pd.read_csv(os.path.join('data', 'pedigree.csv'))


def fit() -> List[LGBMRegressor]:
    """
    Обучить модель градиентного бустинга

    Для обучения готовится авторегрессионный датасет с дополнительными столбцами метаданных

    @return: Обученная модель прогнозирования
    """

    train_dataset = load_train_dataset()
    test_dataset = load_test_dataset(os.path.join('data', PUBLIC_TEST_DATASET_NAME))
    pedigree = load_pedigree()

    pedigree_by_animal_id = pedigree.set_index('animal_id')

    animals_in_train = set(train_dataset['animal_id'])
    animals_child = set(pedigree['animal_id'])

    print('Построение метаданных и временных рядов для животных из обучающей выборки')
    animals_metadata_train = {
        animal_id: make_animal_metadata(animal_id, train_dataset, test_dataset)
        for animal_id in tqdm.tqdm(list(sorted(animals_in_train)))
    }

    # Оценки удоев предков
    estimates_for_animals_train = {
        animal_id: get_pedigree_milk_yield(animal_id, animals_metadata_train, pedigree_by_animal_id)
        for animal_id in list(sorted(animals_in_train & animals_child))
    }

    regressor = [
        LGBMRegressor(
            learning_rate=0.3,
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=12,
            device_type='cpu',  # or CPU
            gpu_platform_id=0,
            gpu_device_id=0,
            max_bin=63,  # Максимальное количество бинов, в которые будут группироваться значения признаков
            # - см.  https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html
            gpu_use_dp=False,  # По возможности старайтесь использовать обучение с одинарной точностью
            # ( ), потому что большинство графических процессоров
            # (особенно потребительские графические процессоры NVIDIA) имеют низкую
            # производительность при двойной точности.
            # num_leaves=255,
            tree_learner='serial',
            boosting='gbdt',  # gbdt быстрее существенно чем dart на gpu
        ) for _ in range(8)
    ]

    x_np = prepare_x_train(animals_metadata_train, estimates_for_animals_train)
    x, y = convert_x_np_to_df(x_np)

    for i in range(8):
        regressor[i].fit(x.values, y[[f'Y{i+3}']].values)

    return regressor


def predict(model: List[LGBMRegressor], test_dataset_path: str) -> pd.DataFrame:
    """
    Построить прогноз с помощью модели градиентного бустинга

    @param model: Обученная ранее модель градиентного бустинга
    @param test_dataset_path: Путь к тестовому датасету
    @return: Датафрейм с построенным прогнозом, заданного формата
    """

    regressor = model

    # Подготовить данные для прогнозирования
    train_dataset = load_train_dataset()
    test_dataset = load_test_dataset(test_dataset_path)
    pedigree = load_pedigree()
    pedigree_by_animal_id = pedigree.set_index('animal_id')

    animals_child = set(pedigree['animal_id'])
    animals_in_test = set(test_dataset['animal_id'])

    print('Построение метаданных и временных рядов для животных из тестовой выборки')
    animals_metadata_test = {
        animal_id: make_animal_metadata(animal_id, train_dataset, test_dataset)
        for animal_id in tqdm.tqdm(list(sorted(animals_in_test)), total=len(animals_in_test))
    }

    factorized_test_animals = pd.DataFrame(test_dataset[['animal_id']])
    factorized_test_animals['idx'] = pd.factorize(factorized_test_animals['animal_id'])[0]

    factorized_test_animals_grouper_by_id = factorized_test_animals.groupby('animal_id')
    factorized_test_animals_grouper_by_idx = factorized_test_animals.groupby('idx')

    estimates_for_animals_test = {
        animal_id: get_pedigree_milk_yield(animal_id, animals_metadata_test, pedigree_by_animal_id)
        for animal_id in list(sorted(animals_in_test & animals_child))
    }

    x_test_np = prepare_x_test(
        animals_metadata_test,
        estimates_for_animals_test,
        factorized_test_animals,
        factorized_test_animals_grouper_by_id
    )
    x_test = convert_x_np_to_df_test(x_test_np)

    # Выполнить прогнозирование
    predictions_np = [r.predict(x_test.values) for r in regressor]

    # Преобразовать выдачу модели в читаемый прогноз
    x_test_pd = pd.DataFrame(x_test_np, columns=X_COLUMNS_TEST)

    submission = pd.DataFrame({
        'animal_id': [
            factorized_test_animals.iloc[factorized_test_animals_grouper_by_idx.groups[int(x)][0]]['animal_id']
            for x in x_test_pd['animal_id']
        ],
        'lactation': x_test_pd['lactation'],
        'milk_yield_3': predictions_np[0],
        'milk_yield_4': predictions_np[1],
        'milk_yield_5': predictions_np[2],
        'milk_yield_6': predictions_np[3],
        'milk_yield_7': predictions_np[4],
        'milk_yield_8': predictions_np[5],
        'milk_yield_9': predictions_np[6],
        'milk_yield_10': predictions_np[7],
    })

    return submission


def get_dummied_farmgroup_column_names():
    train_test_dataset_keys = ['animal_id', 'lactation', 'farm', 'farmgroup']
    train_dataset = pd.read_csv(os.path.join('data', 'train.csv'))
    test_dataset = pd.read_csv(os.path.join('data', PUBLIC_TEST_DATASET_NAME))

    train_test_dataset = train_dataset.merge(
        test_dataset, left_on=train_test_dataset_keys, right_on=train_test_dataset_keys, how='outer'
    )
    dummied_train_test_dataset = pd.get_dummies(train_test_dataset[['farmgroup']], columns=['farmgroup'])

    return dummied_train_test_dataset.columns


if __name__ == '__main__':
    FARMGROUP_COLUMNS = list(sorted(get_dummied_farmgroup_column_names()))

    FARMGROUP_COLUMNS_BY_IDS = {int(x.replace('farmgroup_', '')): idx for idx, x in enumerate(FARMGROUP_COLUMNS)}

    X_COLUMNS_COMMON = FARMGROUP_COLUMNS + [
        'age',
        'avg_mother',
        'avg_grandmother_by_mother',
        'avg_grandmother_by_father',
        'X1',
        'X2',
        'timestamp',
    ]
    X_COLUMNS = X_COLUMNS_COMMON + ['Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10']
    X_COLUMNS_TEST = X_COLUMNS_COMMON + ['animal_id', 'lactation']

    IDX_AGE = len(FARMGROUP_COLUMNS)
    IDX_AVG_MOTHER = len(FARMGROUP_COLUMNS) + 1
    IDX_AVG_GRANDMOTHER_BY_MOTHER = len(FARMGROUP_COLUMNS) + 2
    IDX_AVG_GRANDMOTHER_BY_FATHER = len(FARMGROUP_COLUMNS) + 3
    IDX_X1 = len(FARMGROUP_COLUMNS) + 4
    IDX_X2 = len(FARMGROUP_COLUMNS) + 5
    IDX_TIMESTAMP = len(FARMGROUP_COLUMNS) + 6

    IDX_Y3 = len(FARMGROUP_COLUMNS) + 7
    IDX_Y4 = len(FARMGROUP_COLUMNS) + 8
    IDX_Y5 = len(FARMGROUP_COLUMNS) + 9
    IDX_Y6 = len(FARMGROUP_COLUMNS) + 10
    IDX_Y7 = len(FARMGROUP_COLUMNS) + 11
    IDX_Y8 = len(FARMGROUP_COLUMNS) + 12
    IDX_Y9 = len(FARMGROUP_COLUMNS) + 13
    IDX_Y10 = len(FARMGROUP_COLUMNS) + 14

    IDX_ANIMAL_ID = len(FARMGROUP_COLUMNS) + 7
    IDX_LACTATION = len(FARMGROUP_COLUMNS) + 8

    _model = fit()
    _submission = predict(_model, os.path.join('data', 'X_test_public.csv'))
    _submission.to_csv(os.path.join('data', 'submission_gbm.csv'), sep=',', index=False)

    _submission_private = predict(_model, os.path.join('..', 'external', 'private', 'X_test_private.csv'))
    _submission_private.to_csv(os.path.join('data', 'submission_private_gbm.csv'), sep=',', index=False)
