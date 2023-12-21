# Описание задачи. 

В процессах принятий решений на молочных фермах одним из важных показателей является прогноз продуктивности животных,
который необходим для отбора животных для последующего осеменения или выбраковки. 

Если модель сможет точно предсказывать продуктивность животных (удой молока), это поможет увеличить производительность
фермы и уменьшить затраты на содержание животных.

В качестве измерения удоя животных часто выступают `контрольные дойки` -- замер суточного надоя каждого животного раз в
месяц в течение всей лактации. 

TODO: что такое лактация, сколько их

Задача состоит в том, чтобы на основе данных 
- о контрольных дойках коров;
- и родословных 
  
Создать модель, которая сможет предсказывать продуктивность каждого животного в последующих месяцах.

## Обучающая выборка.

- Данные о первых 10 контрольных дойках для каждого животного, 
- родословный этих животных, 
- данные о лактациях 
  - номер лактации,
  - дата начала лактации, 
  - ферма, 
  - хозяйство, 
  - дата рождения животного.


## Тестовая выборка.

- Пул животных с первыми двумя контрольными дойками за лактацию. 
 
Необходимо дать прогноз на следующие 8 контрольных доек. 

В тестовую выборку попадают как животные, которые представлены в обучающей выборке, но за последующие лактации, 
так и животные, которые вовсе не представлены в обучающей выборке.

TODO: имеются ли данные о родословных в пуле животных, которые не включены в обучающую выборку?

## Метрика качества.

Метрикой качества является RMSE между данными к контрольным дойкам с 3-ей по 10-ю (для тех контрольных доек, где данные
известны) и их прогнозом.

# Описание данных.

## Родословная - predige.csv

| Колонка	  | Описание      |
| ---------- | ------------- |
| animal_id  | ID животного  |
| mother_id  | ID матери     |
| father_id  | ID отца       |


## Контрольные дойки

| Колонка	       | Описание                                                             | Таргет |
| ---------------- | -------------------------------------------------------------------- | ------ | 
| animal_id        | ID животного                                                         |        |
| lactation        | Номер лактации                                                       |        |
| calving_date     | Дата начала лактации                                                 |        |
| farm             | Ферма                                                                |        |
| farmgroup	       | Хозяйство (группа из ферм)                                           |        |
| birth_date       | Дата рождения животного                                              |        |
| milk_yield_1     | Контрольная дойка (удой за 1 день) в 1 месяце после начала лактации  |        |
| milk_yield_2     | Контрольная дойка (удой за 1 день) в 2 месяце после начала лактации  |        |
| milk_yield_3     | Контрольная дойка (удой за 1 день) в 3 месяце после начала лактации  | X      |
| milk_yield_4     | Контрольная дойка (удой за 1 день) в 4 месяце после начала лактации  | X      |
| milk_yield_5     | Контрольная дойка (удой за 1 день) в 5 месяце после начала лактации  | X      |
| milk_yield_6     | Контрольная дойка (удой за 1 день) в 6 месяце после начала лактации  | X      |
| milk_yield_7     | Контрольная дойка (удой за 1 день) в 7 месяце после начала лактации  | X      |
| milk_yield_8     | Контрольная дойка (удой за 1 день) в 8 месяце после начала лактации  | X      |
| milk_yield_9     | Контрольная дойка (удой за 1 день) в 9 месяце после начала лактации  | X      |
| milk_yield_10    | Контрольная дойка (удой за 1 день) в 10 месяце после начала лактации | X      |