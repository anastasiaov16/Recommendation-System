import numpy as np
import pandas as pd
import random
from sklearn import model_selection as cv
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from apyori import apriori
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#-------------------------------все функции--------------------------
#перенумерация списка
def scale_movie_id(input_id):
    return np.where(input == input_id)[0][0] + 1

#преобразовываем оценки, поставленные пользователям в 0 и 1 в соответствии с нуждами
#если нужны только положительные оценки, то передаем в category от 3,5; если нужно просто оцененные фильмы, 
#то передаем в category 1
def transform_array(array, category):
    for i in range(len(array)):
        for j in range(len(array[i])):
            if array[i][j] >= category:
                array[i][j] = 1
            else:
                array[i][j] = 0
    return array


#преобразуем в список транзакций
def transaction_list(df):
    list_external=[]
    for i in range(len(df)):
        list_internal=[]
        for j in range(len(df[i])):
            if df[i][j] == 1:
                list_internal.append(j)
        list_external.append(list_internal)
    return list_external


#вывод рекомендации
def print_result(association_results):
    for item in association_results:
        pair = item.items 
        items = [x for x in pair]
        print("If you watched it: " + movies_tb['title'].iloc[items[0]] + ', you might like this: ' + movies_tb['title'].iloc[items[1]])
        print("Support: " + str(item.support))
        print("Confidence: " + str(item.ordered_statistics[0].confidence))
        print("Lift: " + str(item.ordered_statistics[0].lift))
        print("=====================================")

#СКО(Среднеквадратическое отклонение)
def RMSE(pred_ar, truth_ar, matrix=True):
    if matrix:
        pred_ar = pred_ar[truth_ar.nonzero()].flatten()
        truth_ar = truth_ar[truth_ar.nonzero()].flatten()
    
    return sqrt(mean_squared_error(pred_ar, truth_ar))

#--------------------------------------------------------------------------

ratings_tb = pd.read_csv('rat.csv')
movies_tb = pd.read_csv('movies.csv')

input = ratings_tb['movieId'].unique()

number_of_users = ratings_tb['userId'].unique().shape[0]
number_of_items = ratings_tb['movieId'].unique().shape[0]

ratings_tb['movieId'] = ratings_tb['movieId'].apply(scale_movie_id)


# делим данные на тренировочный и тестовый наборы
train_matrix, test_matrix = cv.train_test_split(ratings_tb, test_size=0.85)

# создаём две user-item матрицы – для обучения и для теста
train_matrix_matrix = np.zeros((number_of_users, number_of_items))
for line in train_matrix.itertuples():
    train_matrix_matrix[line[1] - 1, line[2] - 1] = line[3]

test_matrix_matrix = np.zeros((number_of_users, number_of_items))
for line in test_matrix.itertuples():
    test_matrix_matrix[line[1] - 1, line[2] - 1] = line[3]


#преобразуем оценки для пользователя в 0 и 1
transformed_user_item_train_matrix = transform_array(train_matrix_matrix, 3.5)

#преобразуем в список транзакций
transactions=transaction_list(transformed_user_item_train_matrix)

#запускаем алгоритм apriori
association_rules = apriori(transactions, min_support=0.0045, min_confidence=0.5, min_lift=1.2, min_length=2)
association_results = list(association_rules)

#вывод результата
#print_result(association_results)