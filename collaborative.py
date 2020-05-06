import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics.pairwise import pairwise_distances as pw_dist
from sklearn.metrics import mean_squared_error
import random
from sklearn.model_selection import train_test_split

#загрузка данных
ratings_tb = pd.read_csv('rat.csv')
movies_tb = pd.read_csv('movies.csv')

n_users = ratings_tb['userId'].unique().shape[0]
n_items = ratings_tb['movieId'].unique().shape[0]

# чтобы можно было удобно работать дальше, необходимо отмасштабировать
# значения в колонке movieId (новые значения будут в диапазоне от 1 до
# количества фильмов)
input = ratings_tb['movieId'].unique()

def scale_movie_id(input_id):
    return np.where(input == input_id)[0][0] + 1

ratings_tb['movieId'] = ratings_tb['movieId'].apply(scale_movie_id)

# делим данные на тренировочный и тестовый наборы
train_matrix, test_matrix = train_test_split(ratings_tb, test_size=0.80)

# создаём две user-item матрицы – для обучения и для теста
train_matrix_matrix = np.zeros((n_users, n_items))
for line in train_matrix.itertuples():
    train_matrix_matrix[line[1] - 1, line[2] - 1] = line[3]

test_matrix_matrix = np.zeros((n_users, n_items))
for line in test_matrix.itertuples():
    test_matrix_matrix[line[1] - 1, line[2] - 1] = line[3]

#находим косинусное расстояние
def get_similarities(rating_data_mx, s_type='user'):
    if s_type=='user':
        similarity_mx = 1 - pw_dist(rating_data_mx, metric='cosine')

    return similarity_mx

#находим приближенный рейтинг для пользователя

def predict(rating_mx, similarity_mx, s_type='user'):
    if s_type=='user':
        mean_user_rating_ar = rating_mx.sum(axis=1)/np.count_nonzero(rating_mx, axis=1)
        delta_ratings_mx = (rating_mx - mean_user_rating_ar[:, np.newaxis])
        delta_ratings_mx[rating_mx==0.0] = 0.0
        pred_ar = mean_user_rating_ar[:, np.newaxis] + similarity_mx.dot(delta_ratings_mx) / \
                  np.array([np.abs(similarity_mx).sum(axis=1)]).T
    
    return pred_ar


#СКО(Среднеквадратическое отклонение)
def RMSE(pred_ar, truth_ar, matrix=True):
    if matrix:
        pred_ar = pred_ar[truth_ar.nonzero()].flatten()
        truth_ar = truth_ar[truth_ar.nonzero()].flatten()
    
    return sqrt(mean_squared_error(pred_ar, truth_ar))


usr_sim_mx = get_similarities(train_matrix_matrix, s_type='user')
user_pred_mx = predict(test_matrix_matrix, usr_sim_mx, s_type='user')

print(user_pred_mx)

#print ('User-based RMSE: ' + str(RMSE(user_pred_mx, test_matrix_matrix)))

