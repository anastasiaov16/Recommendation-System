import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics.pairwise import pairwise_distances as pw_dist
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import collections
import time

ratings_tb = pd.read_csv('rat.csv')
movies_tb = pd.read_csv('movies.csv')


# чтобы можно было удобно работать дальше, необходимо отмасштабировать
# значения в колонке movieId (новые значения будут в диапазоне от 1 до
# количества фильмов)
input = ratings_tb['movieId'].unique()

def scale_movie_id(input_id):
    return np.where(input == input_id)[0][0] + 1


def get_similarities(rating_data_mx, s_type='user'):
    if s_type=='user':
        similarity_mx = 1 - pw_dist(rating_data_mx, metric='cosine')

    return similarity_mx

def predict(rating_mx, similarity_mx, s_type='user'):
    if s_type=='user':
        mean_user_rating_ar = rating_mx.sum(axis=1)/np.count_nonzero(rating_mx, axis=1)
        delta_ratings_mx = (rating_mx - mean_user_rating_ar[:, np.newaxis])
        delta_ratings_mx[rating_mx==0.0] = 0.0
        pred_ar = mean_user_rating_ar[:, np.newaxis] + similarity_mx.dot(delta_ratings_mx) / \
                  np.array([np.abs(similarity_mx).sum(axis=1)]).T
    
    return pred_ar


def new_list_movies_id(ratings_tb, inp):
    list_movies=[]
    for i in ratings_tb:
        for j in range(0, len(ratings_tb.index)):
            if((ratings_tb['userId'].iloc[j] == inp) and (ratings_tb['movieId'].iloc[j] not in list_movies)):
                list_movies.append(ratings_tb['movieId'].iloc[j])
    return(list_movies)

def most_genres(idlist,movies_tb):
    mov=''
    for i in idlist:
        for j in range(0, len(movies_tb.index)):
            if(i == movies_tb['movieId'].iloc[j]):
                mov = mov + (movies_tb['genres'].iloc[j])+'|'
    t = mov.replace('|', ' ').split()
    mgen = collections.Counter(t).most_common(3)
    return(mgen)



n_users = ratings_tb['userId'].unique().shape[0]
n_items = ratings_tb['movieId'].unique().shape[0]


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


usr_sim_mx = get_similarities(train_matrix_matrix, s_type='user')
user_pred_mx = predict(test_matrix_matrix, usr_sim_mx, s_type='user')


a=0
b=0
lst = {}
films =[]

for j in user_pred_mx:
    b=0
    mt = 0
    films=[]
    for i in j:
        if float(i)>3.7:
            if a not in lst:
                mt = 1
                films.append(b)

        b+=1
    if mt == 1:
        lst[a]= films
    a+=1
ks = lst.keys()

vectorizer = CountVectorizer()


for t in ks:
    if(t == 5):
        print('Пользователю с id - '+str(t))
        print('Рекомендуемые фильмы: ')
        idlist = new_list_movies_id(ratings_tb, t)
        mostgen = most_genres(idlist,movies_tb)
        k = dict(mostgen)
        key=list(k.keys())
        rek = ratings_tb[((ratings_tb['userId']) == t) & ((ratings_tb['rating']) > 3.7)]['movieId'].values
        ls = ''
        for p in range(0,len(rek)):
            a = movies_tb[movies_tb['movieId'] == rek[p]]
            for j in range(0, len(a.genres)):
                if(a['genres'].iloc[j] == key[0] or a['genres'].iloc[j] == key[1] or a['genres'].iloc[j] == key[2]):
                    ls = ls + a['title'].iloc[j] + ','
        h = ls.split(',')
        print(h)
    else:
        print('***')



