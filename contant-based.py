import numpy as np
import pandas as pd
import random
from sklearn import model_selection as cv
from sklearn.model_selection import KFold
import collections
from sklearn.feature_extraction.text import CountVectorizer
import time

#составляем для пользователя список фильмов, которые он оценил. переходим в таблицу
#с фильмами и находим эти фильмы. заносим в строку жанры фильмов.
#считаем кол-во самых часто встр жанров и берем максимальный.
#проходим по списку фильмов и выбираем нужный жанр

vectorizer = CountVectorizer()

#для пользователя заносим в отдельный список фильмы, которые он оценил
def new_list_movies_id(fd, inp):
    list_movies=[]
    for i in fd:
        for j in range(0, len(fd.index)):
            if((fd['userId'].iloc[j] == inp) and (fd['movieId'].iloc[j] not in list_movies)):
                list_movies.append(fd['movieId'].iloc[j])
    return(list_movies)

#нашли фильмы и занесли в строку. нашли жанры, которые часто повторяются
def most_genres(idlist,movies_tb):
    mov=''
    for i in idlist:
        for j in range(0, len(movies_tb.index)):
            if(i == movies_tb['movieId'].iloc[j]):
                mov = mov + (movies_tb['genres'].iloc[j])+'|'
    t = mov.replace('|', ' ').split()
    #mgen = collections.Counter(t).most_common(3)
    mgen = vectorizer.fit_transform(t)
    df = pd.DataFrame({'genres': vectorizer.get_feature_names(), 'occurrences':np.asarray(mgen.sum(axis=0)).ravel().tolist()})
    df1=df.sort_values(by='occurrences', ascending=False)
    first_top = df1.head(3)
    res = list(first_top.genres)
    return(res)

#проходим по списку фильмов и выбираем нужный жанр
def result_mov(movies_tb, key):
    k=''
    for j in range(0, len(movies_tb.index)):
        if((movies_tb['genres'].iloc[j]).lower() == key[0] or (movies_tb['genres'].iloc[j]).lower() == key[1] or (movies_tb['genres'].iloc[j]).lower() == key[2]):
            k = k + movies_tb['title'].iloc[j] + ','
    h = k.split(',')
    return(h)


ratings_tb = pd.read_csv('rat.csv')
movies_tb = pd.read_csv('movies.csv')

#удалили лишние столбцы 
ratings_tb.drop(ratings_tb.columns[[3]], axis=1, inplace=True)

#выбираем только те фильмы у пользователя, которым он поставил наиб оценку
fd = ratings_tb[(ratings_tb.rating > 4.0)]
#train_matrix, test_matrix = cv.train_test_split(fd, test_size=0.80)
#выбираем нужного пользователя
inp = int(input('Введите ID пользователя: '))
#for i in train_matrix['userId']:
#    idlist = new_list_movies_id(train_matrix, i)
#    mostgen = most_genres(idlist,movies_tb)
#    rm = result_mov(movies_tb, mostgen)
#    random.shuffle(rm)
 #   print('Возможно, это вас заинтересует: ',rm[:10])

#вызываем функцию, которая выбирает все фильмы для пользователя
idlist = new_list_movies_id(fd, inp)

#выбираем наиболее часто повторяющиеся жанры
mostgen = most_genres(idlist,movies_tb)

#находим и рекомендуемые фильмы, на основе выбранных жанров
rm = result_mov(movies_tb, mostgen)
random.shuffle(rm)
print('Возможно, это вас заинтересует: ',rm[:10])

