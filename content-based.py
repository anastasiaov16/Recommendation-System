import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import collections
from sklearn.feature_extraction.text import CountVectorizer
import time
from openpyxl import load_workbook
from sklearn.metrics import mean_squared_error

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
    k=[]
    if len(key)==3:
        for j in range(0, len(movies_tb.index)):
            if((movies_tb['genres'].iloc[j]).lower() == key[0] or (movies_tb['genres'].iloc[j]).lower() == key[1] or (movies_tb['genres'].iloc[j]).lower() == key[2]):
                k.append(movies_tb['movieId'].iloc[j])
    if len(key)==2:
        for j in range(0, len(movies_tb.index)):
            if((movies_tb['genres'].iloc[j]).lower() == key[0] or (movies_tb['genres'].iloc[j]).lower() == key[1]):
                k.append(movies_tb['movieId'].iloc[j])
    if len(key)==1:
        for j in range(0, len(movies_tb.index)):
            if((movies_tb['genres'].iloc[j]).lower() == key[0]):
                k.append(movies_tb['movieId'].iloc[j])
    #h = k.split(',')
    return(k)


ratings_tb = pd.read_csv('rat.csv')
movies_tb = pd.read_csv('movies.csv')

#удалили лишние столбцы 
ratings_tb.drop(ratings_tb.columns[[3]], axis=1, inplace=True)

#выбираем только те фильмы у пользователя, которым он поставил наиб оценку
fd = ratings_tb[(ratings_tb.rating > 4.0)]
train_matrix, test_matrix = train_test_split(fd, test_size=0.80)
#выбираем нужного пользователя
inp = int(input('Введите ID пользователя: '))

#список жанров
#genres = []
#for i in movies_tb['genres']:
#    print(i)
#    words = i.replace('|', ' ').split()
#    for word in words:
#        if word is not genres:
#            genres.append(word)
#auxiliaryList = list(set(genres))

#жанр-фильмы
#filmes={}
#for i in auxiliaryList:
#    ids=[]
#    for j in range(0, len(movies_tb.index)):
#        ar = (movies_tb['genres'].iloc[j]).replace('|', ' ').split()
#        for k in ar:
#            if(k == i):
#                ids.append(movies_tb['movieId'].iloc[j])
#    filmes[i.lower()]=ids

#df = {}



#for i in train_matrix['userId']:
#    idlist = new_list_movies_id(train_matrix, i)
#    mostgen = most_genres(idlist,movies_tb)
#    mov=[]
#    for j in mostgen:
#        if(j in filmes):
#            movies = filmes[j]
#            random.shuffle(movies)
#            arr = movies[:10]
#            for k in arr:
#                mov.append(k)
#    print(mov)
#    df[i]=mov
#print(df)


#pd.DataFrame.from_dict(data=df, orient='index').to_csv('content.csv', header=False)

#inp = int(input('Введите ID пользователя: '))
#вызываем функцию, которая выбирает все фильмы для пользователя
idlist = new_list_movies_id(fd, inp)

#выбираем наиболее часто повторяющиеся жанры
mostgen = most_genres(idlist,movies_tb)

#находим и рекомендуемые фильмы, на основе выбранных жанров
rm = result_mov(movies_tb, mostgen)
random.shuffle(rm)
print('Возможно, это вас заинтересует: ',rm[:10])

