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
from openpyxl import load_workbook

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


#создаем список кандидтов размерностью 1
def createCandidate1(transaction):
	cand1 = []
	for i in transaction:
		for j in i:
			if not [j] in cand1:
				cand1.append([j])
	cand1.sort()
	res = []
	for i in cand1:
		res.append(frozenset(i))
	return res

#возвращаем кандидатов с заданной минимальной поддержкой
#создаем словарь, в который будем класть кандидата и то, сколько раз он встретился в транзакциях
#проходимся по словарю и считаем поддержку для каждого элемента. выводим те, что повторяются чаще, чем минимальная поддержка  
def returnAllCandidateWithMinSupport(data, candidates, minSup):
	candidateRepRate = {}
	count = len(data)
	result	= []
	support_data = {}
	for i in data:
		for j in candidates:
			if j.issubset(i):
				candidateRepRate.setdefault(j,	0)
				candidateRepRate[j] += 1
	for item in candidateRepRate:
		support	= candidateRepRate[item] / count
		if support >= minSup:
			result.append(item)
		support_data[item] = support
	return result,	support_data


#проходимся по списку кандидатов и создаем пары новых кадидатов
def genNewKCandidate(data, n):
	result	= []
	for i in range(len(data)):
		for j in range(i + 1, len(data)):
			newCand1 = list(data[i])[:n - 2]
			newCand2 = list(data[j])[:n - 2]
			if newCand1.sort() == newCand2.sort():
				result.append(data[i] | data[j])
	return result

#запускаем алгоритм априори
#создаем единичных кандидатов. вызвращаем тех, кто повторяется чаще, чем минимальная поддержка
#проходим по получившемуся массиву и создаем возвращаем новых кандидатов к-ой размерности. опять вызвращаем тех, 
#кто повторяется чаще, чем минимальная поддержка
def apriori(dataset, minSup):
	cand1 = createCandidate1(dataset)
	data = []
	for i in dataset:
		data.append(set(i))
	L1,	support_data = returnAllCandidateWithMinSupport(data, cand1, 0.05)
	result = [L1]
	k = 2
	while (len(result[k	- 2]) > 0):
		Ck	= genNewKCandidate(result[k - 2], k)
		Lk,	supportK = returnAllCandidateWithMinSupport(data, Ck, 0.003)
		support_data.update(supportK)
		result.append(Lk)
		k+=1
	return result, support_data

#находим правила
#проходим по получившимся к-кандидатам. разбиваем их на массивы и находим правила для элементов
#
#
def genResRules(L, support_data, minConf=0.1):
	rules = []
	for i in range(1, len(L)):
		for item in L[i]:
			splitKCand=[]
			for j in item:
				splitKCand.append(frozenset([j]))
			#print ("j", j, 'H1', H1)
			if (len(splitKCand) > 2):
				searchRulesForSplitKCand(item, splitKCand, support_data,	rules,	minConf)
			else:
				searchConf(item, splitKCand, support_data, rules,	minConf)
				print()
	return rules

#находим мощность прогноза
#проходимся по элементам и делим поддержку всех случаев для данных элементов на поддержку данного элемента
#запоминаем только те, чья поддержка больше минимальной
def searchConf(item, splitKCand, support_data, rules, minConf):
	result = []
	for i in splitKCand:
		confidence = support_data[item] / support_data[item - i]
		if confidence >= minConf:
			print ('Rules: ',item - i, ' --------> ', i, 'confidence:', confidence)
			rules.append((item - i, i, confidence))
			result.append(i)
	return result

#если в к-кандидатах больше 2 эл-ов, то разбиваем и ищем правила
def searchRulesForSplitKCand(item,	splitKCand,	support_data, rules, minConf):
	if (len(item) > (len(splitKCand[0]) + 1)):
		tmp = genNewKCandidate(splitKCand, len(splitKCand[0]) + 1)
		tmp = searchConf(item,	tmp, support_data,	rules, minConf)
		if len(tmp) > 1:
			searchRulesForSplitKCand(item, tmp, support_data, rules, minConf)

#--------------------------------------------------------------------------

ratings_tb = pd.read_csv('rat.csv')
movies_tb = pd.read_csv('movies.csv')

input = ratings_tb['movieId'].unique()

number_of_users = ratings_tb['userId'].unique().shape[0]
number_of_items = ratings_tb['movieId'].unique().shape[0]

ratings_tb['movieId'] = ratings_tb['movieId'].apply(scale_movie_id)


# делим данные на тренировочный и тестовый наборы
train_matrix, test_matrix = cv.train_test_split(ratings_tb, test_size=0.50)

# создаём две user-item матрицы – для обучения и для теста
train_matrix_matrix = np.zeros((number_of_users, number_of_items))
for line in train_matrix.itertuples():
    train_matrix_matrix[line[1] - 1, line[2] - 1] = line[3]

test_matrix_matrix = np.zeros((number_of_users, number_of_items))
for line in test_matrix.itertuples():
    test_matrix_matrix[line[1] - 1, line[2] - 1] = line[3]


#преобразуем оценки для пользователя в 0 и 1
#transformed_user_item_train_matrix = transform_array(train_matrix_matrix, 3.5)

#преобразуем в список транзакций
#transactions=transaction_list(transformed_user_item_train_matrix)

#L, support_data = apriori(transactions, 0.05)
#riles = genResRules(L, support_data)

#df ={}
#for i in riles:
#	first = list(i[0])
#	second = list(i[1])
#	df[first[0]] = second
#pd.DataFrame.from_dict(data=df, orient='index').to_csv('test.csv', header=False)

def new_list_movies_id(fd, inp):
    list_movies=[]
    for i in fd:
        for j in range(0, len(fd.index)):
            if((fd['userId'].iloc[j] == inp) and (fd['movieId'].iloc[j] not in list_movies)):
                list_movies.append(fd['movieId'].iloc[j])
    return(list_movies)

df = {}
wb = load_workbook('test.xlsx')
first_sheet = wb.worksheets[0]
elem = []
for row in first_sheet.rows:
    elem.append(row[0].value.split(','))

usLen = 0
for item in elem:
	result = []
	for item1 in item:
		if item1 != '':
			result.append(item1)
	for user in range(number_of_users):
		idlist = new_list_movies_id(train_matrix, user)
		movies = []
		for mov in idlist:
			if mov == result[0]:
				for i in range(1,len(result)):
					movies.append(result[i])
		usLen += 1
		df[user]=movies
		print(usLen)
print(df)
pd.DataFrame.from_dict(data=df, orient='index').to_csv('AR.csv', header=False)

		

#запускаем алгоритм apriori
#association_rules = apriori(transactions, min_support=0.0045, min_confidence=0.5, min_lift=1.2, min_length=2)
#association_results = list(association_rules)

#вывод результата
#print_result(association_results)