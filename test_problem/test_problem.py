
# coding: utf-8

# **Задача**: разработка модели предсказания оценки по тексту отзыва.
#     
# **Данные**: отзывы о товарах с сайта mvideo.ru, оценки, которые поставлены авторами отзыва, категория и брэнд товара. 
# 
# **Цель**: Предсказание общей оценки товара
# 
# **Формат предоставления решения**: Ссылка на репозиторий на GitHub с исходным кодом программы, решающей поставленную задачу. На вход программа должна принимать данные в исходном виде (feedback), разбивать на train и test, обучать на train модель и демонстрировать ее качество на test. Репозиторий должен содержать проект для среды разработки и инструкцию по сборке и запуску. Рекомендуем использовать Jupyter Notebook, но окончательный выбор инструментария остаётся за вами.

# **Предложенное решение**
# 
# Предлагаемая "программа" предсказывает числовое значение рейтинга с помощью линейной регрессии, использующей как признаки TF-IDF счётчики слов и биграмм, содержащихся в запросе; полученное значение нормируется монотонной функцией от 1 до 5.
# 
# Модель использует исключительно текст "общих" комментариев, хотя в датасете присутствуют и иные признаки (в т.ч. положительные и отрицательные отзывы). Не зная о планируемом применении модели, мы сделали её максимально простой, и потому переносимой. 

# # Загрузка данных

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import os

import pymorphy2
import nltk


# In[2]:

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score


# Подгружаем из интернетов пунктуацию

# In[3]:

nltk.download('punkt')


# Загружаем файл с данными

# In[4]:

data = pd.read_csv('feedback.csv/X_train.csv')
data.shape


# In[5]:

data.dtypes


# In[6]:

data.reting.describe()


# Почти все оценки - пятёрки. Для MVIDEO это, конечно, хорошо, но для системы рейтингования, наверное, не очень. 
# 
# Посколькую рейтингов разных много, и они упорядочены, проще всего предсказывать их регрессией. 

# In[7]:

data.reting.value_counts().sort_index()


# Разбираемся с временным периодом

# In[8]:

data.date = pd.to_datetime(data.date)


# In[9]:

data.date.hist(bins=30);


# In[10]:

print((data.date<pd.Timestamp('2016-01-01')).mean())


# В качестве теста можно взять 2016-2017 годы, это будет честно.

# In[11]:

data.sort_values('date', inplace=True)


# Из года в год средняя оценка колебалась, но не то чтобы значительно

# In[12]:

data.reting.groupby(data.date.dt.round('180d')).mean().plot();


# Разделяем выборку на тестовую и тренировочную

# In[13]:

train_index = data[data.date<pd.Timestamp('2016-01-01')].index
test_index = data[data.date>=pd.Timestamp('2016-01-01')].index
print('train:{}, test:{} obs'.format(train_index.shape[0], test_index.shape[0]))


# # Предобработка

# Заменяем смайлики и прочие средства выразительности на служебные слова

# In[14]:

import re

reg_smiles = {
    re.compile(r"[\)]{2,}"): ' _BRACKETS_POSITIVE ',
    re.compile(r"[\(]{2,}"): ' _BRACKETS_NEGATIVE ',
    re.compile(';-?\)'): ' _SMILE_GRIN ',
    re.compile(r":[-]?\)"): ' _SMILE_POSITIVE ',
    re.compile(r":[-]?\("): ' _SMILE_NEGATIVE ',
    re.compile(r"!!!"): ' _MANY_EXCLAMATIONS ',
    re.compile(r"[.]{3,}"): ' _PERIOD ',
}

def replace_smiles(text):
    for reg, repl in reg_smiles.items():
        text = re.sub(reg, repl, text)
    return text


# Разбиваем тексты на слова

# In[15]:

def tokenize(text):
    return [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
print(tokenize(data.comment[0]))


# Приводим слова к нормальной форме

# In[16]:

morph = pymorphy2.MorphAnalyzer()


# In[ ]:

def lemmatize(words):
    return [morph.parse(word)[0].normal_form for word in words]
print(lemmatize(tokenize(data.comment[0])))


# Теперь очень долго вычисляем всё это

# In[ ]:

lemma_bags = data.comment.apply(replace_smiles).apply(tokenize).apply(lemmatize)


# # Регрессия

# Оказывается, встроенные инструменты позволяют куда более точную оценку тональности

# In[ ]:

new_texts = lemma_bags.apply(lambda x:' '.join(x))


# In[ ]:

def get_bigrams(lemmas):
    return [w1+'_'+w2 for w1, w2 in zip(lemmas[:-1], lemmas[1:])]
def add_bigrams(lemmas):
    return lemmas + get_bigrams(lemmas)
print(add_bigrams('мама мыла раму'.split()))


# In[ ]:

new_bigrams = lemma_bags.apply(add_bigrams).apply(lambda x:' '.join(x))


# In[ ]:

vec = TfidfVectorizer(min_df=5, max_df = 0.5)
vec.fit(new_texts)
print(len(vec.vocabulary_))


# Первый же заход даёт точность 38% (в эр-квадратах)

# In[ ]:

pipe = make_pipeline(vec, Ridge(1))
cross_val_score(pipe, new_texts, data.reting)


# Биграммы доводят уже до 44%

# In[ ]:

cross_val_score(pipe, new_bigrams[train_index], data.reting[train_index])


# In[ ]:

pipe.fit(new_bigrams[train_index], data.reting[train_index]);


# In[ ]:

inv_idx = {value:key for key, value in pipe.steps[0][1].vocabulary_.items()}
c = pipe.steps[1][1].coef_
best = [inv_idx[i] for i in np.argsort(c)[-10:]]
worst = [inv_idx[i] for i in np.argsort(c)[:10]]
print(', '.join(best))
print(', '.join(worst))


# In[ ]:

raw_preddiction = cross_val_predict(pipe, new_bigrams[train_index], data.reting[train_index])


# Картинка показывает, что прогноз модели может быть ещё здорово улучшен за счёт примерения нелинейных преобразований 

# In[ ]:

plt.scatter(raw_preddiction, data.reting[train_index]+np.random.normal(size = len(train_index), scale = 0.1), lw=0, s=1);
plt.plot([1, 5], [1, 5], color = 'red')

iso = IsotonicRegression(y_min = -10, y_max=10, out_of_bounds='clip').fit(raw_preddiction, data.reting[train_index])
plt.plot(pd.Series(raw_preddiction).sort_values(), iso.predict(pd.Series(raw_preddiction).sort_values()), color = 'red');


# Если нормально учесть нелинейность, модель ещё точнее

# In[ ]:

print(r2_score(data.reting[train_index], iso.predict(raw_preddiction)))


# # Проверка на тестовых данных 

# In[ ]:

test_predict = iso.predict(pipe.predict(new_bigrams[test_index]))


# На тестовых данных из будущего модель отрабатывает сильно хуже, чем на кросс-валидации - видимо, данные 2010-2015 года не совсем релевантны для прогнозов в 2016-2017 году.
# 

# In[ ]:

print(r2_score(data.reting[test_index], test_predict))


# Но в реальности мы бы могли дообучать модель хоть каждый день, поэтому данная проблема не так страшна. 
