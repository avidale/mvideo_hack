
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
# 
# Качество меряется метрикой R^2, поскольку кажется, что путать далёкие рейтинги - сильно опаснее, чем путать соседние. 

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

from sklearn.model_selection import cross_val_score, cross_val_predict, TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score

from xgboost import XGBRegressor, XGBClassifier


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


# Делаем целочисленный индекс - авось, пригодится. 

# In[8]:

data['reting_int'] = data.reting.round(0).astype(int)
data['reting_int'].value_counts().sort_index()


# Разбираемся с временным периодом

# In[9]:

data.date = pd.to_datetime(data.date)


# In[10]:

data.date.hist(bins=30);


# In[11]:

print((data.date<pd.Timestamp('2016-01-01')).mean())


# В качестве теста можно взять 2016-2017 годы, это будет честно.

# In[12]:

data.sort_values('date', inplace=True)


# Из года в год средняя оценка колебалась, но не то чтобы значительно

# In[13]:

data.reting.groupby(data.date.dt.round('180d')).mean().plot();


# Разделяем выборку на тестовую и тренировочную

# In[14]:

train_index = data[data.date<pd.Timestamp('2016-01-01')].index
test_index = data[data.date>=pd.Timestamp('2016-01-01')].index
print('train:{}, test:{} obs'.format(train_index.shape[0], test_index.shape[0]))


# Готовим кроссс-валидацию

# In[15]:

cv = TimeSeriesSplit(3)


# # Предобработка

# Заменяем смайлики и прочие средства выразительности на служебные слова

# In[16]:

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

# In[17]:

def tokenize(text):
    return [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
print(tokenize(data.comment[0]))


# Приводим слова к нормальной форме

# In[18]:

morph = pymorphy2.MorphAnalyzer()
# кэшируем слова, с которыми уже работали
normal_forms = {}
# если слова нет в кэше, пользуемся анализатором и добавляем его туда
def get_normal_form(word):
    if word in normal_forms:
        return normal_forms[word]
    else:
        normal_form = morph.parse(word)[0].normal_form
        normal_forms[word] = normal_form
        return normal_form


# In[19]:

def lemmatize(words):
    return [get_normal_form(word) for word in words]
print(lemmatize(tokenize(data.comment[0])))


# Теперь очень долго вычисляем всё это. Правда, второй раз будет проще. 

# In[20]:

lemma_bags = data.comment.apply(replace_smiles).apply(tokenize).apply(lemmatize)


# # Регрессия

# ### Линейная регрессия

# In[21]:

new_texts = lemma_bags.apply(lambda x:' '.join(x))


# In[22]:

def get_bigrams(lemmas):
    return [w1+'_'+w2 for w1, w2 in zip(lemmas[:-1], lemmas[1:])]
def add_bigrams(lemmas):
    return lemmas + get_bigrams(lemmas)
print(add_bigrams('мама мыла раму'.split()))


# In[23]:

new_bigrams = lemma_bags.apply(add_bigrams).apply(lambda x:' '.join(x))


# In[24]:

vec = TfidfVectorizer(min_df=3)
vec.fit(new_texts)
print(len(vec.vocabulary_))


# Первый же заход даёт точность 36% (в эр-квадратах) на худшем фолде

# In[25]:

pipe = make_pipeline(vec, Ridge(1))
cross_val_score(pipe, new_texts, data.reting, cv= cv, n_jobs=-1)


# Биграммы доводят уже до 40%

# In[26]:

cross_val_score(pipe, new_bigrams[train_index], data.reting[train_index], cv = cv, n_jobs=-1)


# In[27]:

pipe.fit(new_bigrams[train_index], data.reting[train_index]);


# In[ ]:

inv_idx = {value:key for key, value in pipe.steps[0][1].vocabulary_.items()}
c = pipe.steps[1][1].coef_
best = [inv_idx[i] for i in np.argsort(c)[-10:]]
worst = [inv_idx[i] for i in np.argsort(c)[:10]]
print(', '.join(best))
print(', '.join(worst))


# Для получения сырого предсказания немножко нарушим нашу кросс-валидацию

# In[ ]:

raw_preddiction = cross_val_predict(pipe, new_bigrams[train_index], data.reting[train_index], n_jobs=-1)


# Картинка показывает, что прогноз модели может быть ещё здорово улучшен за счёт примерения нелинейных преобразований 

# In[ ]:

plt.scatter(raw_preddiction, data.reting[train_index]+np.random.normal(size = len(train_index), scale = 0.1), lw=0, s=1);
plt.plot([1, 5], [1, 5], color = 'red')

iso = IsotonicRegression(y_min = 1, y_max=5, out_of_bounds='clip').fit(raw_preddiction, data.reting[train_index])
plt.plot(pd.Series(raw_preddiction).sort_values(), iso.predict(pd.Series(raw_preddiction).sort_values()), color = 'red');


# Если нормально учесть нелинейность, модель ещё точнее: на худшем фолде получается почти 50% (но тут немножко данных из будущего)

# In[ ]:

cross_val_score(iso, raw_preddiction, data.reting[train_index], cv = cv, n_jobs=-1)


# ### Градиентный бустинг

# In[ ]:

xgbr = XGBRegressor(max_depth = 9, n_estimators = 100, learning_rate = 0.1)
vec_xgbr = TfidfVectorizer(min_df=3)
pipe_xgb = make_pipeline(vec_xgbr, xgbr)


# In[ ]:

cross_val_score(pipe_xgb, new_bigrams[train_index], data.reting[train_index], cv = cv, n_jobs=-1)


# In[ ]:

pipe_xgb.fit(new_bigrams[train_index], data.reting[train_index]);


# In[ ]:

pred_gb =  cross_val_predict(pipe_xgb, new_bigrams[train_index], data.reting[train_index], n_jobs=-1)


# Пробуем смешать две модели, и получаем результат лучше, чем просто линейная модель. Но не намного. 

# In[ ]:

w_ridge = 0.6
pred_regression = raw_preddiction*w_ridge + pred_gb*(1-w_ridge)
cross_val_score(iso, pred_regression, data.reting[train_index], cv = cv, n_jobs=-1)


# ## Классификация

# Смотрим, насколько хорошо справляются с нашей задачей классификаторы.
# 
# По эр-квадрату оказывается, что сильно хуже, но мало ли, вдруг всё равно пригодятся

# In[ ]:

from sklearn.linear_model import LogisticRegression


# In[ ]:

lr = LogisticRegression(C=1e1, class_weight = 'balanced', multi_class  = 'multinomial', solver = 'lbfgs')
vec_lr = TfidfVectorizer(min_df=3)
pipe_lr = make_pipeline(vec_lr, lr)


# In[ ]:

cross_val_score(pipe_lr, new_bigrams[train_index], data.reting_int[train_index], cv = cv, scoring = 'r2', n_jobs = -1)


# In[ ]:

from collections import Counter
cnt = Counter(data.reting_int[train_index])
weights = {key:sum(cnt.values())/value for key, value in cnt.items()}
sample_weights_data = [weights[k] for k in data.reting_int[train_index]]


# In[ ]:

from sklearn.svm import LinearSVC


# In[ ]:

svm1 = LinearSVC(C=1, class_weight = 'balanced', loss = 'squared_hinge', penalty = 'l2', random_state = 42)
vec_svm1 = TfidfVectorizer(min_df=3)
pipe_svm1 = make_pipeline(vec_svm1, svm1)


# In[ ]:

cross_val_score(pipe_svm1, new_bigrams[train_index], data.reting_int[train_index], cv = cv, scoring = 'r2', n_jobs=-1)


# Бустинг на классификацию отрабатывает вообще никакуще - возможно, потому что не могу задать class_weight

# In[ ]:

xgbc = XGBClassifier(max_depth = 3, n_estimators = 100, learning_rate = 0.1)
vec_xgbc = TfidfVectorizer(min_df=3)
pipe_xgbc = make_pipeline(vec_xgbc, xgbc)


# In[ ]:

cross_val_score(pipe_xgbc, new_bigrams[train_index], data.reting_int[train_index], cv = cv, scoring = 'r2', n_jobs=-1)


# In[ ]:

pred_lr = cross_val_predict(pipe_lr, new_bigrams[train_index], data.reting[train_index], n_jobs=-1)
pred_svm1 = cross_val_predict(pipe_svm1, new_bigrams[train_index], data.reting[train_index], n_jobs=-1)


# Смешиваем классификаторы друг с другом и с регрессорами. Но регрессорам это не очень на пользу, полпроцента едва подняли.  
# 
# Может быть, потому что нужно вытаскивать не predict, а decision_function из классификаторов, и дорабатывать. 

# In[ ]:

w_lr = 0.5
w_clf = 0.1
pred_clf = pred_lr * w_lr + pred_svm1 * (1-w_lr)
pred_all = pred_clf * w_clf + pred_regression * (1-w_clf)
print(cross_val_score(iso, pred_clf, data.reting[train_index], cv = cv, n_jobs=-1))
print(cross_val_score(iso, pred_all, data.reting[train_index], cv = cv, n_jobs=-1))


# # Проверка на тестовых данных 

# Линейный регрессор отработал лучше всех, так что выберем его. 
# 
# Смешивать не будем, т.к. дороже поддерживать модель, чем получать жалкие нестабильные проценты качества. 

# In[ ]:

test_predict = iso.predict(pipe.predict(new_bigrams[test_index]))


# На тестовых данных из будущего модель отрабатывает сильно хуже, чем на кросс-валидации - видимо, данные 2010-2015 года не совсем релевантны для прогнозов в 2016-2017 году.
# 

# In[ ]:

print(r2_score(data.reting[test_index], test_predict))


# Но в реальности мы бы могли дообучать модель хоть каждый день, поэтому данная проблема не так страшна. 
