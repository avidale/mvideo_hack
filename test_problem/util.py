# coding: utf-8
import numpy as np
import pandas as pd
import pymorphy2
import nltk
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
    
def tokenize(text):
    return [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

morph = pymorphy2.MorphAnalyzer()
normal_forms = {}
def get_normal_form(word):
    """Normalize the word with pymorphy2 or fetch from cache
    """
    if word in normal_forms:
        return normal_forms[word]
    else:
        normal_form = morph.parse(word)[0].normal_form
        normal_forms[word] = normal_form
        return normal_form

def lemmatize(words):
    # ToDo: maybe make parallel, because it is VERY slow
    return [get_normal_form(word) for word in words]

def get_bigrams(lemmas):
    return [w1+'_'+w2 for w1, w2 in zip(lemmas[:-1], lemmas[1:])]

def add_bigrams(lemmas):
    return lemmas + get_bigrams(lemmas)
    
def prepare_texts(texts, bigrams = False, join = True):
    """ Apply the full preprocessing pipeline to texts
    """
    #ToDo: also remove punctuation, or maybe make special punctuation features.
    result = pd.Series(texts).apply(replace_smiles).apply(tokenize).apply(lemmatize)
    if bigrams:
        result = result.apply(add_bigrams)
    if join:
        result = result.apply(lambda x:' '.join(x))
    return result
