# -*- coding: utf-8 -*-
"""
A module with a copy of all functions for synthetic reviews from the hackathon.
"""
from test_problem import util
from num2t4ru import num2text
from sklearn.cluster import DBSCAN
import re
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

ROW_SEPARATOR = '\n*******\n'
PART_SEPARATOR = '\n***\n'


def insert_numbers(words):
    """Replace numeric words with digital numbers"""
    # todo: strip small letters from numbers, like 5-и, 5м
    ans = []
    for word in words:
        if word.isdigit():
            ans = ans + util.tokenize(num2text(int(word)))
        else:
            ans.append(word)
    return ans


def space_after_digits(word):
    digits = set('0123456789')
    ans = ''
    prev_is_digit = False
    for letter in word:
        is_digit = (letter in digits)
        if prev_is_digit and not is_digit:
            ans = ans + ' '
        ans = ans + letter
        prev_is_digit = is_digit
    return ans


def replace_punct(word):
    for symbol in r'-/:;':
        word = word.replace(symbol, ' ')
    return word


def get_trigrams(lemmas):
    return [w1+'_'+w2+'_'+w3
            for w1, w2, w3 in zip(lemmas[:-2], lemmas[1:-1], lemmas[2:])]


def bt(review):
    seq = util.tokenize(space_after_digits(
        replace_punct(re.sub('[^a-zA-Zа-яА-Яё]+', ' ', review))))
    arr = util.get_bigrams(seq) + get_trigrams(seq)
    return [a.replace('_', ' ') for a in arr]


def split_reviews(text, min_len=4):
    """ Takes one joint review and splits it into "phrases"
    """
    separators = '[\n\*,\.\!\(\)]'  # todo: move braces out?
    sents = [c.strip()
             for c in re.split(separators, text)
             if len(c) >= min_len]
    ans = []
    for sent in sents:
        ans.append(sent)
        if len(sent) > 40:
            # Long sentences additionally split by more separators
            for d in ')', '(', ';', '-', ':':
                if d in sent:
                    for c in sent.split(d):
                        if len(c) >= min_len:
                            ans.append(c)
            # Long sentences additionally split by conjunctions
            for d in ['и', 'хотя', 'но', 'a']:
                d = ' '+d+' '
                if d in sent:
                    for c in sent.split(d):
                        if len(c) >= min_len:
                            ans.append(c)
    return ans


def sent2words(sent):
    return util.lemmatize(insert_numbers(util.tokenize(
        space_after_digits(replace_punct(sent)))))


def longest(words):
    """ Returns the longest word in a text
    """
    if ' ' not in words:
        return words
    words = words.split()
    max_length = 0
    ans = ''
    for word in words:
        current_length = len(word)
        if current_length > max_length:
            max_length = current_length
            ans = word
    return ans


# todo: get w2v, vocab
def sent2vec(words, w2v, vocab, speller=None, verbose=False):
    """ Turns sentence into a vector
        Requires an instance of w2v gensim model, an instance of total_counter
        Also requires an instance of a speller
    """
    # print(words)
    total = np.zeros(w2v.syn0.shape[1])
    weights = 1e-4
    for word in words:
        word = word.replace('ё', 'е').strip()
        if word not in vocab and speller is not None:
            if verbose:
                print(word, 'not in vocab', end=' ')
            suggestions = []
            try:
                suggestions = [longest(c) for c in speller.suggest(word)]
            except MemoryError:
                pass
                # continue silently
            suggestions = [c for c in suggestions if c in vocab]
            if verbose:
                    print(suggestions)
            if suggestions:
                word = suggestions[0]
        if word in vocab:
            v = w2v.word_vec(word)
            # часто глобально встречающиеся слова мы взвешиваем вниз
            w = 1 # / np.sqrt(1+total_df[word])
            total = total + v
            weights = weights + w
        else:
            pass
    ans = total / weights
    return ans / np.sqrt(sum(ans**2)+1e-4)


# todo: get total_counter, total_df, w2v, vocab
def extract_facts(joined_reviews, total_counter, total_df, w2v, vocab,
                  verbose=False, eps=0.3, grams=False, censor=True):
    # split big review into different reviews and those into facts
    reviews = joined_reviews.split(ROW_SEPARATOR)

    if grams:
        sents2_raw = [bt(review) for review in reviews]
    else:
        sents2_raw = [split_reviews(review) for review in
                      reviews]  # + bt(review)
    sents2 = [[sent2words(r) for r in review] for review in sents2_raw]
    # print(len(sents2_raw), len(sents2))
    # seqs = [util.lemmatize(split_into_words(review)) for review in reviews]
    # sents2 = [util.get_bigrams(seq) + get_trigrams(seq) for seq in seqs]
    # sents2 = [[sent.split('_') for sent in sents] for sents in sents2 ]

    sents_flat = [sent for sents in sents2 for sent in sents]
    id2doc = [i for i, sents in enumerate(sents2) for sent in sents]
    vecs = [sent2vec(sent, w2v, vocab) for sents in sents2 for sent in sents]

    # print(sents_flat)
    # print([' '.join(s) for s in sents_flat])

    vecmat = pd.DataFrame(vecs).fillna(0)
    clu = DBSCAN(min_samples=2, eps=eps, p=2)
    # clu = MeanShift(bandwidth=eps)
    # clu = KMeans(n_clusters=5)
    result = clu.fit_predict(vecmat)
    if verbose:
        print(result)
    s2 = pd.Series(sents_flat)
    who2 = pd.Series(id2doc)

    sents_raw_flat = pd.Series(
        [sent for sents in sents2_raw for sent in sents])

    candidates = dict()
    representatives = dict()

    for ur in np.unique(result[result >= 0]):
        fltr = result == ur
        if sum(fltr) == 1 or sum(fltr) > 30:
            # too small or too large a cluster
            continue
        if who2[fltr].std() < 1e-5 and censor:
            # only one source
            continue
        if verbose:
            print()
            print(sum(fltr), who2[fltr].var())
        candidates[ur] = 0
        min_dist = np.inf
        representative = 'ERROR'
        center = vecmat[fltr].mean()
        for sent, vec, raw in zip(s2[fltr], vecmat[fltr],
                                  sents_raw_flat[fltr]):
            if verbose:
                print('\t', ' '.join(sent).strip())
            # accumulate total tf-idf
            cand_weight = sum([1 / np.log(1.01 + total_counter[word]) * (
            total_df[word] > 1) * len(word) for word in sent])
            candidates[ur] += cand_weight
            dist = sum((center - vec) ** 2) * 0.5 / cand_weight
            if dist < min_dist:
                min_dist = dist
                representative = raw
        representatives[ur] = representative

    result = pd.concat([pd.Series(representatives), pd.Series(candidates)],
                       axis=1)
    result.columns = ['center', 'importance']
    result.sort_values('importance', ascending=False, inplace=True)
    return result


def apply_to_new_data(d1, total_counter, total_df, w2v, vocab):
    """ Generate synthetic review from a dataframe """
    ps = '\n***\n'
    d1['total_text'] = (d1['SUBJECT'].fillna('')
                        + ps + d1['TEXT'].fillna('')
                        + ps + d1['BENEFITS'].fillna('')
                        + ps + d1['DRAWBACKS'].fillna(''))
    text_by_product = d1[['total_text']].groupby(d1['NAME']).aggregate(
        lambda x: ROW_SEPARATOR.join(x))
    ans = dict()
    for key, value in text_by_product.iterrows():
        facts = extract_facts(value.values[0], total_counter, total_df,
                              w2v, vocab, True, 0.4).center
        if len(facts) == 0:
            facts = extract_facts(value.values[0], total_counter, total_df,
                                  w2v, vocab, True, 0.4, False, False).center
        if len(facts) == 0:
            facts = extract_facts(value.values[0], total_counter, total_df,
                                  w2v, vocab, True, 0.4, True, False).center
        val = '\n > '.join(['Пользователи пишут:']
                           + list(facts.apply(lambda x: x.capitalize())))
        ans[key] = val
    return ans


def prepare_texts_for_counters(d1, text_columns=None, group_column=None):
    """ Preprocess comments and group them by products
    """
    if text_columns is None:
        text_columns = ['TEXT', 'SUBJECT', 'BENEFITS', 'DRAWBACKS']
    prep_columns = [c + '_prepared' for c in text_columns]
    if group_column is None:
        group_column = 'PRODUCT'
    # prepare 4 columns of comments
    for old, new in zip(text_columns, prep_columns):
        l = []
        for text in tqdm(d1[old]):
            l.append(
                ' '.join(util.lemmatize(util.tokenize(util.replace_smiles(text)))))
        d1[new] = l

    # join columns
    d1['total_text'] = d1[text_columns[0]].fillna('')
    for c in text_columns[1:]:
        d1['total_text'] = d1['total_text'] + PART_SEPARATOR + d1[c].fillna('')

    d1['total_text_prepared'] = d1[prep_columns[0]].fillna('')
    for c in prep_columns[1:]:
        d1['total_text_prepared'] = d1['total_text_prepared'] \
                                    + PART_SEPARATOR + d1[c].fillna('')

    # aggregate by products
    text_by_product = d1[['total_text', 'total_text_prepared']].groupby(
        d1['PRODUCT']).aggregate(lambda x: ROW_SEPARATOR.join(x))
    return text_by_product


def learn_counters(prepared_texts):
    """ Prepare total_counter and total_df
    This takes REALLY long to work
    Args:
         prepared_texts (list(str)): list of (normalized) texts
    """
    total_counter = Counter()
    counters = []
    print('Counting individual words...')
    for text in tqdm(prepared_texts):
        counters.append(Counter(util.tokenize(text)))
    print('Merging counter...')
    for c in tqdm(counters):
        total_counter += c
    total_df = Counter()
    for c in tqdm(counters):
        for w in c:
            total_df[w] += 1  # Not c[w], but one! Duplicates matter not!
    return total_counter, total_df
