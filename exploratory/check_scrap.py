
from tqdm import tqdm

import pickle
import numpy as np
import os
from multiprocessing import Pool

import tmp


INPUT_ROOT = os.path.join('..','INPUT')

def get_price_tuple_stubborn(product):
    p0, p1 = 0, 'error'
    while p1 == 'error':
        p0, p1 = tmp.get_price_tuple(product)
    return product, p1

if __name__ == '__main__':
    with open('scrap_result.pkl', 'rb') as file:
        result = pickle.load(file)
    print('len', len(result))
    erroneous = [r[0] for r in result if r[1]=='error']
    print('share of errors', len(erroneous)/len(result))
    # now fix all the errors with a new loop
    price_result_dict = dict(result)
    print('working on errors...')
    # small pool: not want to overload
    p = Pool(3)
    max_ = len(erroneous)
    with tqdm(total=max_) as pbar:
        for i, tup in tqdm(enumerate(p.imap_unordered(get_price_tuple_stubborn, erroneous))):
            price_result_dict[tup[0]] = tup[1]
            pbar.update()           
        pbar.close()
        p.close()
        p.join()
    print('finished!')
    with open('price_result_dict.pkl', 'wb') as file:
        pickle.dump(result, file)
    