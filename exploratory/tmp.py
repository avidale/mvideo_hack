

from multiprocessing import Pool
import time

import os
import pandas as pd

from bs4 import BeautifulSoup
import re
import requests

def get_price(product):
    url = r'http://www.mvideo.ru/products/'+str(product)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    div = soup.find('div', {'class': 'c-pdp-price__current sel-product-tile-price'})
    if div is None:
        return None
    price_text = re.sub('\D', '', div.contents[0])
    price = int(price_text)
    return price

    
result = dict()

def get_price_tuple(product):
    try: 
        p = get_price(product)
    except:
        p = "error"
    return product, p

def add_price(product):
    global result
    result[product] = get_price(product)
    
from tqdm import tqdm

import pickle


INPUT_ROOT = os.path.join('..','INPUT')

if __name__ == '__main__':
    d1 = pd.read_csv(os.path.join(INPUT_ROOT, 'dataset1.csv'), index_col=0)
    products = list(d1.PRODUCT.value_counts().index)
    print('products ready')
    start_time = time.time()
    p = Pool(10)
    max_ = len(products)
    result = []
    with tqdm(total=max_) as pbar:
        for i, tup in tqdm(enumerate(p.imap_unordered(get_price_tuple, products[:max_]))):
            result.append(tup)
            pbar.update()           
        pbar.close()
        p.close()
        p.join()
    elapsed_time = time.time() - start_time
    print()
    print(elapsed_time, 'seconds')
    print(len(result))
    pickle.dump(result, open('scrap_result.pkl', 'wb'))