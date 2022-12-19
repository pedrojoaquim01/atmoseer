import pandas as pd
import numpy as np
from pathlib import Path
from math import cos, asin, sqrt

cor_est = ['alto_da_boa_vista','guaratiba','iraja','jardim_botanico','riocentro','santa_cruz','sao_cristovao','vidigal']
for est in cor_est:
    fonte = './data/'+ est +'.csv'
    df = pd.read_csv(fonte)
    del df['Unnamed: 0']
    df = df[df['Hora'].str[2:6] == ':00:']
    df.to_csv('./data/'+ est +'.csv')

