import pandas as pd
import numpy as np
import sys, getopt, os, re
from datetime import datetime
import pickle
from pre_processing import pre_proc

df_estacoes = pd.read_csv('../data/estacoes_local.csv')
lista = df_estacoes['files'].to_list()

for i in lista:
    print(i)
    pre_proc(i,log_CAPE = 1,log_Vento = 1,log_Tempo = 1, log_era = 0)
    pre_proc(i,log_CAPE = 0,log_Vento = 1,log_Tempo = 1, log_era = 1)
    pre_proc(i,log_CAPE = 1,log_Vento = 1,log_Tempo = 1, log_era = 1)