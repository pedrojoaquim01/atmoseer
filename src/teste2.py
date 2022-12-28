import pandas as pd
import numpy as np
import sys, getopt, os, re
from datetime import datetime


df = pd.read_csv('../data/RIO DE JANEIRO - FORTE DE COPACABANA_1997_2022_ERA5_RAD_VENT_TEMP.csv')
df['data'] = pd.to_datetime(df['DT_MEDICAO'] + ' '+ df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(0, 2) + ':' + df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(2, 4) + ':00', format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
del df['Unnamed: 0']    
df_aux = df
result = ['RIO DE JANEIRO - JACAREPAGUA_1997_2022_ERA5_RAD_VENT_TEMP','RIO DE JANEIRO - VILA MILITAR_1997_2022_ERA5_RAD_VENT_TEMP','RIO DE JANEIRO-MARAMBAIA_1997_2022_ERA5_RAD_VENT_TEMP']
count = 0
for s in result:
    fonte = '../data/'+s+'.csv'
    df1 = pd.read_csv(fonte)
    df1 = df1.fillna(0)
    del df1['Unnamed: 0']
    
    df1['data'] = pd.to_datetime(df1['DT_MEDICAO'] + ' '+ df1['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(0, 2) + ':' + df1['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(2, 4) + ':00', format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
    
    suf = str(count)
    count += 1

    df1 = df1.drop(columns=['DC_NOME','UF','DT_MEDICAO','CD_ESTACAO','VL_LATITUDE','VL_LONGITUDE','HR_MEDICAO'])

    df1 = df1.add_suffix('_' + suf)
    df1 = df1.rename(columns={"data_" + suf : "data"})
    
    print(df_aux)
    print(df1)

    df_aux = pd.merge(df_aux,df1, how = 'outer')
    
    df_aux = df_aux.sort_values(by=['DT_MEDICAO','HR_MEDICAO'])

df_final = df_aux[df_aux['data'].isin(df['data'])]
del df_final['data']
df_final.to_csv('../data/FORTE DE COPACABANA INMET.csv')    
