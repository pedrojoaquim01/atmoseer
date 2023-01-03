import pandas as pd
import numpy as np
import sys, getopt, os, re
from datetime import datetime


df = pd.read_csv('../data/RIO DE JANEIRO - FORTE DE COPACABANA_1997_2022_ERA5_RAD_VENT_TEMP.csv')
df = df.drop(columns=['Unnamed: 0','DC_NOME','UF','DT_MEDICAO','CD_ESTACAO','VL_LATITUDE','VL_LONGITUDE','HR_MEDICAO'])
df.iloc[0] = df.iloc[0].fillna(0)
df = df[:].interpolate(method='linear')
print(df.info())
print(df.isnull().sum().sum())
print(df.columns[df.isna().any()].tolist())
print(df.columns)
