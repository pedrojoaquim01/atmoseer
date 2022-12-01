import pandas as pd
import numpy as np
import sys, getopt, os, re
from datetime import datetime


df = pd.read_csv('../data/RIO DE JANEIRO - FORTE DE COPACABANA_CAPE_CIN.csv')

df['CAPE'][0] = 0
df['CIN'][0] = 0
df = df.interpolate(method='linear')


df.to_csv('RIO DE JANEIRO - FORTE DE COPACABANA_CAPE_CIN.csv',index=False)