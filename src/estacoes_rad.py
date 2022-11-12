import pandas as pd
import numpy as np
import sys, getopt, os, re
from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir
from metpy.units import pandas_dataframe_to_unit_arrays
from metpy.units import units
import metpy.calc as mpcalc

def processamento():
    station = 'SBGL'
    a = np.array([[datetime(1997, 1, 1, 0), '0']])
    df_s = pd.DataFrame()
    for y in range(1997,2023):
        for m in range(1,13):
            for d in range(1,32):
                for h in [0,12]:
                    try:
                        date = datetime(y,m,d,h)
                        try:
                            df = WyomingUpperAir.request_data(date, station)
                            #data = pandas_dataframe_to_unit_arrays(df)
                            #x = mpcalc.surface_based_cape_cin(data['pressure'], data['temperature'], data['dewpoint'])
                            df_s = pd.concat(([df_s,df]))
                            #b = np.array([[date, x[0]]])
                            #a = np.concatenate((a, b), axis=0)
                        except:
                            df = pd.DataFrame()
                    except:
                        pass
    
    df_s['CAPE'] = ''
    df_s['CIN'] = ''

    for tempo in df_s.time.unique():
        df_aux = df_s[df_s['time'] == tempo]
        
        try:
            CAPE = mpcalc.surface_based_cape_cin(df_aux['pressure'].to_numpy() * units.hPa , df_aux['temperature'].to_numpy() * units.degC, df_aux['dewpoint'].to_numpy() * units.degC)
            df_s.loc[df_s['time'] == tempo,'CAPE'] = CAPE[0].astype(float)
            df_s.loc[df_s['time'] == tempo,'CIN'] = CAPE[1].astype(float)
        except:
            CAPE = [0,0]
            df_s.loc[df_s['time'] == tempo,'CAPE'] = CAPE[0]
            df_s.loc[df_s['time'] == tempo,'CIN'] = CAPE[1]
    
    df_s.to_csv('sondas_CAPE_CIN.csv',index=False)

if __name__ == "__main__":
    processamento()
