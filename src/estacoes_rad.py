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
    
    
    df_s['showalter'] = 0.0
    df_s['lift_index'] = 0.0
    df_s['k_index'] = 0.0
    df_s['total_totals'] = 0.0
    df_s['parcel_profile'] = 0.0

    for tempo in df.time.unique():
        df_aux = df_s[df_s['time'] == tempo]
        df_aux.drop_duplicates(inplace=True,subset='pressure',ignore_index=True)
        try:
            parcel_profile = mpcalc.parcel_profile(df_aux['pressure'].to_numpy() * units.hPa , df_aux['temperature'][0] * units.degC, df_aux['dewpoint'][0] * units.degC)
            df.loc[df_s['time'] == tempo,'parcel_profile'] =  parcel_profile.magnitude
        except:
            df.loc[df_s['time'] == tempo,'parcel_profile'] = 0

    for tempo in df.time.unique():
            df_aux = df_s[df_s['time'] == tempo]
            
            try:
                showalter = mpcalc.showalter_index(df_aux['pressure'].to_numpy() * units.hPa , df_aux['temperature'].to_numpy() * units.degC, df_aux['dewpoint'].to_numpy() * units.degC)
                df.loc[df_s['time'] == tempo,'showalter'] = showalter.magnitude[0]
            except:
                df.loc[df_s['time'] == tempo,'showalter'] = 0


            try:
                k_index = mpcalc.k_index(df_aux['pressure'].to_numpy() * units.hPa , df_aux['temperature'].to_numpy() * units.degC, df_aux['dewpoint'].to_numpy() * units.degC)
                df.loc[df_s['time'] == tempo,'k_index'] = k_index.magnitude
            except:
                df.loc[df_s['time'] == tempo,'k_index'] = 0

            try:
                total_totals = mpcalc.total_totals_index(df_aux['pressure'].to_numpy() * units.hPa , df_aux['temperature'].to_numpy() * units.degC, df_aux['dewpoint'].to_numpy() * units.degC)
                df.loc[df_s['time'] == tempo,'total_totals'] = total_totals.magnitude
            except:
                df.loc[df_s['time'] == tempo,'total_totals'] = 0
                
            
            try:
                lift_index = mpcalc.lifted_index(df_aux['pressure'].to_numpy() * units.hPa , df_aux['temperature'].to_numpy() * units.degC, df_aux['parcel_profile'].to_numpy() * units.degC)
                df.loc[df_s['time'] == tempo,'lift_index'] = lift_index[0].astype(float)
            except:
                df.loc[df_s['time'] == tempo,'lift_index'] = 0

    df_s.to_csv('sondas_completo.csv',index=False)

if __name__ == "__main__":
    processamento()
