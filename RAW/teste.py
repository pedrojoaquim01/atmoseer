import pandas as pd
import numpy as np
import sys, getopt, os, re
from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir
from metpy.units import pandas_dataframe_to_unit_arrays
from metpy.units import units
import metpy.calc as mpcalc


df = pd.read_csv('../data/sondas_CAPE_CIN.csv')
    
df['showalter'] = 0.0
df['lift_index'] = 0.0
df['k_index'] = 0.0
df['total_totals'] = 0.0
df['parcel_profile'] = 0.0

for tempo in df.time.unique():
    df_aux = df[df['time'] == tempo]
    df_aux.drop_duplicates(inplace=True,subset='pressure',ignore_index=True)
    try:
        parcel_profile = mpcalc.parcel_profile(df_aux['pressure'].to_numpy() * units.hPa , df_aux['temperature'][0] * units.degC, df_aux['dewpoint'][0] * units.degC)
        df.loc[df['time'] == tempo,'parcel_profile'] =  parcel_profile.magnitude
    except:
        df.loc[df['time'] == tempo,'parcel_profile'] = 0

for tempo in df.time.unique():
        df_aux = df[df['time'] == tempo]
        
        try:
            showalter = mpcalc.showalter_index(df_aux['pressure'].to_numpy() * units.hPa , df_aux['temperature'].to_numpy() * units.degC, df_aux['dewpoint'].to_numpy() * units.degC)
            df.loc[df['time'] == tempo,'showalter'] = showalter.magnitude
        except:
            df.loc[df['time'] == tempo,'showalter'] = 0


        try:
            k_index = mpcalc.k_index(df_aux['pressure'].to_numpy() * units.hPa , df_aux['temperature'].to_numpy() * units.degC, df_aux['dewpoint'].to_numpy() * units.degC)
            df.loc[df['time'] == tempo,'k_index'] = k_index.magnitude
        except:
            df.loc[df['time'] == tempo,'k_index'] = 0

        try:
            total_totals = mpcalc.total_totals_index(df_aux['pressure'].to_numpy() * units.hPa , df_aux['temperature'].to_numpy() * units.degC, df_aux['dewpoint'].to_numpy() * units.degC)
            df.loc[df['time'] == tempo,'total_totals'] = total_totals.magnitude
        except:
            df.loc[df['time'] == tempo,'total_totals'] = 0
            
        
        try:
            lift_index = mpcalc.lifted_index(df_aux['pressure'].to_numpy() * units.hPa , df_aux['temperature'].to_numpy() * units.degC, df_aux['parcel_profile'].to_numpy() * units.degC)
            df.loc[df['time'] == tempo,'lift_index'] = lift_index[0].astype(float)
        except:
            df.loc[df['time'] == tempo,'lift_index'] = 0
            
        
        
        
                
        

df.to_csv('sondas_completo.csv',index=False)