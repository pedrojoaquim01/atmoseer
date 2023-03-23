import pandas as pd
import numpy as np
import sys, getopt, os, re
from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir
from metpy.units import pandas_dataframe_to_unit_arrays
from metpy.units import units
import metpy.calc as mpcalc

def processamento(arg_begin = 1997,arg_end = 2022):
    station = 'SBGL'
    a = np.array([[datetime(arg_begin, 1, 1, 0), '0']])
    df_s = pd.DataFrame()
    for y in range(arg_begin,arg_end + 1):
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
    
    df_s.to_csv('../data/sondas.csv',index=False)

def myfunc(argv):
    arg_begin = 1997
    arg_end = 2022
    arg_help = "{0} -b <begin> -e <end>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hb:e:", ["help","begin=","end="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-b", "--begin"):
            arg_begin = arg
        elif opt in ("-e", "--end"):
            arg_end = arg

    processamento(arg_begin,arg_end)


if __name__ == "__main__":
    myfunc(sys.argv)
