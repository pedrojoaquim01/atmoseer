import pandas as pd
import numpy as np
from pathlib import Path
from math import cos, asin, sqrt
import sys
import getopt
from utils.near_stations import prox
from metpy.calc import wind_components
from metpy.units import units
from globals import *

def transform_wind(wind_speed, wind_direction):
    """
    Calculate the U, V wind vector components from the speed and direction.
    """
    return wind_components(wind_speed * units('m/s'), wind_direction * units.deg)

def transform_hour(df):
    """
    Transforms a DataFrame's datetime index into two new columns representing the hour in sin and cosine form.

    Args:
    - df: A pandas DataFrame with a datetime index.

    Returns:
    - A pandas DataFrame with two new columns named 'hour_sin' and 'hour_cos' representing the hour in sin and cosine form.
    """
    dt = df.index
    hourfloat = dt.hour + dt.minute/60.0
    df['hour_sin'] = np.sin(2. * np.pi * hourfloat/24.)
    df['hour_cos'] = np.cos(2. * np.pi * hourfloat/24.)
    return df

def main(argv):
    arg_file = ""
    sounding_data_source = None
    numerical_model_data_source = None
    num_neighbors = 0
    help_message = "Usage: {0} -s <file> -d <data_source_spec> -n <num_neighbors>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hf:d:n:", ["help", "file=", "datasources=", "neighbors="])
    except:
        print(help_message)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_message)  # print the help message
            sys.exit(2)
        elif opt in ("-s", "--station"):
            station_id = arg
            if not ((station_id in INMET_STATION_CODES_RJ) or station_id in COR_STATION_NAMES_RJ):
                print(help_message)
                sys.exit(2)
        elif opt in ("-f", "--file"):
            ws_data = arg
        elif opt in ("-d", "--datasources"):
            if arg.find('R') != -1:
                sounding_data_source = '../data/sounding/SBGL_indices_1997-01-01_2022-12-31.csv'
            if arg.find('N') != -1:
                numerical_model_data_source = '../data/numerical_models/ERA5_A652 _1997-01-01_2021-12-31.csv'
        elif opt in ("-n", "--neighbors"):
            num_neighbors = arg

if __name__ == "__main__":
    main(sys.argv)