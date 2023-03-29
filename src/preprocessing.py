import pandas as pd
import numpy as np
from pathlib import Path
from math import cos, asin, sqrt
import sys
import getopt
import xarray as xr
from utils.near_stations import prox
from datetime import datetime, timedelta
from metpy.calc import wind_components
from metpy.units import units

    # if arquivo in cor_est:
    #     wind_speed = df['VelVento']
    #     wind_direction = df['DirVento']
    # else:
    #     wind_speed = df['VEN_VEL']
    #     wind_direction = df['VEN_DIR']
# def transform_wind(wind_speed, wind_direction):
#     wv = wind_speed / 3.6
#     wd_rad = wind_direction * np.pi / 180
#     wind_x = wv * np.cos(wd_rad)
#     wind_y = wv * np.sin(wd_rad)
#     return wind_x, wind_y

def transform_wind(wind_speed, wind_direction):
    """
    Calculate the U, V wind vector components from the speed and direction.
    """
    return wind_components(wind_speed * units('m/s'), wind_direction * units.deg)

    # if arquivo in cor_est:
    #     date_time = pd.to_datetime(df['Dia'] +' '+ df['Hora'], format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
    # else:
    #     date_time = pd.to_datetime(df['DT_MEDICAO'] + ' '+ df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(0, 2) + ':' + df['HR_MEDICAO'].apply(lambda x: '{0:0>4}'.format(x)).str.slice(2, 4) + ':00', format='%Y-%m-%d%H:%M:%S', infer_datetime_format=True)
# def transform_time(df, date_time):
#     timestamp_s = date_time.map(pd.Timestamp.timestamp)
#     day = 24*60*60
#     year = (365.2425)*day
#     df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
#     df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
#     df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
#     df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

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
    use_sounding_as_data_source = 0
    use_numerical_model_as_data_source = 0
    num_neighbors = 0
    help_message = "Usage: {0} -f <file> -d <data_source_spec> -n <num_neighbors>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hf:d:n:", ["help", "file=", "datasources=", "neighbors="])
    except:
        print(help_message)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_message)  # print the help message
            sys.exit(2)
        elif opt in ("-f", "--file"):
            arg_file = arg
        elif opt in ("-d", "--datasources"):
            if arg.find('R') != -1:
                use_sounding_as_data_source = 1
            if arg.find('N') != -1:
                use_numerical_model_as_data_source = 1
        elif opt in ("-n", "--neighbors"):
            num_neighbors = arg

if __name__ == "__main__":
    main(sys.argv)