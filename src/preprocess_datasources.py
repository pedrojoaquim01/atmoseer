import pandas as pd
import numpy as np
from pathlib import Path
import sys
import getopt
from globals import *
from util import *

def pre_process_sounding_data(sounding_data_source):
    df_sounding = pd.read_csv(sounding_data_source)
    format_string = '%Y-%m-%d %H:%M:%S'
    df_sounding['time'] = df_sounding['time'].apply(lambda x: utc_to_local(x, "America/Sao_paulo", format_string))

    df_sounding['Datetime'] = pd.to_datetime(df_sounding['time'], format=format_string)

    df_sounding = df_sounding.set_index(pd.DatetimeIndex(df_sounding['Datetime']))
    df_sounding = df_sounding.drop(['time', 'Datetime'], axis = 1)
    filename_and_extension = get_filename_and_extension(sounding_data_source)
    preprocessed_filename = filename_and_extension[0] + '_preprocessed' + filename_and_extension[1]
    df_sounding.to_csv(preprocessed_filename)
    df_sounding.to_parquet(filename_and_extension[0] + '_preprocessed.parquet.gzip', compression='gzip')

def main(argv):
    arg_file = ""
    sounding_data_source = None
    numerical_model_data_source = None
    num_neighbors = 0
    help_message = "Usage: {0} -s <station_id> -d <data_source_spec> -n <num_neighbors>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hs:d:n:", ["help", "station_id=", "datasources=", "neighbors="])
    except:
        print("Invalid syntax!")
        print(help_message)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_message)  # print the help message
            sys.exit(2)
        elif opt in ("-s", "--station"):
            station_id = arg
            if not ((station_id in INMET_STATION_CODES_RJ) or (station_id in COR_STATION_NAMES_RJ)):
                print(f"Invalida station identifier: {station_id}")
                print(help_message)
                sys.exit(2)
        elif opt in ("-f", "--file"):
            ws_data = arg
        elif opt in ("-d", "--datasources"):
            if arg.find('R') != -1:
                sounding_data_source = '../data/sounding_stations/SBGL_indices_1997-01-01_2022-12-31.csv'
            if arg.find('N') != -1:
                numerical_model_data_source = '../data/numerical_models/ERA5_A652 _1997-01-01_2021-12-31.csv'
        elif opt in ("-n", "--neighbors"):
            num_neighbors = arg

    print('Going to preprocess the specified data sources...')

    if sounding_data_source is not None:
        pre_process_sounding_data(sounding_data_source)

if __name__ == "__main__":
    main(sys.argv)