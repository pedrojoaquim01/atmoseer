import pandas as pd
import numpy as np
from pathlib import Path
import sys
import getopt
from globals import *
from util import transform_wind, format_time, get_filename_and_extension, utc_to_local

def preprocess_sounding_data(sounding_data_source):
    df = pd.read_csv(sounding_data_source)
    format_string = '%Y-%m-%d %H:%M:%S'
    df['time'] = df['time'].apply(lambda x: utc_to_local(x, "America/Sao_paulo", format_string))

    #
    # Add index to dataframe using the timestamps.
    df['Datetime'] = pd.to_datetime(df['time'], format=format_string)
    df = df.set_index(pd.DatetimeIndex(df['Datetime']))
    print(f"Range of timestamps after preprocessing {sounding_data_source}: [{min(df.index)}, {max(df.index)}]")

    #
    # Remove irrelevant columns
    df = df.drop(['time', 'Datetime'], axis = 1)

    #
    # Save preprocessed data source file
    filename_and_extension = get_filename_and_extension(sounding_data_source)
    preprocessed_filename = filename_and_extension[0] + '_preprocessed' + filename_and_extension[1]
    df.to_parquet(filename_and_extension[0] + '_preprocessed.parquet.gzip', compression='gzip')

def preprocess_numerical_model_data(numerical_model_data_source):
    df = pd.read_csv(numerical_model_data_source)
    format_string = '%Y-%m-%d %H:%M:%S'
    df['time'] = df['time'].apply(lambda x: utc_to_local(x, "America/Sao_paulo", format_string))

    df['Datetime'] = pd.to_datetime(df['time'], format=format_string)

    #
    # Add index to dataframe using the timestamps.
    df = df.set_index(pd.DatetimeIndex(df['Datetime']))
    df = df.drop(['time', 'Datetime', 'Unnamed: 0'], axis = 1)
    print(f"Range of timestamps after preprocessing {numerical_model_data_source}: [{min(df.index)}, {max(df.index)}]")

    #
    # Save preprocessed data source file
    filename_and_extension = get_filename_and_extension(numerical_model_data_source)
    preprocessed_filename = filename_and_extension[0] + '_preprocessed' + filename_and_extension[1]
    df.to_parquet(filename_and_extension[0] + '_preprocessed.parquet.gzip', compression='gzip')

def preprocess_ws_datasource(station_id):
    ws_datasource = '../data/weather_stations/A652_1997_2022.csv'
    df = pd.read_csv(ws_datasource)

    #
    # Drop observations in which the target variable is not defined.
    n_obser_before_drop = len(df)
    df = df[df['CHUVA'].notna()]
    n_obser_after_drop = len(df)
    print(f"Number of observations before/after dropping null target entries: {n_obser_before_drop}/{n_obser_after_drop}.")

    #
    # Create U and V components of wind observations.
    df['wind_u'] = df.apply(lambda x: transform_wind(x.VEN_VEL, x.VEN_DIR, 0),axis=1)
    df['wind_v'] = df.apply(lambda x: transform_wind(x.VEN_VEL, x.VEN_DIR, 1),axis=1)

    ###############################
    # TODO: create temporal features
    ###############################

    #
    # Add index to dataframe using the timestamps.
    df.HR_MEDICAO = df.HR_MEDICAO.apply(format_time) # e.g., 1800 --> 18:00
    df['Datetime'] = pd.to_datetime(df.DT_MEDICAO + ' ' + df.HR_MEDICAO)
    df = df.set_index(pd.DatetimeIndex(df['Datetime']))
    print(f"Range of timestamps after preprocessing {ws_datasource}: [{min(df.index)}, {max(df.index)}]")

    #
    # Remove irrelevant columns
    df = df.drop(['HR_MEDICAO','DT_MEDICAO','Unnamed: 0', 'DC_NOME', 'CD_ESTACAO', 'UF', 'Datetime'], axis=1)

    #
    # Save preprocessed data source file
    filename_and_extension = get_filename_and_extension(ws_datasource)
    preprocessed_filename = filename_and_extension[0] + '_preprocessed' + filename_and_extension[1]
    df.to_parquet(filename_and_extension[0] + '_preprocessed.parquet.gzip', compression='gzip')

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
        elif opt in ("-s", "--station_id"):
            station_id = arg
            if not ((station_id in INMET_STATION_CODES_RJ) or (station_id in COR_STATION_NAMES_RJ)):
                print(f"Invalid station identifier: {station_id}")
                print(help_message)
                sys.exit(2)
        elif opt in ("-f", "--file"):
            ws_data = arg
        elif opt in ("-d", "--datasources"):
            if arg.find('R') != -1:
                sounding_data_source = '../data/sounding_stations/SBGL_indices_1997-01-01_2022-12-31.csv'
            if arg.find('N') != -1:
                numerical_model_data_source = '../data/numerical_models/ERA5_A652_1997-01-01_2021-12-31.csv'
        elif opt in ("-n", "--neighbors"):
            num_neighbors = arg

    print('Going to preprocess the specified data sources...')

    preprocess_ws_datasource(station_id)
    
    if sounding_data_source is not None:
        preprocess_sounding_data(sounding_data_source)

    if numerical_model_data_source is not None:
        preprocess_numerical_model_data(numerical_model_data_source)

    print('Done!')

if __name__ == "__main__":
    main(sys.argv)