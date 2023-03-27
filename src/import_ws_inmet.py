import pandas as pd
import sys, getopt
from datetime import datetime
from util import is_posintstring

# API_TOKEN = "ejlJQjBhRWw0bUlUNlBnY0taRWFjWnFoSExSVFUwNW4=z9IB0aEl4mIT6PgcKZEacZqhHLRTU05n"
API_BASE_URL = "https://apitempo.inmet.gov.br"
STATION_CODES_FOR_RJ = ('A636', 'A621', 'A602', 'A652')

def import_from_station(station_code, initial_year, final_year, api_token):
    years = list(range(initial_year, final_year))
    df_inmet_stations = pd.read_json(API_BASE_URL + '/estacoes/T')
    station_row = df_inmet_stations[df_inmet_stations['CD_ESTACAO'] == station_code]
    df_observations_for_all_years = None
    print(f"Downloading observations from weather station {station_code}...")
    for year in years:
        print(f"Downloading observations for year {year}...")
        query_str = API_BASE_URL + '/token/estacao/' + str(year) + '-01-01/' + str(year) + '-12-31/' + station_code + "/" + api_token
        print(query_str)
        df_observations_for_a_year = pd.read_json(query_str)
        if df_observations_for_all_years is None:
            temp = [df_observations_for_a_year]
        else:
            temp = [df_observations_for_all_years, df_observations_for_a_year]
        df_observations_for_all_years = pd.concat(temp)
    filename = '../data/weather_stations/' + station_row['CD_ESTACAO'].iloc[0] + '_'+ str(initial_year) +'_'+ str(final_year) +'.csv'
    print(f"Done! Saving dowloaded content to '{filename}'.")
    df_observations_for_all_years.to_csv(filename)

def import_data(station_code, initial_year, final_year, api_token):
    if station_code == "all":
        df_inmet_stations = pd.read_json(API_BASE_URL + '/estacoes/T')
        station_row = df_inmet_stations[df_inmet_stations['CD_ESTACAO'].isin(STATION_CODES_FOR_RJ)]
        for j in list(range(0, len(station_row))):
            station_code = station_row['CD_ESTACAO'].iloc[j]
            import_from_station(station_code, initial_year, final_year, api_token)
    else:
        import_from_station(station_code, initial_year, final_year, api_token)

def main(argv):
    '''
        python import_ws_inmet.py -s A652 -b 2020 -e 2022 --api_token <token>
    '''
    station_code = ""

    start_year = 1997
    end_year = datetime.now().year

    help_message = "{0} -s <station> -b <begin> -e <end> -t <api_token>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hs:b:e:t:", ["help", "station=", "begin=", "end=", "api_token="])
    except:
        print(help_message)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_message)
            sys.exit(2)
        elif opt in ("-t", "--api_token"):
            api_token = arg
        elif opt in ("-s", "--station"):
            station_code = arg
            if not ((station_code == "all") or (station_code in STATION_CODES_FOR_RJ)):
                print(help_message)
                sys.exit(2)
        elif opt in ("-b", "--begin"):
            if not is_posintstring(arg):
                sys.exit("Argument start_year must be an integer. Exit.")
            start_year = int(arg)
        elif opt in ("-e", "--end"):
            if not is_posintstring(arg):
                sys.exit("Argument end_year must be an integer. Exit.")
            end_year = int(arg)

    assert (api_token is not None) and (api_token != '')
    assert (station_code is not None) and (station_code != '')
    assert (start_year <= end_year)

    import_data(station_code, start_year, end_year, api_token)


if __name__ == "__main__":
    main(sys.argv)


