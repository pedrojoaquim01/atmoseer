import pandas as pd
import sys, getopt
from datetime import datetime, timedelta
from siphon.simplewebservice.wyoming import WyomingUpperAir
from util import is_posintstring
import time
import requests

def get_data(station_name, start_date, end_date):

    print(f"Downloading observations from radiosonde {station_name}...")
    df_all_probes = pd.DataFrame()

    end_date = min([end_date, datetime.today()])

    unsuccesfully_downloaded_probes = 0

    next_date = start_date
    while next_date <= end_date:
        try:
            first_probe = next_date + timedelta(hours=0)
            df_probe = WyomingUpperAir.request_data(first_probe, station_name)
            print(f"Data successfully downloaded for {first_probe} ({len(df_probe)} lines).")
            df_all_probes = pd.concat(([df_all_probes, df_probe]))
        except IndexError as e:
            print(f'{repr(e)}')
            unsuccesfully_downloaded_probes += 1
        except ValueError as e:
            print(f'{str(e)}')
            unsuccesfully_downloaded_probes += 1
        except requests.HTTPError as e:
            print(f'{repr(e)}')
            print("Server seems to be busy. Going to sleep for a while...")
            time.sleep(10) # Sleep for a moment
            print("Back to work!")
            continue
        except Exception as e:
            print(f'Unexpected error! {repr(e)}')
            sys.exit(2)

        # Now, go to next day
        next_date = next_date + timedelta(days=1)

    next_date = start_date
    while next_date <= end_date:
        try:
            second_probe = next_date + timedelta(hours=12)
            df_probe = WyomingUpperAir.request_data(second_probe, station_name)
            print(f"Data successfully downloaded for {second_probe} ({len(df_probe)} lines).")
            df_all_probes = pd.concat(([df_all_probes, df_probe]))
        except IndexError as e:
            print(f'{repr(e)}')
            unsuccesfully_downloaded_probes += 1
        except ValueError as e:
            print(f'{str(e)}')
            unsuccesfully_downloaded_probes += 1
        except requests.HTTPError as e:
            # end_date = next_date
            print(f'{repr(e)}')
            print("Server seems to be busy. Going to sleep for a while...")
            time.sleep(10) # Sleep for a moment
            print("Back to work!")
            continue
            # break
        except Exception as e:
            print(f'Unexpected error! {repr(e)}')
            sys.exit(2)

        # Calculate next date
        next_date = next_date + timedelta(days=1)

    print(f"Done! Number of unsuccesfully downloaded probes: {unsuccesfully_downloaded_probes}.")
    filename = '../data/' + station_name + '_'+ start_date.strftime('%Y-%m-%d') + '_' + end_date.strftime('%Y-%m-%d') + '.csv'
    print(f"Saving dowloaded content to file {filename}.")
    df_all_probes.to_csv(filename, index = False)

def main(argv):
    help_message = "Usage: {0} -s <station_name> -b <start_year> -e <end_year>".format(argv[0])

    station_name = 'SBGL'

    date_format_str = '%Y-%m-%d'
    
    try:
        opts, args = getopt.getopt(argv[1:], "hs:b:e:", ["help","station_name=","start_year=","end_year="])
    except:
        print("Invalid arguments. Use -h or --help for more information.")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_message)
            sys.exit(2)
        elif opt in ("-s", "--station_name"):
            station_name = arg
        elif opt in ("-b", "--start_date"):
            try:
                start_date = datetime.strptime(arg, date_format_str)
            except ValueError:
                print("Invalid date format. Use -h or --help for more information.")
                sys.exit(2)
        elif opt in ("-e", "--end_date"):
            try:
                end_date = datetime.strptime(arg, date_format_str)
            except ValueError:
                print("Invalid date format. Use -h or --help for more information.")
                sys.exit(2)

    assert (station_name is not None) and (station_name != '')
    assert (start_date <= end_date)

    get_data(station_name, start_date, end_date)


if __name__ == "__main__":
    main(sys.argv)
