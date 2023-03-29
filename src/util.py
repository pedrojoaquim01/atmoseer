import pytz
from datetime import datetime
import os
from metpy.calc import wind_components
from metpy.units import units

def is_posintstring(s):
    try:
        temp = int(s)
        if temp > 0:
            return True
        else:
            return False
    except ValueError:
        return False

def format_time(input_str):
    '''
    This function first converts the input string to an integer using the int function. 
    It then extracts the hours and minutes from the input integer using integer division and modulus operations. 
    Finally, it formats the output string using f-strings to ensure that both hours and minutes are represented with two digits.
    Usage examples:
        print(format_time("100"))   # Output: "01:00"
        print(format_time("1200"))  # Output: "12:00"
        print(format_time("2300"))  # Output: "23:00"
    '''
    # Convert input string to integer
    input_int = int(input_str)
    
    # Extract hours and minutes from the input integer
    hours = input_int // 100
    minutes = input_int % 100
    
    # Format the output string
    output_str = f"{hours:02}:{minutes:02}"
    
    return output_str

def utc_to_local(utc_string, local_tz, format_string):
    '''
    This function first converts the UTC string to a datetime object with timezone information 
    using strptime() and replace(). Then it converts the datetime object to the local timezone 
    using astimezone(). Finally, it formats the resulting datetime as a string in the same
    format as the input string.

    Here's an example usage of the function:
    
        utc_string = '1972-09-13 12:00:00'
        local_tz = 'America/Sao_Paulo'
        format_string = '%Y-%m-%d %H:%M'
        local_string = utc_to_local(utc_string, local_tz, format_string)
        print(local_string)

    '''
    # convert utc string to datetime object
    utc_dt = datetime.strptime(utc_string, format_string).replace(tzinfo=pytz.UTC)
    
    # convert to local timezone
    local_tz = pytz.timezone(local_tz)
    local_dt = utc_dt.astimezone(local_tz)
    
    # format as string and return
    return local_dt.strftime(format_string)

def transform_wind(wind_speed, wind_direction, comp_idx):
    """
    This function calculates either the U or V wind vector component from the speed and direction.
    comp_idx = 0 --> computes the U component
    comp_idx = 1 --> computes the V component
    """
    assert(comp_idx == 0 or comp_idx == 1)
    return wind_components(wind_speed * units('m/s'), wind_direction * units.deg)[comp_idx].magnitude

def transform_hour(df):
    """
    Transforms a DataFrame's datetime index into two new columns representing the hour in sin and cosine form.

    Args:
    - df: A pandas DataFrame with a datetime index.

    Returns:
    - The input pandas DataFrame with two new columns named 'hour_sin' and 'hour_cos' representing the hour in sin and cosine form.
    """
    dt = df.index
    hourfloat = dt.hour + dt.minute/60.0
    df['hour_sin'] = np.sin(2. * np.pi * hourfloat/24.)
    df['hour_cos'] = np.cos(2. * np.pi * hourfloat/24.)
    return df

def get_filename_and_extension(filename):
    """
    Given a filename, returns a tuple with the base filename and extension.
    """
    basename = os.path.basename(filename)
    filename_parts = os.path.splitext(basename)
    return filename_parts
