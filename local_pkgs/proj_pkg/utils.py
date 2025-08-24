import re
import pandas as pd
import colorsys
import warnings 
from matplotlib.colors import Normalize
import math

# Function to calculate text color based on background luminance
def get_text_color(value, colormap, vmin, vmax):
    # Normalize the value within the colorbar range
    norm = Normalize(vmin=vmin, vmax=vmax)
    normalized_value = norm(value)
    
    # Get RGB color from colormap
    rgba = colormap(normalized_value)
    r, g, b, _ = rgba  # Ignore alpha value
    
    # Calculate luminance
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Return black or white based on luminance
    return "white" if luminance < 0.5 else "black"

def convert_percentage_to_fraction(value):
    """
    Converts a percentage string, regex match with percentage, or decimal fraction (including as a string) into a decimal form.
    
    Parameters
    ----------
    value : str, int, float, or re.Match
        The value to convert. Accepts:
        - A string with a percentage sign (e.g., '5%')
        - A regex match object containing a captured group with a percentage (e.g., from r"(\\d+)%")
        - A numeric decimal between 0 and 1, as a float or a string (e.g., 0.05 or '0.05')
    
    Returns
    -------
    float
        The value converted to a decimal fraction. For example, '5%' or match object with '50%' becomes 0.05, and '0.05' remains 0.05.
    
    Raises
    ------
    ValueError
        If the input is not a valid percentage string, regex match, or decimal fraction between 0 and 1.
    
    Examples
    --------
    >>> convert_percentage_to_fraction('5%')
    0.05
    
    >>> convert_percentage_to_fraction(0.05)
    0.05
    
    >>> convert_percentage_to_fraction('0.05')
    0.05
    
    >>> import re
    >>> pattern = re.compile(r"(\\d+)%")
    >>> match = pattern.search("50%")
    >>> convert_percentage_to_fraction(match)
    0.5
    
    >>> convert_percentage_to_fraction('0.5%')
    0.005
    
    >>> convert_percentage_to_fraction('1')
    1.0
    """
    # Handle regex match object with captured percentage
    if isinstance(value, re.Match):
        number = float(value.group(1))
        return number / 100
    
    # Handle percentage string with '%'
    elif isinstance(value, str) and value.endswith('%'):
        return float(value.strip('%')) / 100
    
    # Handle fraction as a string or a numeric value
    elif isinstance(value, (str, float, int)):
        try:
            fraction = float(value)
            if 0 < fraction <= 1:
                return fraction
        except ValueError:
            raise ValueError("Input must be a percentage string (e.g., '5%'), regex match object with a percentage, or a decimal fraction (e.g., 0.05)")

    # Raise an error for invalid input types or ranges
    raise ValueError("Input must be a percentage string (e.g., '5%'), regex match object with a percentage, or a decimal fraction (e.g., 0.05)")


def composition_to_string(composition, decimal_places=8):
    # Create a dictionary with elements and their amounts
    element_amounts = composition.get_el_amt_dict()
    # Convert each amount to a string with regular notation
    format_string = f"{{:.{decimal_places}f}}"
    converted = {
        el: format_string.format(amt).rstrip('0').rstrip('.') if amt < 1 or amt != int(amt) else str(int(amt))
        for el, amt in element_amounts.items()
    }
    # Join the elements with their amounts as a string
    return ''.join([f"{el}{amt}" for el, amt in converted.items()])

def load_dataframe(input_data):
    """
    Checks whether the input is a DataFrame or a string representing a file path.
    If it's a string, loads it as a DataFrame. Otherwise, returns the DataFrame as is.    
    
    Parameters
    ----------
    input_data : str or pd.DataFrame
        File path to the CSV file or a DataFrame.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
        
    Raises
    ------
    ValueError
        If the input is not a valid file path or a DataFrame.
        
    TypeError
        If the input is not a string or a DataFrame.
    """
    
    if isinstance(input_data, pd.DataFrame):
        return input_data
    elif isinstance(input_data, str):
        try:
            df = pd.read_csv(input_data)
            return df
        except Exception as e:
            raise ValueError(f"Failed to load data from {input_data}. Error: {e}")
    else:
        raise TypeError("input_data must be a pandas DataFrame or a string representing a file path.")

def adjust_hsv(rgb_color, h_factor=1.0, s_factor=1.0, v_factor=1.0):
    '''
    Adjust the HSV values of an RGB color.
    
    Parameters
    ----------
    rgb_color : tuple
        A tuple of RGB values.
        
    h_factor : float, optional
        Factor to adjust the hue value by.
        
    s_factor : float, optional
        Factor to adjust the saturation value by.
        
    v_factor : float, optional
        Factor to adjust the value (brightness) value by.

    Returns
    -------
    tuple
        A tuple of RGB values after adjusting the HSV values.
    '''
    
    if h_factor == 1.0 and s_factor == 1.0 and v_factor == 1.0:
        warnings.warn("No adjustment made to the RGB color.")
    
    r, g, b = rgb_color
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = min(max(h * h_factor, 0), 1)
    s = min(max(s * s_factor, 0), 1)
    v = min(max(v * v_factor, 0), 1)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b)

def equalsIgnoreCase(str1, str2):
    '''
    Check if two strings are equal ignoring case
    
    Parameters
    ----------
    str1 : str
        The first string to compare.

    str2 : str
        The second string to compare.

    Returns
    -------
    bool
        True if the strings are equal ignoring case, False otherwise.

    '''
    if str1 is None or str2 is None:
        return False
    
    return str1.lower() == str2.lower()

def equalsIgnoreCaseAndSpace(str1, str2):
    """
    Compares two strings for equality, ignoring case and spaces.
    
    Parameters
    ----------
    str1 : str
        The first string to compare.
        
    str2 : str
        The second string to compare.
        
    Returns
    -------
    bool
        True if the strings are equal ignoring case and spaces, False otherwise.
    """
    
    if str1 is None or str2 is None:
        return False
    
    # Remove all spaces and convert to lower case
    str1_cleaned = ''.join(str1.split()).lower()
    str2_cleaned = ''.join(str2.split()).lower()
    
    return str1_cleaned == str2_cleaned


def everyday_round(number):
    return math.floor(number + 0.5) if number > 0 else math.ceil(number - 0.5)