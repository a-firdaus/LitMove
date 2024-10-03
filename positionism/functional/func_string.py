import numpy as np


def replace(i):
    """
    Replace a string with NaN if it cannot be converted to a float.

    Source: https://stackoverflow.com/questions/57048617/how-do-i-replace-all-string-values-with-nan-dynamically
    
    Args
    ====
    i: str
        The input string.

    Returns
    =======
    i : float or np.nan
        The converted float or NaN.

    """
    try:
        float(i)
        return float(i)
    except:
        return np.nan


def modify_line(line, old_part, new_label):
    """
    Replace a specific part of a line with a new label.
    
    Args
    ====
    line: str
        The original line of text.
    old_part: str
        The part of the line to be replaced.
    new_label: str
        The new label to insert in place of the old part.
    
    Returns
    =======
    str
        The modified line with the new label.
    """
    return line.replace(old_part, new_label)


def replace_values_in_series(series, replacements):
    """
    Replace values in a Pandas Series according to a replacements dictionary.
    
    Args
    ====
    series: pd.Series
        The Pandas Series to modify.
    replacements: dict
        A dictionary where keys are old values and values are new values.
    
    Returns
    =======
    pd.Series
        The modified Series with replaced values.
    """
    return series.replace(replacements)


def is_number(s):
    """
    Function to check if a string is a number
    """
    try:
        int(s)
        return True
    except ValueError:
        return False