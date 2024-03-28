from collections import defaultdict


def merge_dictionaries(dict1, dict2):
    """
    Merges two dictionaries into a single dictionary, combining the values of 
    any duplicate keys into lists.

    Parameters:
    - dict1 (dict): The first input dictionary.
    - dict2 (dict): The second input dictionary.

    Returns:
    - defaultdict(list): A dictionary where each key holds a list of values 
    from both input dictionaries. If a key is present in both dict1 and dict2, 
    both values will be in the list. Otherwise, the list will contain the 
    single value from whichever dictionary the key originates from.
    """
    merged_dict = defaultdict(list)

    for d in (dict1, dict2): # Extendable for more dictionaries
        for key, value in d.items():
            merged_dict[key].append(value)
    
    return merged_dict


def get_duplicate_values(dictionary):
    """
    Identifies and returns a list of duplicate values in the given dictionary. If a value appears
    more than once across all values in the dictionary, it will be included in the return list.

    Parameters:
    - dictionary (dict): The dictionary whose values are to be checked for duplicates.

    Returns:
    - list: A list of values that appear more than once in the dictionary.

    Note: 
    This implementation is designed to work effectively with immutable value types (e.g.,
    integers, strings, tuples). For mutable types like lists, it treats the value as seen only once
    because lists cannot be used as dictionary keys or added to sets directly without conversion.
    """
    seen_values = []
    duplicate_values = []

    for value in dictionary.values():
        if value in seen_values:
            duplicate_values.append(value)
        else:
            seen_values.append(value)

    return duplicate_values