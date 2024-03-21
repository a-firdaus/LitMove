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


class Mapping:
    def get_duplicate_closest24_w_data(dict):
        """
        Identifies and returns a list of duplicate values (closest24 with its data: coor)
        """
        duplicate_closest24 = {}
        for coorreference, values in dict.items():
            for entry in values:
                closest24 = entry["closest24"]
                dist = entry["dist"]

            if closest24 in duplicate_closest24:
                duplicate_closest24[closest24].append({"coorreference": coorreference, "dist": dist})
            else:
                duplicate_closest24[closest24] = [{"coorreference": coorreference, "dist": dist}]

        duplicate_closest24_w_data = {}
        for closest24, coorreferences_dists in duplicate_closest24.items():
            if len(coorreferences_dists) > 1:
                duplicate_closest24_w_data[f"Duplicate closest24: {closest24}"] = [{"coorreferences and dists": coorreferences_dists}]

        return duplicate_closest24_w_data


    def get_atom_mapping_el_w_dist_closestduplicate(dict):
        """
        Identifies and returns closest mapped atom with its distance
        """
        filtered_data = {}
        for coorreference, values in dict.items():
            for entry in values:
                closest24 = entry["closest24"]
                dist = entry["dist"]
                
            if closest24 in filtered_data:
                if dist < filtered_data[closest24]["dist"]:
                    filtered_data[closest24] = {"coorreference": coorreference, "dist": dist}
            else:
                filtered_data[closest24] = {"coorreference": coorreference, "dist": dist}

        atom_mapping_el_w_dist_closestduplicate = {entry["coorreference"]: {"closest24": key, "dist": entry["dist"]} for key, entry in filtered_data.items()}
        return atom_mapping_el_w_dist_closestduplicate
