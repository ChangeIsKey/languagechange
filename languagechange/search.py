from typing import List, Set

def expand_dictionary(words: List[str]):
    raise NotImplementedError

class SearchTerm():

    def __init__(self, feature_value_pairs : dict | list | str, regex : bool = False):
        self.regex = regex
        if isinstance(feature_value_pairs, str):
            self.feature_value_pairs = {"token": feature_value_pairs}
        elif isinstance(feature_value_pairs, list):
            self.feature_value_pairs = {"token": "(" + "|".join(feature_value_pairs) + ")"}
            self.regex = True
        else:
            self.feature_value_pairs = feature_value_pairs