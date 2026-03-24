from typing import List, Set

def expand_dictionary(words: List[str]):
    raise NotImplementedError

class SearchTerm():

    VALID_WORD_FEATURES = ['lemma', 'token', 'pos']

    def __init__(self, feature_value_pairs : dict | list | str, regex : bool = False):
        self.regex = regex
        if isinstance(feature_value_pairs, str):
            self.feature_value_pairs = {"token": feature_value_pairs}
        elif isinstance(feature_value_pairs, list):
            self.feature_value_pairs = {"token": "(" + "|".join(feature_value_pairs) + ")"}
            self.regex = True
        else:
            self.feature_value_pairs = feature_value_pairs
        for word_feature in self.feature_value_pairs.keys():
            if not word_feature in self.VALID_WORD_FEATURES:
                raise ValueError(f"key '{word_feature}' must be set to one of the following values: \
                    {self.VALID_WORD_FEATURES}")