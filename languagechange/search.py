from typing import List, Set, Union

def expand_dictionary(words: List[str]):
    raise NotImplementedError

class SearchTerm():

    def __init__(self, term : Union[str, list[str]]=None, regex : bool = False, **kwargs):
        self.regex = regex
        if term is not None:    
            if isinstance(term, str):
                self.feature_value_pairs = {"token": term}
            elif isinstance(term, list):
                self.feature_value_pairs = {"token": "(" + "|".join(term) + ")"}
                self.regex = True
        else:
            self.feature_value_pairs = kwargs

    def __str__(self):
        return "_".join(f"{k}={v}" for k, v in self.feature_value_pairs.items()) + ("_regex" if self.regex else "")

    def __repr__(self):
        return f"SearchTerm({self.feature_value_pairs}, regex={self.regex})"