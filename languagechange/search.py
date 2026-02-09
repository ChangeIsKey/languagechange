"""Helper utilities for searching corpora for target terms."""

from typing import List, Set


def expand_dictionary(words: List[str]):
    """Placeholder for future dictionary expansion utilities.

    Args:
        words (List[str]): Words to expand into additional search terms.
    """
    raise NotImplementedError


class SearchTerm:
    """Describes a search target and the features to scan within a corpus line."""

    VALID_WORD_FEATURES = ['lemma', 'token', 'pos']

    def __init__(self, term: str, regex: bool = False, word_feature: str | Set = 'token'):
        """Initialise a search term for corpus queries.

        Args:
            term (str): The string pattern to look for.
            regex (bool, optional): Whether to treat the term as a regular expression. Defaults to False.
            word_feature (str|Set, optional): Features to consider ('token', 'lemma', 'pos'). Defaults to 'token'.
        """
        self.term = term
        self.regex = regex
        self.word_feature = word_feature if isinstance(word_feature, Set) else {word_feature}
        if not self.word_feature.issubset(self.VALID_WORD_FEATURES):
            raise ValueError("'word_feature' must be set to one of the following values:", self.VALID_WORD_FEATURES)
