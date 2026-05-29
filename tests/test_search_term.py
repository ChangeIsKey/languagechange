import pytest

from languagechange.search import SearchTerm


def test_search_term_default_feature():
    term = SearchTerm("hello")
    assert term.feature_value_pairs == {"token": "hello"}


def test_search_term_multiple_values():
    term = SearchTerm(["hello", "hi"])
    assert term.feature_value_pairs == {"token": "(hello|hi)"}
    assert term.regex


def test_search_term_multiple_features():
    term = SearchTerm(token="ducks", lemma="duck")
    assert term.feature_value_pairs == {"token": "ducks", "lemma": "duck"}