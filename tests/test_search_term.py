import pytest

from languagechange.search import SearchTerm


def test_search_term_default_feature():
    term = SearchTerm("hello")
    assert term.feature_value_pairs == {"token": "hello"}
