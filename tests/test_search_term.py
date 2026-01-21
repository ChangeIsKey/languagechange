import pytest

from languagechange.search import SearchTerm


def test_search_term_default_feature():
    term = SearchTerm("hello")
    assert term.word_feature == {"token"}


def test_search_term_invalid_feature():
    with pytest.raises(ValueError):
        SearchTerm("hello", word_feature="invalid_feature")
