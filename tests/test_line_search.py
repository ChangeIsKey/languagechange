import pytest

from languagechange.corpora import Line
from languagechange.search import SearchTerm


@pytest.fixture
def line_data():
    return Line(
        raw_text="alpha beta gamma",
        tokens=["alpha", "beta", "gamma"],
        lemmas=["alpha", "beta", "gamma"]
    )


def test_tokens_by_feature_valid(line_data):
    assert line_data.tokens_by_feature("token") == ["alpha", "beta", "gamma"]
    assert line_data.tokens_by_feature("lemma") == ["alpha", "beta", "gamma"]


def test_tokens_by_feature_invalid(line_data):
    with pytest.raises(ValueError):
        line_data.tokens_by_feature("invalid_feature")


def test_line_search_returns_correct_offsets(line_data):
    term = SearchTerm("beta", word_feature="token")
    usages = line_data.search(term)
    assert len(usages) == 1
    usage = usages[0]
    assert usage.start() == 6
    assert usage.end() == 10
    assert usage.text() == "alpha beta gamma"


def test_line_search_can_use_different_features(line_data):
    term = SearchTerm("gamma", word_feature="lemma")
    usages = line_data.search(term)
    assert len(usages) == 1
    usage = usages[0]
    assert usage.start() == 11
    assert usage.end() == 16
