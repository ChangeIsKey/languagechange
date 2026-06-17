from pathlib import Path

import pytest

from languagechange.corpora import SprakBankenCorpus, ParquetCorpus
from languagechange.search import SearchTerm


@pytest.fixture(scope="module")
def xml_corpus():
    return SprakBankenCorpus("svt-2004.xml.bz2")


@pytest.fixture(scope="module")
def parquet_corpus(xml_corpus):
    xml_corpus.cast_to_parquet()

    parquet_path = Path("svt-2004.parquet")
    assert parquet_path.exists(), f"Expected parquet file was not created: {parquet_path}"

    return ParquetCorpus(str(parquet_path))


def test_xml_and_parquet_search_results_match(xml_corpus, parquet_corpus):
    search_terms = [SearchTerm(lemma="kvinna"), SearchTerm(lemma="man")]

    xml_results = xml_corpus.search(search_terms)
    parquet_results = parquet_corpus.search(search_terms)

    assert xml_results.keys() == parquet_results.keys()

    for term in xml_results.keys():
        for u1, u2 in zip(xml_results[term], parquet_results[term]):
            assert u1.text() == u2.text()
            assert u1.offsets == u2.offsets
            assert u1.time == u2.time

