import os
from pathlib import Path

import pytest

from languagechange.benchmark import SemEval2020Task1
from languagechange.resource_manager import LanguageChange


def _write_corpus(file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("alpha beta gamma\nbeta alpha gamma\n")


def _write_truth_file(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows))


@pytest.fixture
def semeval_structure(tmp_path):
    resource_root = tmp_path / "benchmarks" / "SemEval 2020 Task 1" / "EN" / "no-version" / "sample"
    corpus_paths = [
        ("corpus1", "lemma"),
        ("corpus1", "token"),
        ("corpus2", "lemma"),
        ("corpus2", "token"),
    ]
    for corpus_root, variant in corpus_paths:
        corpus_dir = resource_root / corpus_root / variant
        _write_corpus(corpus_dir / "data.txt")
    truth_dir = resource_root / "truth"
    _write_truth_file(truth_dir / "binary.txt", ["change 1", "stable 0"])
    _write_truth_file(truth_dir / "graded.txt", ["change 0.73", "stable 0.01"])
    return resource_root


def test_sem_eval_benchmark_loads_corpora(monkeypatch, semeval_structure):
    def fake_get_resource(self, resource_type, resource_name, dataset, version):
        assert resource_type == "benchmarks"
        assert resource_name == "SemEval 2020 Task 1"
        assert dataset == "EN"
        assert version == "no-version"
        return str(semeval_structure.parent)

    monkeypatch.setattr(LanguageChange, "get_resource", fake_get_resource)

    benchmark = SemEval2020Task1("EN")

    assert benchmark.corpus1_lemma.path.endswith(os.path.join("corpus1", "lemma"))
    assert benchmark.corpus2_token.path.endswith(os.path.join("corpus2", "token"))
    assert len(benchmark.binary_task) == 2
    assert any(target.target == "change" for target in benchmark.binary_task)
    assert any(target.target == "stable" for target in benchmark.graded_task)
