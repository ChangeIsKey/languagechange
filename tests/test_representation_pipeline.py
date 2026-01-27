from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from languagechange.corpora import LinebyLineCorpus
from languagechange.models.representation import static as static_module
from languagechange.models.representation.static import CountModel, PPMI, SVD


class FakeSpace:
    saved_spaces = {}
    last_loaded = {}

    def __init__(self, matrix=None, rows=None, columns=None, format=None):
        if isinstance(matrix, str):
            saved = FakeSpace.saved_spaces.get(matrix)
            if saved is None:
                raise FileNotFoundError(matrix)
            self.matrix = saved.matrix
            self.rows = saved.rows
            self.columns = saved.columns
            self.id2row = saved.id2row
            self.row2id = saved.row2id
            self.id2column = saved.id2column
            self.column2id = saved.column2id
            self.operations = list(saved.operations)
            self.format = format or saved.format
            FakeSpace.last_loaded[matrix] = self
        else:
            self.matrix = matrix
            self.rows = list(rows or [])
            self.columns = list(columns or [])
            self.id2row = {i: token for i, token in enumerate(self.rows)}
            self.row2id = {token: i for i, token in enumerate(self.rows)}
            self.id2column = {i: column for i, column in enumerate(self.columns)}
            self.column2id = {column: i for i, column in enumerate(self.columns)}
            self.operations = []
            self.format = format

    def save(self, path, format=None):
        saved = FakeSpace(matrix=self.matrix, rows=self.rows, columns=self.columns, format=format or self.format)
        saved.operations = list(self.operations)
        saved.row2id = dict(self.row2id)
        saved.id2row = dict(self.id2row)
        saved.id2column = dict(self.id2column)
        saved.column2id = dict(self.column2id)
        FakeSpace.saved_spaces[path] = saved

    def epmi_weighting(self, smoothing_parameter):
        self.operations.append(("epmi_weighting", smoothing_parameter))

    def log_weighting(self):
        self.operations.append("log_weighting")

    def shifting(self, value):
        self.operations.append(("shifting", value))

    def eliminate_negative(self):
        self.operations.append("eliminate_negative")

    def eliminate_zeros(self):
        self.operations.append("eliminate_zeros")

    def l2_normalize(self):
        self.operations.append("l2_normalize")


class FakePathLineSentences:
    def __init__(self, path):
        paths = [path] if isinstance(path, (str, Path)) else list(path)
        self.sentences = []
        for path_item in paths:
            current = Path(path_item)
            if current.is_dir():
                files = sorted(current.glob("*.txt"))
            else:
                files = [current]
            for file in files:
                with file.open(encoding="utf-8") as fh:
                    for line in fh:
                        tokens = [token for token in line.strip().split() if token]
                        if tokens:
                            self.sentences.append(tokens)

    def __iter__(self):
        return iter(self.sentences)


def fake_randomized_svd(matrix, n_components, n_iter, transpose):
    rows, cols = matrix.shape
    u = np.tile(np.arange(1, n_components + 1, dtype=float), (rows, 1))
    s = np.arange(1, n_components + 1, dtype=float)
    v = np.zeros((n_components, cols), dtype=float)
    return u, s, v


@pytest.fixture(autouse=True)
def reset_fake_space():
    FakeSpace.saved_spaces.clear()
    FakeSpace.last_loaded.clear()


def test_count_ppmi_svd_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(static_module, "Space", FakeSpace)
    monkeypatch.setattr(static_module, "PathLineSentences", FakePathLineSentences)
    monkeypatch.setattr(static_module, "randomized_svd", fake_randomized_svd)

    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text("alpha beta gamma\nbeta alpha gamma\n")
    corpus = LinebyLineCorpus(str(corpus_file), name="demo", language="EN", is_tokenized=True)

    count_path = tmp_path / "count"
    count_model = CountModel(corpus=corpus, window_size=1, savepath=str(count_path))
    count_model.encode()

    assert count_model.savepath in FakeSpace.saved_spaces
    assert len(FakeSpace.saved_spaces[count_model.savepath].rows) >= 2

    ppmi_path = tmp_path / "ppmi"
    ppmi_model = PPMI(count_model, shifting_parameter=1, smoothing_parameter=1, savepath=str(ppmi_path))
    ppmi_model.encode()

    recorded_space = FakeSpace.last_loaded[count_model.matrix_path]
    assert ("epmi_weighting", 1) in recorded_space.operations
    assert "eliminate_zeros" in recorded_space.operations
    assert ppmi_model.savepath in FakeSpace.saved_spaces

    svd_path = tmp_path / "svd"
    svd_model = SVD(ppmi_model, dimensionality=2, gamma=1.0, savepath=str(svd_path))
    svd_model.encode()
    svd_model.load()

    assert svd_model.space is not None
    saved_svd = FakeSpace.saved_spaces[svd_model.savepath]
    assert saved_svd.format == "w2v"
