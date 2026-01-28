from abc import ABC


class MeaningModel(ABC):

    def __init__(self):
        pass


class WordSenseInduction(MeaningModel):
    pass


class StaticEmbedding(ABC):
    """Placeholder base for static embedding types."""
    pass


# todo
class SGNS(StaticEmbedding):

    def __init__(self):
        self.align_strategies = {"OP", "SRV", "WI"}
        pass
