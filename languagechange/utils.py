from numbers import Number

import numpy as np
import matplotlib as mpl

class Time:
    def __init__(self):
        pass


class LiteralTime(Time):
    def __init__(self, time: str):
        self.time = time

    def __eq__(self, other):
        assert type(other) == LiteralTime
        return self.time == other.time
    
    def __lt__(self, other):
        assert type(other) == LiteralTime
        return self.time < other.time
    
    def __le__(self, other):
        assert type(other) == LiteralTime
        return self.time <= other.time
    
    def __repr__(self):
        return self.time


class NumericalTime(Time):
    def __init__(self, time: Number):
        self.time = time

    def __eq__(self, other):
        if type(other) == NumericalTime:
            return self.time == other.time
        elif type(other) == TimeInterval:
            return self.time == other.start.time

    # todo: what if the other is a literal time?
    def __lt__(self, other):
        if type(other) == NumericalTime:
            return self.time < other.time
        elif type(other) == TimeInterval:
            return self.time < other.start.time
    
    def __le__(self, other):
        if type(other) == NumericalTime:
            return self.time <= other.time
        elif type(other) == TimeInterval:
            return self.time <= other.start.time
        
    def __repr__(self):
        return str(self.time)


class TimeInterval(Time):
    def __init__(self, start: Time, end:Time):
        self.start = start
        self.end = end
        if type(self.start).__name__ == type(self.end).__name__:
            if type(self.start) == NumericalTime:
                self.duration = self.end.time - self.start.time
        else:
            raise Exception('start and end points have to be of the same type')
        
    def __eq__(self, other):
        assert type(other) == TimeInterval
        return self.start == other.start and self.end == other.end
        
    # todo: what if the other is a literal time?
    def __lt__(self, other):
        if type(other) == TimeInterval:
            if self.start == other.start:
                return self.duration < other.duration
            else:
                return self.start < other.start
        elif type(other) == NumericalTime:
            return self.start.time < other.time
        
    def __le__(self, other):
        if type(other) == TimeInterval:
            if self.start == other.start:
                return self.duration <= other.duration
            else:
                return self.start <= other.start
        elif type(other) == NumericalTime:
            return self.start.time <= other.time
        
    def __repr__(self):
        return f"{self.start.time} - {self.end.time}"


def generate_colormap(n_classes):
    """
        Generate a colormap with 'n_classes' colors that are distinguishable from each other, along with grey, useful 
        for clustering.

        Args:
            n_classes (int): the amount of unique classes, excluding outliers (no class)

        Returns:
            cmap (matplotlib.colors.ListedColormap): a colormap mapping 0 to grey and c to a unique color, for all c in 
                range(1, n_classes+1).
    """
    hues = np.linspace(0, 1, n_classes, endpoint=False)
    saturations = np.full(n_classes, 1)
    values = np.tile(np.linspace(0.6, 1, 3), n_classes // 3 + 1)[:n_classes]
    hsv = np.stack([hues, saturations, values], axis=1)
    including_grey = np.vstack(([(0.7, 0.7, 0.7)], mpl.colors.hsv_to_rgb(hsv)))
    cmap = mpl.colors.ListedColormap(including_grey)
    return cmap