"""Simple time representations used across the LanguageChange toolkit."""

from numbers import Number


class Time:
    def __init__(self):
        pass


class LiteralTime(Time):
    """Represents a literal timestamp or label for usage references."""

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
    """Numeric timestamp (e.g., time slice) that supports comparisons."""

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
    """Represents an interval between two Time points."""

    def __init__(self, start: Time, end: Time):
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
        
