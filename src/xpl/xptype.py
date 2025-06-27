

from abc import ABC, abstractmethod

#
#====================================================================#
class Explanation(ABC):
    """
        Interface class to compute Explanation
    """

    def __init__(self, xpl):
        self.x = xpl

    @abstractmethod
    def dicho(self, hypos):
        pass

    @abstractmethod
    def linear(self, hypos):
        pass

    @abstractmethod
    def swift(self, hypos):
        pass

    # @abstractmethod
    # def disjunct(self, hypos, core, fts):
    #     pass