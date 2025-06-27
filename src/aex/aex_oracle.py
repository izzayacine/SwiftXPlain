#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   oracle interface for finding adversarial examples
#
################################################################################
import sys
import os
import time
from abc import ABC, abstractmethod
import numpy as np
################################################################################


class AExOracle(ABC):
    # oracle interface for finding adversarial examples

    def __init__(self):
        """
            Initialize the oracle.
        """
        # neural network
        self.neural_network = None
        # input data point, the output label of this data point,
        # and perturbation bound
        self.in_values = None
        self.out_label = None
        self.epsilon = None

        # initialize the oracle, including neural networks and the robustness tool,
        # e.g. load the neural network, encode the neural network, etc.
        # ...

    @abstractmethod
    def delete(self):
        """
            Delete the neural network and the robustness tool.
            e.g. free the resources, etc.
        """
        pass

    @abstractmethod
    def encode_aex(self, instance, epsilon):
        """
            Do whatever need to be done before checking if there exist adversarial examples.
            e.g. encode the input instance, and the perturbation bound, etc.
            :param instance: input instance which is a pair of (data point, label)
            :param epsilon: perturbation bound
        """
        pass

    @abstractmethod
    def has_aex(self, fixed=[], timeout=60, verbose=0) -> bool:
        """
            Check if there exist adversarial examples.
            :param fixed: 1d np.array feature/pixel indices to fix.
            :param timeout: timeout for the robustness tool
            :return: True if there exist adversarial examples, False otherwise
        """
        pass
    
