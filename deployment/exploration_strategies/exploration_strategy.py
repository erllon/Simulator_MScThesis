from abc import ABC, abstractmethod
import numpy as np

class AtLandingConditionException(Exception):
    """
    Raised when the exploring MIN fulfills its landing condition
    """

class ExplorationStrategy(ABC):

    MAX_EXPLORATION_SPEED = 2
    MIN_RSSI_STRENGTH_BEFORE_LAND = 0.22

    def prepare_exploration(self, target):
        self.target = target
        
    @abstractmethod
    def get_exploration_velocity(self, MIN, beacons, ENV):
        pass