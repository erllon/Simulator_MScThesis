from beacons.MIN.min import MinState
from deployment.following_strategies.following_strategy import AtTargetException
from deployment.exploration_strategies.exploration_strategy import AtLandingConditionException
from numpy import zeros

class DeploymentFSM():

    def __init__(self, following_strategy, exploration_strategy):
        self.__fs = following_strategy
        self.__es = exploration_strategy

    def get_velocity_vector(self, MIN, beacons, SCS, ENV):
        if MIN.state == MinState.SPAWNED:
            self.__fs.prepare_following(MIN, beacons, SCS)
            MIN.state = MinState.FOLLOWING
        if MIN.state == MinState.FOLLOWING:
            try:
                return self.__fs.get_following_velocity(MIN, beacons, ENV)
            except AtTargetException:
                if MIN.prev == SCS:
                    target_pos_explore = MIN.prev.generate_target_pos(beacons, ENV, MIN)
                else:
                    target_pos_explore = MIN.prev.generate_target_pos(beacons, ENV, MIN.prev.prev, MIN)
                self.__es.prepare_exploration(target_pos_explore)
                MIN.state = MinState.EXPLORING
                print(f"{MIN.ID} exploring")
        if MIN.state == MinState.EXPLORING:
            try:
                return self.__es.get_exploration_velocity(MIN, beacons, ENV)
            except AtLandingConditionException:
                MIN.state = MinState.LANDED
                return zeros((2, ))
        else:
            print("MIN ALREADY LANDED")
            exit(0)
    
    def get_target(self):
        return self.__fs.target