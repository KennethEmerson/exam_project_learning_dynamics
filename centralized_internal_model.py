
from internalmodel import *

class Actions:
    def __init__(self, action_1, action_2):
        self.action_1 = action_1
        self.action_2 = action_2

class Centralized_Internal_Model(InternalModel): #diff is action is tpl, Q_bar = Q

        # def __init__() : same as super

        # def get_actual_theta() : same as super

        # reimplement to use Actions object as part of key
        def get_dict_key(self,state:State,actions:Actions):
            """creates the correct key to be use in the model dict

            Args:
                state (State): the state of the two hunters given by their relative positions
                action (Action): the object containing both actions

            Returns:
                tuple: the tuple containing the key components
            """
            return(state.rel_position, state.other_rel_position, actions.action_1, action.action_2)

        # def get_state_action_estimation() : same as super to be caled with Actions

        # no need to estimate other action : Q_bar = Q
        def update_state_action_estimation(self,state:State,actual_actions:Actions,learning_episode=1):
            pass

        # no need to estimate other action : Q_bar = Q
        def get_action_prob(self,state:State):
            pass
