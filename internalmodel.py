import numpy as np
from agent import State

class Internal_Model:
    """class to handle the internal model of the other player's actions
    """
    def __init__(self,nbr_of_actions):
        """ intialize the internal model

        Args:
            nbr_of_actions (int): the total number of available actions
        """
        self.model = {}
        self.nbr_of_actions = nbr_of_actions
        self.init_value = 1/nbr_of_actions

    def get_actual_theta(self,episode):
        """calculates the value of theta in function of the episode (see paper page 5 bottom left)

        Args:
            episode (int): the learning episode 

        Returns:
            float: the actual value of theta for the specific episode
        """
        return 0.2* pow(0.998849,episode)

    def get_dict_key(self,state,action):
        """creates the correct key to be use in the model dict

        Args:
            state (State): the state of the two hunters given by their relative positions
            action (int): the number of the action used by the opponent
        
        Returns:
            tuple: the tuple containing the key components
        """
        return(state.rel_position, state.other_rel_position, action)

    def get_state_action_estimation(self,state,action):
        """gets the estimation from the model given the state and the action of the other player

        Args:
            state (State): the state of the two hunters given by their relative positions
            action (int): the number of the action used by the opponent

        Returns:
            [type]: [description]
        """
        index = self.get_dict_key(state,action)
        if index in self.model:
            estimation = self.model.get(self.get_dict_key(state,action))
        else:
            estimation = self.init_value
            self.model.update({index:self.init_value})
        return estimation

    def update_state_action_estimation(self,state,actual_action,learning_episode=1):
        """ updates the estimation for the given state and action

        Args:
            state (State): the state of the two hunters given by their relative positions
            actual_action (int): the number of the action used by the opponent
            learning_episode (int, optional): the number of the learning episode. Defaults to 1.
        """
        for action in range(0,self.nbr_of_actions):
            index = self.get_dict_key(state,action)
            old_estimation = self.get_state_action_estimation(state,action)
            
            if(action == actual_action):
                factor = self.get_actual_theta(learning_episode)
            else:
                factor = 0
            
            new_estimation = (1- self.get_actual_theta(learning_episode))*old_estimation + factor
            self.model.update({index:new_estimation})
            
    def get_action_prob(self,state):
        """gets the probabilities as stored in the internal model for all possible actions given sthe state

        Args:
            state (State): the state of the two hunters given by their relative positions

        Returns:
            [numpy array]: an array with a probability for every possible action by the other player
        """
        prob = np.zeros(self.nbr_of_actions)
        for action in range(0,self.nbr_of_actions):
            prob[action] = self.get_state_action_estimation(state,action)
        return prob




class Internal_Model_Random(Internal_Model):
    """ class with a pseudo internal model. All probabilities will remain set at 0,2
    """
    
    def get_state_action_estimation(self,state,action):
        """[summary]

        Args:
            state (State): the state of the two hunters given by their relative positions
            action (int): the number of the action used by the opponent

        Returns:
            [int]: will always return the initial value as a probability
        """
        return self.init_value

    def update_state_action_estimation(self,state,actual_action,learning_episode=1):
        # this does nothing, all estimations will remain set to initial value
        dummy = 1



def test():
    state = State((-1,2),(2,2))
    state_2 = State((3,3),(2,2))
    int_model = Internal_Model(5)
    print(int_model.get_action_prob(state))

    for i in range(0,4):
        int_model.update_state_action_estimation(state,0,learning_episode=1)
        print(int_model.get_state_action_estimation(state,0))
        print(int_model.get_action_prob(state))
    
    for i in range(0,4):
        int_model.update_state_action_estimation(state,1,learning_episode=1)
        print(int_model.get_action_prob(state))

    int_model2 = Internal_Model_Random(5)
    print(int_model2.get_action_prob(state))
    int_model2.update_state_action_estimation(state_2,1,learning_episode=1)
    print(int_model2.get_action_prob(state_2))

if __name__ == "__main__":
    test()