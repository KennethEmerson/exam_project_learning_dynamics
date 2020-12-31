import numpy as np
from agent import State
from move import *


class InternalModel:
    """
    Class to handle the internal model of the other player's actions.
    """

    def __init__(self, initial_theta: float):
        """
        Initialize the internal model.

        :param initial_theta: The initial theta value.
        """
        self.model = {}
        self.init_value = 1 / NB_MOVES  # 0.2 for five possible moves
        self.initial_theta = initial_theta

    def get_actual_theta(self, episode: int) -> float:
        """
        Calculate the value of theta in function of the
        episode (see paper page 5, bottom left).

        :param episode: the learning episode

        :return: The actual value of theta for the specific episode
        """
        return 0.2 * (self.initial_theta ** episode)

    def get_dict_key(self, state: State, action: int) -> ((int, int), (int, int), int):
        """
        Create the correct key to be use in the model dictionary.

        :param state: The state of the two hunters given by their
             relative positions
        :param action: The number of the action used by the opponent.

        :return: The tuple containing the key components.
        """
        return state.rel_position, state.other_rel_position, action

    def get_state_action_estimation(self, state: State, action: int) -> float:
        """
        Get the estimation from the model given the state and the action
        of the other player.

        :param state: The state of the two hunters given by their relative
            positions.
        :param action: The number of the action used by the opponent.

        :return: The action estimation.
        """
        index = self.get_dict_key(state, action)

        if index in self.model:
            estimation = self.model.get(self.get_dict_key(state, action))
        else:
            estimation = self.init_value
            self.model.update({index: self.init_value})

        return estimation

    def update_state_action_estimation(self, state: State, actual_action: int, learning_episode=1):
        """
        Update the estimation for the given state and action.

        :param state: The state of the two hunters given by their
            relative positions.
        :param actual_action: The number of the action used by the
            opponent.
        :param learning_episode: The number of the learning episode.
            Set to 1 by default.
        """
        for action in range(NB_MOVES):
            index = self.get_dict_key(state, action)
            old_estimation = self.get_state_action_estimation(state, action)

            if action == actual_action:
                factor = self.get_actual_theta(learning_episode)
            else:
                factor = 0

            new_estimation = (1 - self.get_actual_theta(learning_episode)) * old_estimation + factor
            self.model.update({index: new_estimation})

    def get_action_prob(self, state: State) -> [float]:
        """
        Get the probabilities (as stored in the internal model) for
        all possible actions in given the state.

        :param state: The state of the two hunters given by their relative
            positions.

        :return: An array with the probability (based on the internal model),
            for each action, that the other player chooses that action.
        """

        return [self.get_state_action_estimation(state, action) for action in range(NB_MOVES)]


class InternalModelRandom(InternalModel):
    """
    Class with a pseudo internal model. All probabilities will remain
    set at 0.2.
    """

    def get_state_action_estimation(self, state, action):
        """
        Get the estimation from the model given the state and the action
        of the other player.

        :param state: The state of the two hunters given by their relative
            positions.
        :param action: The number of the action used by the opponent.

        :return: The action estimation.
        """
        return self.init_value

    def update_state_action_estimation(self, state, actual_action, learning_episode=1):
        """
        Ignored for this class.
        """
        pass


def test():
    state = State((-1, 2), (2, 2))
    state_2 = State((3, 3), (2, 2))
    int_model = InternalModel(5, 0.998849)
    print(int_model.get_action_prob(state))

    for i in range(0, 4):
        int_model.update_state_action_estimation(state, 0, learning_episode=1)
        print(int_model.get_state_action_estimation(state, 0))
        print(int_model.get_action_prob(state))

    for i in range(0, 4):
        int_model.update_state_action_estimation(state, 1, learning_episode=1)
        print(int_model.get_action_prob(state))

    int_model2 = InternalModelRandom(5, 0.998849)
    print(int_model2.get_action_prob(state))
    int_model2.update_state_action_estimation(state_2, 1, learning_episode=1)
    print(int_model2.get_action_prob(state_2))


if __name__ == "__main__":
    test()
