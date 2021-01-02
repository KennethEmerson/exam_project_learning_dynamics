import numpy as np
from agent import State, Agent

MOVE_LEFT = 0
MOVE_RIGHT = 1
MOVE_TOP = 2
MOVE_BOTTOM = 3
MOVE_STAY = 4
NB_MOVES = MOVE_STAY + 1


class Agent_Interface:

    def __init__(self, id, CA): # should have CA ?
        self.id = id
        self.CA = CA

    def choose_next_action(self): # is set by CA
        return self.CA.get_next_action(self.id)

    def set_state(self, state):
        if self.id == 0:
            self.CA.set_state(state)

    def update(self, new_state: State, action: int, reward: float, other_action: int,episode=1) -> None: # update the CA
        """
        Update the Centralized Agent the first time only.

        """
        if self.id == 0:
            self.CA.update(new_state, action, reward, other_action, episode)



class Centralized_Agent(Agent):
    def __init__(self, learning_rate: float, discount_rate: float, temperature: float, initial_state: State,
                initial_q_value=0.0, theta=0.998849):
        Agent.__init__(self, learning_rate, discount_rate, temperature, initial_state,
                    initial_q_value, theta)

        self.initial_theta = theta
        self.action_choice = (None,None)

    def set_state(self, state):
        self.state = state

    def get_next_action(self, id):
        if id==0:
            self.set_next_action()
            return self.action_choice[0]
        return self.action_choice[1]


    def set_next_action(self):
        """
        Choose the next action based on the current state and set it
        to the action_choice variable.

        """
        self.action_choice = self.boltzmann()

    #override
    def boltzmann(self) -> tuple:
        """
        Boltzmann function based on the Q-values of previous actions

        :return: The action chosen (MOVE_*)
        """
        action_pairs = [(action_1,action_2) for action_1 in range(NB_MOVES) for action_2 in range(NB_MOVES)]
        probas = [np.exp(self.get_q_value_for_action_pair(self.state, action_pair) / self.temperature) for action_pair in action_pairs]
        tot = sum(probas)
        probas = [p / tot for p in probas]
        action_choice_idx = np.random.choice(range(len(action_pairs)), p=probas)
        action_choice = action_pairs[action_choice_idx]
        return action_choice


    def get_q_value_for_action_pair(self, state: State, action: tuple) -> float:
        """
        Get the q value based on the current state.

        :param action: The action taken.
        :param other_action: The other player action (None if
            ignored).

        :return: The q value.
        """
        qIndex = (state.rel_position, state.other_rel_position, action[0], action[1])
        if qIndex in self.q_table:
            return self.q_table[qIndex]
        else:
            self.q_table[qIndex] = self.initial_q_value
            return self.initial_q_value


    def update(self, new_state: State, action: int, reward: float, other_action: int,episode=1) -> None:
        """
        Update the agent.

        :param new_state: The new state the agent is in.
        :param action: The action done by the agent (MOVE_*).
        :param reward: The reward obtained.
        :param other_action: The other agent action
        :param episode: the episode of the game
        """

        self.temperature = self.get_actual_theta(episode) # in paper theta and tau are equal

        if self.theta is None:
            other_action = None

        qValue = self.get_q_value_for_action_pair(self.state, (action, other_action))  # create & initialize it if it doesn't exist

        qValue = (1 - self.learning_rate) * qValue\
                 + self.learning_rate * (reward + self.discount_rate * self.max_EV_next(new_state))

        self.update_q_value(qValue, action, other_action)

        self.set_state(new_state)


    def max_EV_next(self, new_state: State) -> float:
        """
        Find the maximal expected value for every possible action, starting
        from the specified state, by using the current Q-table.

        :param new_state: The new state from which every possible action
            is computed.

        :return: The maximal expected value between all the action.
        """
        global NB_MOVES
        # @TODO: take into account the prey moves (if it changes something?)

        ev_max = None

        action_pairs = [(action_1,action_2) for action_1 in range(NB_MOVES) for action_2 in range(NB_MOVES)]
        for action in action_pairs:
            ev = self.predict_reward(new_state, action)

            if ev_max is None or ev > ev_max:
                ev_max = ev

        return ev_max


    def predict_reward(self, future_state: State, action: tuple)->float:
        """
        Predict the reward if we go into future_state by
        doing the specified action.

        :param future_state: The future state to be in.
        :param action: The action done to get into that
            state.

        :return: The predicted reward.
        """
        return self.get_q_value_for_action_pair(future_state, action)

    def get_actual_theta(self, episode: int) -> float:
        """
        Calculate the value of theta in function of the
        episode (see paper page 5, bottom left).

        :param episode: the learning episode

        :return: The actual value of theta for the specific episode
        """
        return 0.2 * (self.initial_theta ** episode)
