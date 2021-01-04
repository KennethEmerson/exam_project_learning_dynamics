from agent.agent import State, Agent
from internalmodel import InternalModel, InternalModelRandom
from agent.move import *
import numpy as np


class QwProposedAEAgent(Agent):
    def __init__(self, learning_rate: float, discount_rate: float, temperature: float, initial_state: State,
                 initial_q_value=0.0, theta=0.998849):
        Agent.__init__(self, learning_rate, discount_rate, temperature, initial_state,
                       initial_q_value, theta)
        self.internal_model = InternalModel(theta)

    def get_q_value_with_random_state(self, state: State, action: int, other_action: int = None) -> float:
        """
        Get the q value based on the current state.

        :param action: The action taken.
        :param other_action: The other player action (None if
            ignored).

        :return: The q value.
        """
        qIndex = (state.rel_position, state.other_rel_position, action, other_action)
        if qIndex in self.q_table:
            return self.q_table[qIndex]
        else:
            self.q_table[qIndex] = self.initial_q_value
            return self.initial_q_value

    def expected_value(self, action: int) -> float:
        """
        Compute the expected value for an action for the current
        state (used by the boltzmann function).

        :param action: The action (i.e. MOVE_*)

        :return: The expected value.
        """
        sum = 0
        for other_action in range(NB_MOVES):
            sum = sum + (self.internal_model.get_state_action_estimation(self.state, other_action) *
                         self.get_q_value(action, other_action))
        return sum

    def predict_reward(self, future_state: State, action: int) -> float:
        """
        Predict the reward if we go into future_state by
        doing the specified action.

        :param future_state: The future state to be in.
        :param action: The action done to get into that
            state.

        :return: The predicted reward.
        """
        moves_probability = self.internal_model.get_action_prob(future_state)
        max_probability = max(moves_probability)
        max_q = None

        for other_action in range(len(moves_probability)):
            if moves_probability[other_action] == max_probability:
                q = self.get_q_value_with_random_state(future_state, action, other_action)
                if max_q is None or q > max_q:
                    max_q = q

        return max_q

    def update(self, new_state: State, action: int, reward: float, other_action: int, episode=1):
        """
        Update the agent.

        :param new_state: The new state the agent is in.
        :param action: The action done by the agent (MOVE_*).
        :param reward: The reward obtained.
        :param other_action: The other agent action
        :param episode: the episode of the game
        """
        self.temperature = self.internal_model.get_actual_theta(episode)  # in paper theta and tau are equal
        self.internal_model.update_state_action_estimation(self.state, other_action, episode)
        super().update(new_state, action, reward, other_action)


class QwRandomAEAgent(QwProposedAEAgent):
    def __init__(self, learning_rate: float, discount_rate: float, temperature: float, initial_state: State,
                 initial_q_value=0.0, theta=0.998849):
        Agent.__init__(self, learning_rate, discount_rate, temperature, initial_state,
                       initial_q_value, theta)
        self.internal_model = InternalModelRandom(theta)

    def predict_reward(self, future_state: State, action: int) -> float:
        """
        Predict the reward if we go into future_state by
        doing the specified action.

        :param future_state: The future state to be in.
        :param action: The action done to get into that
            state.

        :return: The predicted reward.
        """
        return self.get_q_value_with_random_state(future_state, action, np.random.choice(NB_MOVES))


def test():
    alpha = 0.1
    gamma = 0.5
    tau = 4.
    state = State((-10, -10), (10, 10))
    initial_q = 0.0

    agent = QwProposedAEAgent(alpha, gamma, tau, state, initial_q)
    action = agent.choose_next_action()
    print(action)
    new_pos = agent.compute_new_position(action, (-10, -10))
    new_state = State(new_pos, (10, 9))
    agent.update(new_state, action, 5, action)

    action = agent.choose_next_action()
    print(action)


if __name__ == "__main__":
    test()
