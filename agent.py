import numpy as np

MOVE_LEFT = 0
MOVE_RIGHT = 1
MOVE_TOP = 2
MOVE_BOTTOM = 3
MOVE_STAY = 4
NB_MOVES = MOVE_STAY + 1


class State:
    def __init__(self, rel_position, other_rel_position):
        self.rel_position = rel_position
        self.other_rel_position = other_rel_position


class Agent:
    """
    Abstract agent.
    """

    def __init__(self, learning_rate: float, discount_rate: float, temperature: float, initial_state: State,
                 initial_q_value=0.0, theta=None):
        """
        Initialize an agent.

        :param learning_rate: The learning rate (alpha)
        :param discount_rate: The discount rate (gamma)
        :param temperature: The temperature (Boltzmann tau)
        :param initial_state: The initial state
        :param initial_q_value: The initial values of the Q-table
        :param theta: The theta for the internal model (None if the
            internal model is not used).
        """
        self.q_table = dict()
        self.initial_q_value = initial_q_value
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.temperature = temperature
        self.state = initial_state
        self.theta = theta

    def get_q_value(self, action: int, other_action: int=None) -> float:
        """
        Get the q value based on the current state.

        :param action: The action taken.
        :param other_action: The other player action (None if
            ignored).

        :return: The q value.
        """
        qIndex = (self.state.rel_position, self.state.other_rel_position, action, other_action)
        if qIndex in self.q_table:
            return self.q_table[qIndex]
        else:
            self.q_table[qIndex] = self.initial_q_value
            return self.initial_q_value

    def update_q_value(self, q_value: float, action: int, other_action=None):
        """
        Update the Q-table (for the state we are in).

        :param q_value: The new Q-value.
        :param action: The action done.
        :param other_action: The other agent action (ignored if None).
        """
        qIndex = (self.state.rel_position, self.state.other_rel_position, action, other_action)
        self.q_table[qIndex] = q_value

    def set_state(self, state: State):
        """
        Set the state to a new one without updating anything else.

        :param state: The new value of the agent state.
        """
        self.state = state

    def update(self, new_state: State, action: int, reward: float, other_action: int = None) -> None:
        """
        Update the agent.

        :param new_state: The new state the agent is in.
        :param action: The action done by the agent (MOVE_*).
        :param reward: The reward obtained.
        :param other_action: The other agent accent (might
            be ignored depending on the agent).
        """
        if self.theta is None:
            other_action = None

        qValue = self.get_q_value(action, other_action)  # create & initialize it if it doesn't exist

        qValue = (1 - self.learning_rate) * qValue\
                 + self.learning_rate * (reward + self.discount_rate * self.max_EV_next(new_state))

        self.update_q_value(qValue, action, other_action)

        self.state = new_state

    def compute_new_position(self, action: int, initial_position: (int, int)) -> (int, int):
        """
        Compute a new position from an action.

        :param action: The action done (MOVE_*).
        :param initial_position: The initial position (x, y).

        :return: The new relative x and y coordinate
        """
        if action == MOVE_LEFT:
            return initial_position[0] - 1, initial_position[1]
        elif action == MOVE_RIGHT:
            return initial_position[0] + 1, initial_position[1]
        elif action == MOVE_BOTTOM:
            return initial_position[0], initial_position[1] + 1
        elif action == MOVE_TOP:
            return initial_position[0], initial_position[1] - 1
        else:
            return initial_position

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
        for action in range(NB_MOVES):
            own_position = self.compute_new_position(action, new_state.rel_position)
            for other_action in range(NB_MOVES):
                other_position = self.compute_new_position(other_action, new_state.other_rel_position)

                state = State(own_position, other_position)
                ev = self.predict_reward(state, action)

                if ev_max is None or ev > ev_max:
                    ev_max = ev

        return ev_max

    def boltzmann(self) -> int:
        """
        Boltzmann function based on the Q-values of the possible next
        actions.

        :return: The action chosen (MOVE_*)
        """
        probas = [np.exp(self.expected_value(action) / self.temperature) for action in range(NB_MOVES)]
        tot = sum(probas)
        probas = [p / tot for p in probas]

        action = np.random.choice(range(NB_MOVES), p=probas)

        return action

    def choose_next_action(self) -> int:
        """
        Choose the next action based on the current state.

        :return: Choose the next action (MOVE_*)
        """
        return self.boltzmann()

    def predict_reward(self, future_state: State, action: int):
        """
        Predict the reward if we go into future_state by
        doing the specified action.

        :param future_state: The future state to be in.
        :param action: The action done to get into that
            state.

        :return: The predicted reward.
        """
        print("predict_reward() must be implemented.")
        return 0.0

    def expected_value(self, action: int) -> float:
        """
        Compute the expected value for an action for the current
        state (used by the boltzmann function).

        Note: need to be overwritten.

        :param action: The action (i.e. MOVE_*)

        :return: The expected value.
        """
        print("expected_value() must be implemented.")
        return 0.0


if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.5
    tau = 4.
    state = State((-10, -10), (10, 10))
    initial_q = 0.0

    agent = Agent(alpha, gamma, tau, state, initial_q)
    action = agent.choose_next_action()

    new_pos = agent.compute_new_position(action, (-10, -10))
    new_state = State(new_pos, (10, 9))
    agent.update(new_state, action, 5)

    action = agent.choose_next_action()
