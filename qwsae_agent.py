from agent import State
from internalmodel import InternalSelfModel
from move import *
from qwpae_agent import QwProposedAEAgent


class QwSelfModelBaseAEAgent(QwProposedAEAgent):
    def __init__(self, learning_rate: float, discount_rate: float, temperature: float, initial_state: State,
                 initial_q_value=0.0, theta=0.998849):
        super().__init__(learning_rate, discount_rate, temperature, initial_state, initial_q_value, theta)
        self.internal_model = InternalSelfModel(theta, self)


def test():
    alpha = 0.1
    gamma = 0.5
    tau = 4.
    state = State((-10, -10), (10, 10))
    initial_q = 0.0

    agent = QwSelfModelBaseAEAgent(alpha, gamma, tau, state, initial_q)
    action = agent.choose_next_action()
    print(action)
    new_pos = agent.compute_new_position(action, (-10, -10))
    new_state = State(new_pos, (10, 9))
    agent.update(new_state, action, 5, action)

    action = agent.choose_next_action()
    print(action)


if __name__ == "__main__":
    test()
