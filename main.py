"""
Authors: Kenneth Emerson (VUB), Florian Ilzkovitz (ULB)
         Arnaud Leponce (ULB), Aliaksei Vainilovich (ULB)

Description:
    Review of the paper "Multi-agent reinforcement learning: An approach
    based on the other agent's internal model".

    Those files contain a reproduction of the code made by the original
    authors of that paper.
"""

import simulation


def main():
    simulation.test_centralized_learner(10, 100, 2000)


main()
