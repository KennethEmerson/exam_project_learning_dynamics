from game import is_prey_caught_heterogeneous
from simulation import *


def simulation_figure_5():
    # NOTE: to be sure, all parameters are set explicitly !!!!!
    # game parameters:

    playing_field = (7, 7)
    is_prey_caught_function = is_prey_caught_heterogeneous
    reward_all = 1
    penalty_all = 0

    game = Game(playing_field, reward_all, penalty_all, reward_all, penalty_all, is_prey_caught_function)

    # general hunter parameters
    alpha = 0.3
    gamma = 0.9
    tau = 0.998849
    initial_q = 0.0
    theta = 0.998849

    # Use this for self model
    """config = HunterConfig_Std(name="Q-learning with self-model based estimation",
                                agent_type= QwSelfModelBaseAEAgent,
                                game=game,
                                alpha= alpha,
                                gamma= gamma,
                                tau= tau,
                                initial_q=initial_q,
                                theta=0.998849)"""

    # Use this for QwPAE test run
    config = HunterConfig_Std(name="Q-learning with proposed action estimation",
                              agent_type=QwProposedAEAgent,
                              game=game,
                              alpha=alpha,
                              gamma=gamma,
                              tau=tau,
                              initial_q=initial_q,
                              theta=0.998849)

    # sim parameters
    train_episodes_batch = 10
    eval_episodes = 100
    total_train_episodes = 2000

    simulation(game=game,
               hunter_config=config,
               train_episodes_batch=train_episodes_batch,
               eval_episodes=eval_episodes,
               total_train_episodes=total_train_episodes)

    save_results(config, total_train_episodes)


if __name__ == "__main__":
    simulation_figure_5()
