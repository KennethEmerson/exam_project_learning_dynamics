import pickle
from datetime import datetime

import numpy as np

from centralized_agent import Centralized_Agent, Agent_Interface
from game import Game
from qwpae_agent import QwProposedAEAgent


class HunterConfig:
    """
    Contain the configuration of the hunters playing the game.
    """

    def __init__(self, name, agent_type, game, alpha=0.3, gamma=0.9, tau=0.998849, initial_q=0.0, theta=None):
        """initialize the hunter configuration

        :param name: Name of the agent type, will be used as label for the plot.
        :param agent_type: The agent class to be used to initialize the agents.
        :param game: The game that will be played.
        :param alpha: Alpha value used by agents. Defaults to 0.1.
        :param gamma: Gamma value used by agents. Defaults to 0.5.
        :param tau: Temperature used. Defaults to 0.998849.
        :param initial_q: The initial Q-value in the Q-table. Defaults to 0.0.
        :param theta: Used as initial theta for agents with internal model.
        """
        self.name = name
        self.hunter_1 = agent_type(alpha, gamma, tau, game.get_state_hunter_1(), initial_q, theta)
        self.hunter_2 = agent_type(alpha, gamma, tau, game.get_state_hunter_2(), initial_q, theta)
        self.average_time_steps = None
        self.std_time_steps = None
        self.total_training_episodes = 0


class HunterConfig_Std(HunterConfig):
    """adds Std to the Hunter configuration"""

    def __init__(self, name, agent_type, game, alpha=0.3, gamma=0.9, tau=0.998849, initial_q=0.0, theta=None):
        HunterConfig.__init__(self, name, agent_type, game, alpha, gamma, tau, initial_q, theta)
        self.std_time_steps = None
        self.max_time_steps = None
        self.min_time_steps = None
        self.mae_time_Steps = None


class Centralized_Config:
    """
    Contain the configuration an agent coordinating the action of two hunters.
    """

    def __init__(self, name, game, alpha=0.3, gamma=0.9, tau=0.998849, initial_q=0.0, theta=None):
        """initialize the hunter configuration

        :param name: Name of the agent type, will be used as label for the plot.
        :param agent_type: The agent class to be used to initialize the agents.
        :param game: The game that will be played.
        :param alpha: Alpha value used by agents. Defaults to 0.1.
        :param gamma: Gamma value used by agents. Defaults to 0.5.
        :param tau: Temperature used. Defaults to 0.998849.
        :param initial_q: The initial Q-value in the Q-table. Defaults to 0.0.
        :param theta: Used as initial theta for agents with internal model.
        """
        self.name = name
        hunter_manager = Centralized_Agent(alpha, gamma, tau, game.get_state_hunter_1(), initial_q, theta)
        self.hunter_1 = Agent_Interface(0, hunter_manager)
        self.hunter_2 = Agent_Interface(1, hunter_manager)
        self.std_time_steps = None
        self.average_timesteps = None
        self.total_training_episodes = 0


class Centralized_Config_Std(Centralized_Config):
    """adds Std to centralized Configuration """

    def __init__(self, name, game, alpha=0.3, gamma=0.9, tau=0.998849, initial_q=0.0, theta=None):
        Centralized_Config.__init__(self, name, game, alpha, gamma, tau, initial_q, theta)
        self.std_time_steps = None
        self.max_time_steps = None
        self.min_time_steps = None
        self.mae_time_Steps = None


def do_learning_episode(game: Game, hunters, episode: int):
    """
    Play one learning episode (i.e. the hunters parameters
    get updated).

    :param game: The game to be played.
    :param hunters: A tuple with the 2 hunters.
    :param episode: The current episode of the game.
    """

    score_hunter_1 = game.penalty_hunter_1
    score_hunter_2 = game.penalty_hunter_2
    game.reset_positions()

    while score_hunter_1 != game.reward_hunter_1 and score_hunter_2 != game.reward_hunter_2:
        actions = hunters[0].choose_next_action(), hunters[1].choose_next_action()

        score_hunter_1, score_hunter_2 = game.play_one_episode(actions[0], actions[1])

        hunters[0].update(game.get_state_hunter_1(), actions[0], score_hunter_1, actions[1], episode)
        hunters[1].update(game.get_state_hunter_2(), actions[1], score_hunter_2, actions[0], episode)


def do_evaluation_episode(game: Game, hunters: tuple) -> int:
    """
    Play one evaluation episode (not hunters' parameters update).

    :param game: The game played
    :param hunters: A tuple with the first and second hunters.

    :return: The number of time steps before hunting successfully
        the prey.
    """
    score_hunter_1 = game.penalty_hunter_1
    score_hunter_2 = game.penalty_hunter_2
    game.reset_positions()

    counter = 0
    while score_hunter_1 != game.reward_hunter_1 and score_hunter_2 != game.reward_hunter_2:
        actions = hunters[0].choose_next_action(), hunters[1].choose_next_action()
        score_hunter_1, score_hunter_2 = game.play_one_episode(actions[0], actions[1])

        hunters[0].set_state(game.get_state_hunter_1())
        hunters[1].set_state(game.get_state_hunter_2())

        counter += 1

    return counter


def simulation(game: Game, hunter_config: HunterConfig, train_episodes_batch: int, eval_episodes: int,
               total_train_episodes: int):
    """
    Launch the complete simulation (i.e. training and estimation) for one set
    of hunters. The result is stored into the hunter configuration.

    :param game: The game played.
    :param hunter_config: The hunter configuration object containing the hunters.
    :param train_episodes_batch: Number of consecutive training episodes to be
        played before evaluation.
    :param eval_episodes: number of evaluation episodes to be played between
        learning.
    :param total_train_episodes: the total amount of training episodes.
    """

    hunter_1 = hunter_config.hunter_1
    hunter_2 = hunter_config.hunter_2

    average_time_steps = np.zeros(total_train_episodes // train_episodes_batch)
    std_time_steps = np.zeros(total_train_episodes // train_episodes_batch)
    max_time_steps = np.zeros(total_train_episodes // train_episodes_batch)
    min_time_steps = np.zeros(total_train_episodes // train_episodes_batch)
    mae_time_steps = np.zeros(total_train_episodes // train_episodes_batch)

    start_time = datetime.now()

    for episode in range(total_train_episodes):
        if episode % 10 == 0:
            print(f"learning episode {episode}")

        # Estimate the performances
        if episode % train_episodes_batch == 0:
            time_steps = np.zeros(eval_episodes)
            for eval_episode in range(eval_episodes):
                time_steps[eval_episode] = do_evaluation_episode(game, (hunter_1, hunter_2))

            index = episode // train_episodes_batch
            average_time_steps[index] = np.average(time_steps)
            std_time_steps[index] = np.std(time_steps)
            max_time_steps[index] = np.max(time_steps)
            min_time_steps[index] = np.min(time_steps)
            mae_time_steps[index] = np.average(np.abs(time_steps - average_time_steps[index]))
            print(f"timesteps evaluation: (average: {average_time_steps[index]}," +
                  f" std: {round(std_time_steps[index])})" +
                  f" min: {min_time_steps[index]}, max: {max_time_steps[index]}," +
                  f" MAE: {mae_time_steps[index]}")

        # Do one learning episode
        do_learning_episode(game, (hunter_1, hunter_2), episode)

    hunter_config.average_time_steps = average_time_steps

    # added for backward compatibility with older hunter_configs
    if hasattr(hunter_config, 'std_time_steps'):
        hunter_config.std_time_steps = std_time_steps
        hunter_config.max_time_steps = max_time_steps
        hunter_config.min_time_steps = min_time_steps
        hunter_config.mae_time_Steps = mae_time_steps

    end_time = datetime.now()
    print(f"\nduration testrun:{end_time - start_time}")


def save_results(hunter_config: HunterConfig, total_train_episodes: int):
    """
    Save the result into a .csv and .bin files.

    :param hunter_config: The hunter configuration (where
        the results are stored).
    :param total_train_episodes: The total number of episodes
        the agents were trained.
    """
    now = datetime.now()
    timestamp = now.strftime('%d%m%Y_%H%M')
    filename_results = f"results_{hunter_config.name}_{timestamp}.csv"
    filename_hunter_config = f"hunters_{hunter_config.name}_{timestamp}.bin"
    hunter_config.total_training_episodes = total_train_episodes

    np.savetxt(filename_results, hunter_config.average_time_steps,
               header=f"{hunter_config.name} {total_train_episodes}", delimiter=';', fmt='%u')

    with open(filename_hunter_config, 'wb') as hunter_config_list_file:
        pickle.dump(hunter_config, hunter_config_list_file)


def start_simulation(train_episodes_batch=10, eval_episodes=100, total_train_episodes=2000):
    """
    Launch the simulation (and training) and saves test results in CSV
    and hunters in bin file.
    """
    playing_field = (7, 7)

    reward = 1
    penalty = 0  # -1

    game = Game(playing_field, reward, penalty)

    # Add additional configurations to list

    # Use this for QwPAE test run
    config = HunterConfig("QwPAE", QwProposedAEAgent, game, theta=0.998849)

    # Use this for QwRAE test run
    # config = HunterConfig("QwRAE",QwRandomAEAgent,game,theta=0.998849)

    # Use this for QwSAE test run
    # config = HunterConfig("QwSAE", QwSelfModelBaseAEAgent, game, theta=0.998849)

    simulation(game, config, train_episodes_batch, eval_episodes, total_train_episodes)
    save_results(config, total_train_episodes)


def test_centralized_learner(train_episodes_batch=10, eval_episodes=100, total_train_episodes=2000):
    playing_field = (7, 7)
    reward = 1
    penalty = 0  # -1

    game = Game(playing_field, reward, penalty)
    config = Centralized_Config("Centralized", game, theta=0.998849)

    simulation(game, config, train_episodes_batch, eval_episodes, total_train_episodes)
    save_results(config, total_train_episodes)


if __name__ == "__main__":
    # start_simulation()
    test_centralized_learner()
