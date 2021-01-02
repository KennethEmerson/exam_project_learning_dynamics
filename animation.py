import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from game import Game
from simulation import HunterConfig, Centralized_Config


def game_showcase(game: Game, hunter_config: HunterConfig):
    """shows animation of the game

    Args:
        game (Game): the game to be played
        hunter_config (HunterConfig): the hunter configuration

    Returns:
        [type]: [description]
    """

    hunter_1 = hunter_config.hunter_1
    hunter_2 = hunter_config.hunter_2
    data = np.zeros([game.x_max, game.y_max])
    fig, ax = plt.subplots(figsize=(7, 7))
    game.reset_positions()
    reward_hunter_1 = game.reward_hunter_1
    reward_hunter_2 = game.reward_hunter_2

    def animate(_=None):
        actions = hunter_1.choose_next_action(), hunter_2.choose_next_action()
        score_hunter_1, score_hunter_2 = game.play_one_episode(actions[0], actions[1])
        if score_hunter_1 == reward_hunter_1 or score_hunter_2 == reward_hunter_2:
            ani.event_source.stop()

        hunter_1.set_state(game.get_state_hunter_1())
        hunter_2.set_state(game.get_state_hunter_2())

        hunter_1_pos = game.hunter_1_position
        hunter_2_pos = game.hunter_2_position
        prey_pos = game.prey_position

        data = np.zeros([game.x_max, game.y_max])
        data[prey_pos[1], prey_pos[
            0]] = 2  # numpy array uses first index for rows (y coord) and second for columns (x coord)
        data[hunter_1_pos[1], hunter_1_pos[
            0]] = 1  # numpy array uses first index for rows (y coord) and second for columns (x coord)
        data[hunter_2_pos[1], hunter_2_pos[
            0]] = 1  # numpy array uses first index for rows (y coord) and second for columns (x coord)

        ax.clear()
        ax.imshow(data)
        return [ax]

    #game.prey_action_prob = np.array([1/4,1/4,1/4,1/4,0])
    ani = FuncAnimation(fig, animate, interval=200, blit=True)
    plt.show()


def load_hunter_config(filename: str) -> HunterConfig:
    with open(filename, 'rb') as hunter_config_file:
        return pickle.load(hunter_config_file)


def showcase_from_file(game: Game, filename: str):
    hunter_config = load_hunter_config(filename)
    game_showcase(game, hunter_config)


if __name__ == "__main__":
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_TOP = 2
    MOVE_BOTTOM = 3
    MOVE_STAY = 4
    NB_MOVES = MOVE_STAY + 1

    playing_field = (7, 7)
    dict_action_to_coord = {MOVE_TOP: (0, -1), MOVE_RIGHT: (1, 0), MOVE_BOTTOM: (0, 1), MOVE_LEFT: (-1, 0),
                            MOVE_STAY: (0, 0)}

    reward = 1
    penalty = -1

    game = Game(playing_field, reward, penalty)
    # showcase_from_file(game, "hunters_QwPAE_01012021_0134.bin")
    showcase_from_file(game, "hunters_Centralized_02012021_0053.bin")
