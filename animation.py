import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


from game import Game
from qwpae_agent import Qwpae_Agent
from simulation import HunterConfig

def game_showcase(game:Game,hunter_config):
    """shows animation of the game

    Args:
        game (Game): the game to be played
        hunter_config (HunterConfig): the hunter configuration

    Returns:
        [type]: [description]
    """
    
    hunter_1 = hunter_config.hunter_1
    hunter_2 = hunter_config.hunter_2
    data = np.zeros([game.x_max,game.y_max])
    fig, ax = plt.subplots(figsize=(7,7))
    game._reset_positions()
    reward = game.reward
    
    def animate(i):
        hunter_1_action,score,hunter_2_action = game.play(hunter_1.choose_next_action(),hunter_2.choose_next_action())
        if(score == reward):ani.event_source.stop()
        
        hunter_1.set_state(game.get_state_hunter_1())
        hunter_2.set_state(game.get_state_hunter_2())
        
        hunter_1_pos = game.hunter_1_position
        hunter_2_pos = game.hunter_2_position
        prey_pos = game.prey_position
        
        data = np.zeros([game.x_max,game.y_max])
        data[prey_pos[0],prey_pos[1]] = 2
        data[hunter_1_pos[0],hunter_1_pos[1]] = 1
        data[hunter_2_pos[0],hunter_2_pos[1]] = 1

        ax.clear()
        ax.imshow(data)
        return [ax]

    ani = FuncAnimation(fig, animate, interval=200,blit=True)
    plt.show()  

def save_hunter_config_list(filename,hunter_config_list):
  with open(filename, 'wb') as hunter_config_list_file:
    pickle.dump(hunter_config_list, hunter_config_list_file)

def load_hunter_config_list(filename):
  with open(filename, 'rb') as hunter_config_list_file:
     return pickle.load(hunter_config_list_file)

def showcase_from_file(game:Game,filename,hunter_config_list_index:int):
  hunter_config_list = load_hunter_config_list(filename)
  game_showcase(game,hunter_config_list[hunter_config_list_index])

if __name__ == "__main__":

  MOVE_LEFT = 0
  MOVE_RIGHT = 1
  MOVE_TOP = 2
  MOVE_BOTTOM = 3
  MOVE_STAY = 4
  NB_MOVES = MOVE_STAY + 1

  playing_field = (7,7)
  dict_action_to_coord = {MOVE_TOP:(0,-1), MOVE_RIGHT:(1,0), MOVE_BOTTOM:(0,1), MOVE_LEFT:(-1,0), MOVE_STAY:(0,0)}
  prey_action_prob = np.array([1/3,1/3,0,1/3,0])
  reward = 1
  penalty = -1
    
  game =  Game(playing_field,reward,penalty,dict_action_to_coord,prey_action_prob)  
  showcase_from_file(game,"testfile.bin",0)
