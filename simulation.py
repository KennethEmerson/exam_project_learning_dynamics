import numpy as np
import matplotlib as plt
from qwpae_agent import Qwpae_Agent
from game import Game

MOVE_LEFT = 0
MOVE_RIGHT = 1
MOVE_TOP = 2
MOVE_BOTTOM = 3
MOVE_STAY = 4
NB_MOVES = MOVE_STAY + 1

def learning_episode(game:Game, hunter_1:Qwpae_Agent,hunter_2:Qwpae_Agent):
    state = game.get_state()

def test():
    
    playing_field = (7,7)
    dict_action_to_coord = {MOVE_TOP:(0,-1), MOVE_RIGHT:(1,0), MOVE_BOTTOM:(0,1), MOVE_LEFT:(-1,0), MOVE_STAY:(0,0)}
    prey_action_prob = np.array([1/3,1/3,1/3,0,0])
    reward = 1
    penalty = -1
  
    game =  Game(playing_field,reward,penalty,dict_action_to_coord,prey_action_prob) 
    
    alpha = 0.1
    gamma = 0.5
    tau = 0.998849
    initial_q = 0.0


    hunter_1 = Qwpae_Agent(alpha, gamma, tau, game.get_state_hunter_1(), initial_q)
    hunter_2 = Qwpae_Agent(alpha, gamma, tau, game.get_state_hunter_2(), initial_q)
    
    for episode in range(0,5):
    
        hunter_1_action,score,hunter_2_action = game.play(hunter_1.choose_next_action(),hunter_2.choose_next_action())
        
        print(f"hunter_1 state: {game.get_state_hunter_1().rel_position}{game.get_state_hunter_1().other_rel_position}")
        print(f"hunter_2 state: {game.get_state_hunter_2().rel_position}{game.get_state_hunter_2().other_rel_position}")
        print(f"hunter 1 action: {hunter_1_action}")
        print(f"hunter 2 action: {hunter_2_action}")
        print(f"score:{score}")

        hunter_1.update(game.get_state_hunter_1(),hunter_1_action,score,hunter_2_action)
        hunter_2.update(game.get_state_hunter_2(),hunter_2_action,score,hunter_1_action)
        hunter_1.internal_model.update_state_action_estimation(game.get_state_hunter_1(),hunter_2_action,episode)
        hunter_2.internal_model.update_state_action_estimation(game.get_state_hunter_1(),hunter_2_action,episode)


if __name__ == "__main__":
    test()