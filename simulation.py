import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from qwpae_agent import QwProposedAE_Agent, QwRandomAE_agent
from game import Game


MOVE_LEFT = 0
MOVE_RIGHT = 1
MOVE_TOP = 2
MOVE_BOTTOM = 3
MOVE_STAY = 4
NB_MOVES = MOVE_STAY + 1


class HunterConfig:
    """contains the configuration of two agents playing the game
    """
    def __init__(self,name,agenttype,game,alpha = 0.3, gamma = 0.9, tau = 0.998849, initial_q = 0.0,theta=None):
        """initialize the hunter configuration

        Args:
            name (string): name of the agent type, will be used as label for the plot
            agenttype (class): the agent class to be used to initialize the agents
            game (Game): the game that will be played
            alpha (float, optional): alpha value used by agents. Defaults to 0.1.
            gamma (float, optional): gamma value used by agents. Defaults to 0.5.
            tau (float, optional): temperature used. Defaults to 0.998849.
            initial_q (float, optional): [description]. Defaults to 0.0.
            theta (float,optional): used as initial theta for agents with internal model
        """
        self.name = name
        self.hunter_1 = agenttype(alpha, gamma, tau, game.get_state_hunter_1(), initial_q,theta)
        self.hunter_2 = agenttype(alpha, gamma, tau, game.get_state_hunter_2(), initial_q,theta)
        self.average_timesteps = None
        self.total_training_episodes = 0



def learning_episode(game:Game, hunter_1,hunter_2,episode:int):
    """playes one learning episode where the hunters can update their internal parameters

    Args:
        game (Game): the game to be played
        hunter_1 (Agent): the agent object for hunter 1
        hunter_2 (Agent): the agent object for hunter 2
        episode (int): the episode played
    """
    score = game.penalty
    game._reset_positions()
    while(score != game.reward):
        hunter_1_action,score,hunter_2_action = game.play(hunter_1.choose_next_action(),hunter_2.choose_next_action())
        hunter_1.update(game.get_state_hunter_1(),hunter_1_action,score,hunter_2_action,episode)
        hunter_2.update(game.get_state_hunter_2(),hunter_2_action,score,hunter_1_action,episode)


def evaluation_episode(game:Game, hunter_1,hunter_2,episode:int):
    """playes one evaluation episode where the hunters are not allowed to update

    Args:
        game (Game): the game to be played
        hunter_1 (Agent): the agent object for hunter 1
        hunter_2 (Agent): the agent object for hunter 2
        episode (int): the episode played
    Returns:
        [int]: number of timesteps required to catch the prey
    """
    score = game.penalty
    game._reset_positions()
    counter = 0
    while(score != game.reward):
        hunter_1_action,score,hunter_2_action = game.play(hunter_1.choose_next_action(),hunter_2.choose_next_action())

        hunter_1.set_state(game.get_state_hunter_1())
        hunter_2.set_state(game.get_state_hunter_2())

        counter += 1

    #print("Hunter 1\n\n", hunter_1.q_table)
    #print("Hunter 2\n\n", hunter_2.q_table)
    """print(max(hunter_1.q_table.values()))

    v = list(hunter_1.q_table.values())
    #v.sort()
    s = set(v)

    for val in s:
        print(val, v.count(val))

    input("...")"""

    return counter


def simulation(game:Game,hunter_config:HunterConfig,train_episodes_batch,eval_episodes,total_train_episodes):
    """playes the complete simulation for one set of hunters

    Args:
        game (Game): the game to be played
        hunter_config (HunterConfig): the hunter configuration object containing the hunters
        train_episodes_batch (int): number of consecutive training episodes to be played before evaluation
        eval_episodes (int): number of evaluation episodes to be played between learning
        total_train_episodes (int): the total amount of training episodes
    """

    hunter_1 = hunter_config.hunter_1
    hunter_2 = hunter_config.hunter_2
    
    average_timesteps = np.zeros(total_train_episodes//train_episodes_batch)
    start_time = datetime.now()
    for episode in range(total_train_episodes):
        
        if(episode%10==0): print(f"learning episode {episode}")

        #learning_episode(game, hunter_1,hunter_2,episode)

        if(episode % train_episodes_batch==0):
            timesteps = np.zeros(eval_episodes)
            for eval_episode in range(eval_episodes):
                timesteps[eval_episode] = evaluation_episode(game, hunter_1,hunter_2,episode)
                #print(f"eval episode {eval_episode}: timesteps:{timesteps[eval_episode]}")
            average_timesteps[episode//train_episodes_batch] = np.average(timesteps)
            print(f"average timesteps evaluation: {average_timesteps[episode//train_episodes_batch]}")
        learning_episode(game, hunter_1,hunter_2,episode)
    hunter_config.average_timesteps =  average_timesteps
    
    end_time = datetime.now()
    print(f"\nduration testrun:{end_time-start_time}")

def save_results(test_case,total_train_episodes):
    now = datetime.now()
    timestamp = now.strftime('%d%m%Y_%H%M')
    filename_results = f"results_{test_case.name}_{timestamp}.csv" 
    filename_hunter_config = f"hunters_{test_case.name}_{timestamp}.bin" 
    test_case.total_training_episodes = total_train_episodes

    np.savetxt(filename_results, test_case.average_timesteps, 
                header=f"{test_case.name} {total_train_episodes}", delimiter=';',fmt='%u')
    
    with open(filename_hunter_config, 'wb') as hunter_config_list_file:
        pickle.dump(test_case, hunter_config_list_file)

def test():
    """main test: does simulation and saves test results in CSV and hunters in bin file
    """
    playing_field = (7,7)
    dict_action_to_coord = {MOVE_TOP:(0,-1), MOVE_RIGHT:(1,0), MOVE_BOTTOM:(0,1), MOVE_LEFT:(-1,0), MOVE_STAY:(0,0)}
    prey_action_prob = np.array([0,1/3,1/3,1/3,0])
    reward = 1
    penalty = 0#-1
    
    game =  Game(playing_field,reward,penalty,dict_action_to_coord,prey_action_prob) 
    
    # add additional configurations to list

    test_case = HunterConfig("QwPAE",QwProposedAE_Agent,game,theta=0.998849) # use this for QwPAE testrun
    #test_case = HunterConfig("QwRAE",QwRandomAE_agent,game,theta=0.998849)    # use this for QwRAE testrun
    
    train_episodes_batch = 10   #should be 10
    eval_episodes = 100           #should be 100
    total_train_episodes = 2000    #should be 2000


    simulation(game,test_case,train_episodes_batch,eval_episodes,total_train_episodes)
    save_results(test_case,total_train_episodes)

if __name__ == "__main__":
    test()