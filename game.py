import numpy as np

class Game:
    """class to create and interact with the game environment
    """
    def __init__(self,playing_field_size,reward,penalty,dict_action_to_coord,prey_action_prob):
        """
        initialize game and place prey and hunters on random positions

        :param playing_field_size (tuple(int,int)): size of game board (x,y)
        :param reward (numeric value): score returned to hunters when prey is catched after episode
        :param penalty (numeric value): score returned to hunters when prey is NOT catched after episode
        :param dict_action_to_coord (dict): maps actions to actual change in
                                            coordinates. the key is a integer, values are tuples of size 2
                                            example: {0:(0,-1), 1:(1,0), 2:(0,1), 3:(-1,0), 4:(0,0)}
        :param prey_action_prob (1-D array-like): probability distribution for each action of the prey.
                                                  must match the number of keys in dict_action_to_coord
                                                  example: np.array([1/3,1/3,1/3,0,0])
        """
        self.x_max = playing_field_size[0]
        self.y_max = playing_field_size[1]
        self.reward = reward   
        self.penalty = penalty
        self.dict_action_to_coord = dict_action_to_coord
        self.prey_action_prob = prey_action_prob

        self._reset_positions()


    def _reset_positions(self): 
        """
        place prey and hunters randomly in the playing field
        """
        self.prey_position = np.array([np.random.randint(self.x_max),np.random.randint(self.y_max)])
        self.hunter_1_position = np.array([np.random.randint(self.x_max),np.random.randint(self.y_max)])
        self.hunter_2_position = np.array([np.random.randint(self.x_max),np.random.randint(self.y_max)])


    def update_position(self,position,action):
        """ 
        update the given position considering the action coordinates provided

        :param position (numpy array): the current x and y position [x,y]
        :param action (integer): key as used in dict_action_to_coord for the action.

        Returns: 
            [numpy array]: updated position
        """
        position = (position + self.dict_action_to_coord.get(action))
        
        if(position[0]>(self.x_max-1)): position[0] = position[0] - self.x_max
        if(position[0]<0):              position[0] = self.x_max - 1
        if(position[1]>(self.y_max-1)): position[1] = position[1] - self.y_max
        if(position[1]<0):              position[1] = self.y_max - 1
        return position
    

    def get_relative_locations(self):
        """
        transform the absolute positions to relative positions of the hunters to the prey 

        Returns:
            [numpy array],[numpy array]: relative position of hunter 1 and hunter 2 in context of the prey
        """
        def get_relative_location(prey_coord,hunter_coord,max_coord):
            dist = np.array([   
                            (prey_coord - hunter_coord),
                            (prey_coord  +  max_coord - hunter_coord),
                            (prey_coord - max_coord -hunter_coord)])
            min_index = np.argmin(np.abs(dist)) # get index of lowest abs distance
            return dist[min_index] # return value of index

        rel_loc_hunter_1 =  np.array([
                            get_relative_location(self.prey_position[0],self.hunter_1_position[0],self.x_max),
                            get_relative_location(self.prey_position[1],self.hunter_1_position[1],self.y_max)
                            ])
        rel_loc_hunter_2 =  np.array([
                            get_relative_location(self.prey_position[0],self.hunter_2_position[0],self.x_max),
                            get_relative_location(self.prey_position[1],self.hunter_2_position[1],self.y_max)
                            ])
        
        return rel_loc_hunter_1,rel_loc_hunter_2
    

    def check_score(self):
        """
        test if prey is captured and determine score for hunters as such

        Returns:
            [numeric value]: the score
        """
        score = self.penalty
        hunter_1_rel_pos, hunter_2_rel_pos = self.get_relative_locations()
        
        #check if the prey is captured horizontally
        if(hunter_1_rel_pos[0] == 0 and hunter_2_rel_pos[0] == 0):
            if(hunter_1_rel_pos[1] == -1 and hunter_2_rel_pos[1] == 1): score = self.reward
            if(hunter_1_rel_pos[1] == 1 and hunter_2_rel_pos[1] == -1): score = self.reward
        
        #check if the prey is captured vertically
        if(hunter_1_rel_pos[1] == 0 and hunter_2_rel_pos[1] == 0):
            if(hunter_1_rel_pos[0] == -1 and hunter_2_rel_pos[0] == 1): score = self.reward
            if(hunter_1_rel_pos[0] == 1 and hunter_2_rel_pos[0] == -1): score = self.reward 

        return score
    

    def play(self,hunter_1_action,hunter_2_action):
        """
        play one episode of the game

        :param hunter_1_action (integer): action selected by hunter 1 (key as used in dict_action_to_coord for the action).
        :param hunter_2_action (integer): action selected by hunter 2 (key as used in dict_action_to_coord for the action).
        
        Returns:
            [numpy array]: relative position of hunter 1 in context of the prey
            [numpy array]: relative position of hunter 2 in context of the prey
            [numeric value]: score for hunters earned during this episode
        """
        
        #randomly choose prey action with given probabilities 
        prey_action = np.random.choice(self.prey_action_prob.size, p=self.prey_action_prob)
        
        #update the absolute position of prey and hunters
        self.prey_position = self.update_position(self.prey_position,prey_action)
        self.hunter_1_position = self.update_position(self.hunter_1_position,hunter_1_action)
        self.hunter_2_position = self.update_position(self.hunter_2_position,hunter_2_action)
        
        return self.get_relative_locations(),self.check_score() 


#######################################################################################
# scenario to test class
#######################################################################################

def test():
  playing_field = (7,7)
  dict_action_to_coord = {0:(0,-1), 1:(1,0), 2:(0,1), 3:(-1,0), 4:(0,0)}
  prey_action_prob = np.array([1/3,1/3,1/3,0,0])
  reward = 1
  penalty = -1
  
  test_game =  Game(playing_field,reward,penalty,dict_action_to_coord,prey_action_prob) 
  print()
  print(f"initial position prey: {test_game.prey_position}")  
  print(f"initial position hunter 1: {test_game.hunter_1_position}")  
  print(f"initial position hunter 2: {test_game.hunter_2_position}\n")  

  for i in range(0,8): 
    hunter_1_action = 0
    hunter_2_action = 1
    print(test_game.play(hunter_1_action,hunter_2_action))

    print(f"position prey: {test_game.prey_position}")  
    print(f"position hunter 1: {test_game.hunter_1_position}")  
    print(f"position hunter 2: {test_game.hunter_2_position}") 
    print(f"relative positions hunters: {test_game.get_relative_locations()}")

if __name__ == "__main__":
    test()
