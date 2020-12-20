import numpy as np

###########################################################################################
# class to create and interact with the game environment
#  
# on initialisation following paramters must be provided:
#   reward: value that player receives when pray is captured
#   penalty: penalty cost for every action taken without capturing the pray
#   dict_action_to_coord: python dictionary that maps actions to actual change in
#                         coordinates. the key is a integer, values are tuples of size 2
#                         example: {0:(0,-1), 1:(1,0), 2:(0,1), 3:(-1,0), 4:(0,0)}
#                                  where up:0, right: 1, down:2, left: 3, stay:4
#   pray_action_prob: probability distribution np.array for each action of the pray.
#                     example: np.array([1/3,1/3,1/3,0,0])

# note: coordinates are counted from top left of the playing field, hence (0,0) is topleft cell
###########################################################################################
class game:
    def __init__(self,playing_field_size,reward,penalty,dict_action_to_coord,pray_action_prob):
        
        print(f"playing field: {playing_field_size}")
        print(f"playing field: {playing_field_size[0]}")
        print(f"playing field: {type(playing_field_size[0])}")
        self.x_max = playing_field_size[0]
        self.y_max = playing_field_size[1]
        self.reward = reward   
        self.penalty = penalty
        self.dict_action_to_coord = dict_action_to_coord
        
        #randomly places pray and hunters in the playing field
        # (note: playing field is hardcoded to size (7,7))
        self.pray_position = np.array([np.random.randint(self.x_max),np.random.randint(self.y_max)])
        self.hunter_1_position = np.array([np.random.randint(self.x_max),np.random.randint(self.y_max)])
        self.hunter_2_position = np.array([np.random.randint(self.x_max),np.random.randint(self.y_max)])
    
        #probabilities for moving :up, right, down, left, staying
        self.pray_action_prob = pray_action_prob


    #######################################################################################
    # updates the given position considering the action coordinates provided
    #   position: the actual position of the subject (numpy array (x_coord,y_coord))
    #   action: the action taken as defined as key in dict_action_to_coord (integer value)
    #   returns updated position
    #######################################################################################

    def update_position(self,position,action):
        # update position conform the action taken
        position = (position + self.dict_action_to_coord.get(action))
        
        # adjust coordinates if they exceed game environment limits
        if(position[0]>(self.x_max-1)): position[0] = position[0] - self.x_max
        if(position[0]<0):              position[0] = self.x_max - 1
        if(position[1]>(self.y_max-1)): position[1] = position[1] - self.y_max
        if(position[1]<0):              position[1] = self.y_max - 1
        return position
    

    #######################################################################################
    # transforms the absolute position to relative positions of the hunters to the pray 
    #   returns the relative positions of both hunters to the pray, used as state definition
    #   by both hunters 
    #######################################################################################
    
    #transforms the absolute position to relative positions of the hunters to the pray 
    def get_relative_locations(self):
        rel_loc_hunter_1 =  self.pray_position-self.hunter_1_position
        rel_loc_hunter_2 =  self.pray_position-self.hunter_2_position
        return rel_loc_hunter_1,rel_loc_hunter_2
    

    #######################################################################################
    #  playes one round in the game
    #     hunter_1_action: the action 
    # 
    #######################################################################################


    #up:0, right, down, left, staying
    def play(self,hunter_1_action,hunter_2_action):
        
        #randomly choose pray action with given probabilities 
        pray_action = np.random.choice(self.pray_action_prob.size, p=self.pray_action_prob)
        
        #update the absolute position of pray and hunters
        self.pray_position = self.update_position(self.pray_position,pray_action)
        self.hunter_1_position = self.update_position(self.hunter_1_position,hunter_1_action)
        self.hunter_2_position = self.update_position(self.hunter_2_position,hunter_2_action)
        
        #check if the pray is captured
        score = self.penalty
        hunter_1_rel_pos, hunter_2_rel_pos = self.get_relative_locations()
        
        #check if the pray is captured horizontally
        if(hunter_1_rel_pos[0] == 0 and hunter_2_rel_pos[0] == 0):
            if(hunter_1_rel_pos[1] == -1 and hunter_2_rel_pos[1] == 1): score = self.reward
            if(hunter_1_rel_pos[1] == 1 and hunter_2_rel_pos[1] == -1): score = self.reward
        
        #check if the pray is captured vertically
        if(hunter_1_rel_pos[1] == 0 and hunter_2_rel_pos[1] == 0):
            if(hunter_1_rel_pos[0] == -1 and hunter_2_rel_pos[0] == 1): score = self.reward
            if(hunter_1_rel_pos[0] == 1 and hunter_2_rel_pos[0] == -1): score = self.reward    
         
        #returns the relative positions of the hunters and the score achieved 
        return self.get_relative_locations(),score


#######################################################################################
# scenario to test class
#######################################################################################

def test():
  playing_field = (7,7)
  dict_action_to_coord = {0:(0,-1), 1:(1,0), 2:(0,1), 3:(-1,0), 4:(0,0)}
  pray_action_prob = np.array([1/3,1/3,1/3,0,0])
  reward = 1
  penalty = -1
  
  test_game =  game(playing_field,reward,penalty,dict_action_to_coord,pray_action_prob) 
  print()
  print(f"initial position pray: {test_game.pray_position}")  
  print(f"initial position hunter 1: {test_game.hunter_1_position}")  
  print(f"initial position hunter 2: {test_game.hunter_2_position}\n")  

  for i in range(0,8): 
    hunter_1_action = 0
    hunter_2_action = 1
    test_game.play(hunter_1_action,hunter_2_action)

    print(f"position pray: {test_game.pray_position}")  
    print(f"position hunter 1: {test_game.hunter_1_position}")  
    print(f"position hunter 2: {test_game.hunter_2_position}") 
    print(f"relative positions hunters: {test_game.get_relative_locations()}")

if __name__ == "__main__":
    test()
