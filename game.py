import numpy as np
from agent import State
from move import *

class Game:
    """
    Create a game playable step by step.
    """
    

    def __init__(self, playing_field_size:tuple, 
                 reward_hunter_1:int,
                 penalty_hunter_1:int,
                 reward_hunter_2=None,
                 penalty_hunter_2=None):
        """
        initialize game and place prey and hunters on random positions

        :param playing_field_size: Size of game board (width, height).
        :param reward: Reward if the prey is caught.
        :param penalty: Score if the prey is NOT caught.
        :param dict_action_to_coord: maps actions to actual change in
            coordinates. the key is a integer, values are tuples of size 2
            (e.g. {0:(0,-1), 1:(1,0), 2:(0,1), 3:(-1,0), 4:(0,0)})

        :param prey_action_prob: Probability distribution for each action of
            the prey. Must match the number of keys in dict_action_to_coord
            (e.g. np.array([1/3,1/3,1/3,0,0])).
        """
        
        dict_action_to_coord = {MOVE_TOP: (0, -1), MOVE_RIGHT: (1, 0), MOVE_BOTTOM: (0, 1),
                            MOVE_LEFT: (-1, 0), MOVE_STAY: (0, 0)}

        prey_action_prob = np.array([0, 1 / 3, 1 / 3, 1 / 3, 0])
        
        self.x_max = playing_field_size[0]
        self.y_max = playing_field_size[1]
        self.reward_hunter_1 = reward_hunter_1
        self.penalty_hunter_1 = penalty_hunter_1
        if reward_hunter_2 == None: self.reward_hunter_2 = reward_hunter_1 
        else: self.reward_hunter_2 = reward_hunter_1
        if penalty_hunter_2 == None: self.penalty_hunter_2 = penalty_hunter_1
        else: self.penalty_hunter_2 = penalty_hunter_1
        self.dict_action_to_coord = dict_action_to_coord
        self.prey_action_prob = prey_action_prob
        self.prey_position, self.hunter_1_position, self.hunter_2_position = None, None, None
        self.reset_positions()

    def reset_positions(self):
        """
        Place the prey and hunters randomly in the playing field.
        """
        self.prey_position = np.array([np.random.randint(self.x_max), np.random.randint(self.y_max)])
        self.hunter_1_position = np.array([np.random.randint(self.x_max), np.random.randint(self.y_max)])
        self.hunter_2_position = np.array([np.random.randint(self.x_max), np.random.randint(self.y_max)])
        self.move_prey()  # Forbid that the prey start at the same position as the hunters

    def update_position(self, position: np.array, action: int) -> np.array:
        """ 
        Update the given position considering the action coordinates provided.

        :param position: The current x and y position [x,y].
        :param action: Key corresponding to the action as used in
            dict_action_to_coord for the action.

        :return: The updated position.
        """
        position = (position + self.dict_action_to_coord.get(action))

        position[0] = (self.x_max + position[0]) % self.x_max
        position[1] = (self.y_max + position[1]) % self.y_max

        return position

    def get_relative_locations(self):
        """
        Transform the hunters absolute positions to the positions
        relative to the prey.

        :return: The relative positions of the hunters.
        """

        def get_relative_location(prey_coord, hunter_coord, max_coord):
            dist = np.array([
                (prey_coord - hunter_coord),
                (prey_coord + max_coord - hunter_coord),
                (prey_coord - max_coord - hunter_coord)])
            min_index = np.argmin(np.abs(dist))  # get index of lowest abs distance
            return dist[min_index]  # return value of index

        rel_loc_hunter_1 = np.array([
            get_relative_location(self.prey_position[0], self.hunter_1_position[0], self.x_max),
            get_relative_location(self.prey_position[1], self.hunter_1_position[1], self.y_max)
        ])
        rel_loc_hunter_2 = np.array([
            get_relative_location(self.prey_position[0], self.hunter_2_position[0], self.x_max),
            get_relative_location(self.prey_position[1], self.hunter_2_position[1], self.y_max)
        ])

        return rel_loc_hunter_1, rel_loc_hunter_2

    # ADDED on 28/12 KE
    def get_state_hunter_1(self) -> State:
        """
        Return a state object with the current relative locations
        hunter 1.

        :return: The state with the current relative locations.
        """
        rel_loc_hunter_1, rel_loc_hunter_2 = self.get_relative_locations()
        return State(tuple(rel_loc_hunter_1), tuple(rel_loc_hunter_2))

    # ADDED on 28/12 KE
    def get_state_hunter_2(self) -> State:
        """
        Get a state object with the current relative locations for
        hunter 2 (relative positions are inverted).

        :return: The state object with the current relative locations.
        """
        rel_loc_hunter_1, rel_loc_hunter_2 = self.get_relative_locations()
        return State(tuple(rel_loc_hunter_2), tuple(rel_loc_hunter_1))

    def is_prey_caught(self, x1, y1, x2, y2):
        return x1 == 0 == x2 and abs(y1) == 1 == abs(y2) and np.sign(y1) != np.sign(y2)

    def compute_score(self) -> float:
        """
        Compute the score of the players.

        :return: The score of the players.
        """
        score_hunter_1 = self.penalty_hunter_1
        score_hunter_2 = self.penalty_hunter_2

        hunter_1_rel_pos, hunter_2_rel_pos = self.get_relative_locations()

        x1, y1 = hunter_1_rel_pos
        x2, y2 = hunter_2_rel_pos

        if self.is_prey_caught(x1, y1, x2, y2) or self.is_prey_caught(y1, x1, y2, x2):
            score_hunter_1 = self.reward_hunter_1
            score_hunter_2 = self.reward_hunter_2

        return score_hunter_1,score_hunter_2

    def move_prey(self):
        bad_position, new_position = True, None
        while bad_position:
            prey_action = np.random.choice(self.prey_action_prob.size, p=self.prey_action_prob)
            new_position = self.update_position(self.prey_position, prey_action)

            bad_position = np.array_equal(new_position, self.hunter_1_position) \
                           or np.array_equal(new_position, self.hunter_2_position)

        self.prey_position = new_position

    def play_one_episode(self, hunter_1_action: int, hunter_2_action: int) -> float:
        """
        Play one episode of the game.

        :param hunter_1_action: Action selected by hunter 1 (key as
            used in dict_action_to_coord for the action).
        :param hunter_2_action: action selected by hunter 2 (key as
            used in dict_action_to_coord for the action).

        :return: The score of the players.
        """

        self.hunter_1_position = self.update_position(self.hunter_1_position, hunter_1_action)
        self.hunter_2_position = self.update_position(self.hunter_2_position, hunter_2_action)

        self.move_prey()

        return self.compute_score()


#######################################################################################
# scenario to test class
#######################################################################################

def test():
    playing_field = (7, 7)
    dict_action_to_coord = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0, 0)}
    prey_action_prob = np.array([1 / 3, 1 / 3, 1 / 3, 0, 0])
    reward = 1
    penalty = -1

    test_game = Game(playing_field, reward, penalty, dict_action_to_coord, prey_action_prob)
    print()
    print(f"initial position prey: {test_game.prey_position}")
    print(f"initial position hunter 1: {test_game.hunter_1_position}")
    print(f"initial position hunter 2: {test_game.hunter_2_position}\n")
    print(test_game.get_relative_locations())

    for i in range(0, 8):
        hunter_1_action = 0
        hunter_2_action = 1
        print(test_game.play_one_episode(hunter_1_action, hunter_2_action))

        print(f"position prey: {test_game.prey_position}")
        print(f"position hunter 1: {test_game.hunter_1_position}")
        print(f"position hunter 2: {test_game.hunter_2_position}")
        print(f"relative positions hunters: {test_game.get_relative_locations()}")


if __name__ == "__main__":
    test()
