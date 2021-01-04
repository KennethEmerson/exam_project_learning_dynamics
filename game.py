import numpy as np
from agent.agent import State
from agent.move import *


def is_prey_caught_homogeneous(x1: int, y1: int, x2: int, y2: int) -> (bool, bool):
    """
    check if prey is caught using the homogeneous definition in the paper.
    both hunters should be at each side of the prey, vertically or horizontally.
    Thus both players get reward
    function needs to be injected as a parameter in game class 

    :param x1 (int): relative x position of hunter 1 vs prey
    :param y1 (int): relative y position of hunter 1 vs prey
    :param x2 (int): relative x position of hunter 2 vs prey
    :param y2 (int): relative y position of hunter 2 vs prey
    
    :Return (Bool,Bool): (has hunter 1 caught the prey,has hunter 2 caught the prey)
    """
    is_caught_hunter_1 = (x1 == 0 == x2 and abs(y1) == 1 == abs(y2) and np.sign(y1) != np.sign(y2)) or \
                         (y1 == 0 == y2 and abs(x1) == 1 == abs(x2) and np.sign(x1) != np.sign(x2))

    is_caught_hunter_2 = is_caught_hunter_1

    return is_caught_hunter_1, is_caught_hunter_2


def is_prey_caught_heterogeneous(x1: int, y1: int, x2: int, y2: int) -> (bool, bool):
    """
    check if prey is caught using the heterogeneous definition in the paper.
    both hunters should be at each side of the prey, vertically or horizontally.
    Then hunter 1 gets reward. Hunter 2 only gets a reward if the prey is caught and
    hunter 2 is at the left or bottom of the prey
    function needs to be injected as a parameter in game class 

    :param x1 (int): relative x position of hunter 1 vs prey
    :param y1 (int): relative y position of hunter 1 vs prey
    :param x2 (int): relative x position of hunter 2 vs prey
    :param y2 (int): relative y position of hunter 2 vs prey

    :Return (Bool,Bool): (has hunter 1 caught the prey,has hunter 2 caught the prey)
    """
    is_caught_hunter_1 = (x1 == 0 == x2 and abs(y1) == 1 == abs(y2) and np.sign(y1) != np.sign(y2)) or \
                         (y1 == 0 == y2 and abs(x1) == 1 == abs(x2) and np.sign(x1) != np.sign(x2))

    # hunter 2 must be on the left or the bottom (see paper page 6)
    # when prey is catched hunter 1 always gets reward, hunter 2 only gets reward
    # when it is on the left or bottom.
    # x1 and x2 are relative positions, if x2 == 1 then x2 is ont the left
    # same for y only coordinates here go downward
    is_caught_hunter_2 = is_caught_hunter_1 and (x2 == 1 or y2 == -1)

    return is_caught_hunter_1, is_caught_hunter_2


class Game:
    """
    Create a game playable step by step.
    """

    def __init__(self, playing_field_size: tuple,
                 reward_hunter_1: int,
                 penalty_hunter_1: int,
                 reward_hunter_2=None,
                 penalty_hunter_2=None,
                 is_prey_caught_function=is_prey_caught_homogeneous):
        """
        initialize game and place prey and hunters on random positions

        :param playing_field_size: Size of game board (width, height).
        :param reward_hunter_1: Reward for hunter 1 if the prey is caught.
        :param penalty_hunter_1: Score for hunter 1 if the prey is NOT caught.
        :param reward_hunter_2: Reward for hunter 2 if the prey is caught.
        :param penalty_hunter_2: Score for hunter 2 if the prey is NOT caught.
        :param is_prey_caught_function: function to define if the prey is caught (int,int,int,int) -> (bool,bool)
        """

        dict_action_to_coord = {MOVE_TOP: (0, -1), MOVE_RIGHT: (1, 0), MOVE_BOTTOM: (0, 1),
                                MOVE_LEFT: (-1, 0), MOVE_STAY: (0, 0)}

        prey_action_prob = np.array([0, 1 / 3, 1 / 3, 1 / 3, 0])

        self.dict_action_to_coord = dict_action_to_coord
        self.prey_action_prob = prey_action_prob

        self.x_max = playing_field_size[0]
        self.y_max = playing_field_size[1]

        self.reward_hunter_1 = reward_hunter_1
        self.penalty_hunter_1 = penalty_hunter_1

        if reward_hunter_2 == None:
            self.reward_hunter_2 = reward_hunter_1
        else:
            self.reward_hunter_2 = reward_hunter_2

        if penalty_hunter_2 == None:
            self.penalty_hunter_2 = penalty_hunter_1
        else:
            self.penalty_hunter_2 = penalty_hunter_2

        self.is_prey_caught = is_prey_caught_function

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

        is_caught_hunter_1, is_caught_hunter_2 = self.is_prey_caught(x1, y1, x2, y2)

        if is_caught_hunter_1: score_hunter_1 = self.reward_hunter_1
        if is_caught_hunter_2: score_hunter_2 = self.reward_hunter_2

        return score_hunter_1, score_hunter_2

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
    # standard homogeneous game identical rewards
    game = Game((7, 7), 1, 0)
    game.hunter_1_position = np.array([0, 0])
    game.hunter_2_position = np.array([4, 4])
    game.prey_position = np.array([2, 2])
    print(f"should be (0,0):{game.compute_score()}")

    game.hunter_1_position = np.array([0, 0])
    game.hunter_2_position = np.array([0, 2])
    game.prey_position = np.array([0, 1])
    print(f"should be (1,1):{game.compute_score()}")

    game.hunter_1_position = np.array([0, 0])
    game.hunter_2_position = np.array([2, 0])
    game.prey_position = np.array([1, 0])
    print(f"should be (1,1):{game.compute_score()}")

    # standard homogeneous game different rewards
    game = Game((7, 7), 1, 0, 2, -2)
    game.hunter_1_position = np.array([0, 0])
    game.hunter_2_position = np.array([4, 4])
    game.prey_position = np.array([2, 2])
    print(f"should be (0,-2):{game.compute_score()}")

    game.hunter_1_position = np.array([0, 0])
    game.hunter_2_position = np.array([0, 2])
    game.prey_position = np.array([0, 1])
    print(f"should be (1,2):{game.compute_score()}")

    game.hunter_1_position = np.array([0, 0])
    game.hunter_2_position = np.array([2, 0])
    game.prey_position = np.array([1, 0])
    print(f"should be (1,2):{game.compute_score()}")

    # heterogeneous game identical rewards
    game = Game((7, 7), 1, 0, is_prey_caught_function=is_prey_caught_heterogeneous)
    game.hunter_1_position = np.array([0, 0])
    game.hunter_2_position = np.array([4, 4])
    game.prey_position = np.array([2, 2])
    print(f"should be (0,0):{game.compute_score()}")

    game.hunter_1_position = np.array([0, 0])
    game.hunter_2_position = np.array([2, 0])
    game.prey_position = np.array([1, 0])
    print(f"should be (1,0):{game.compute_score()}")

    game.hunter_1_position = np.array([2, 0])
    game.hunter_2_position = np.array([0, 0])
    game.prey_position = np.array([1, 0])
    print(f"should be (1,1):{game.compute_score()}")

    game.hunter_1_position = np.array([0, 0])
    game.hunter_2_position = np.array([0, 2])
    game.prey_position = np.array([0, 1])
    print(f"should be (1,1):{game.compute_score()}")

    game.hunter_1_position = np.array([0, 2])
    game.hunter_2_position = np.array([0, 0])
    game.prey_position = np.array([0, 1])
    print(f"should be (1,0):{game.compute_score()}")


if __name__ == "__main__":
    test()
