import unittest
import numpy as np
from game import Game


#########################################################################################
# generic tests
#########################################################################################

def test_init_position(test_case: unittest.TestCase, participant_position: np.array):
    """
    Test if the initial position of the participant is within game limits.

    :param test_case: The test case.
    :param participant_position: Array with participant coordinates.
    """
    test_case.assertTrue(participant_position[0] < test_case.playing_field[0])
    test_case.assertTrue(participant_position[0] >= 0)
    test_case.assertTrue(participant_position[0] < test_case.playing_field[1])
    test_case.assertTrue(participant_position[0] >= 0)


#########################################################################################

def set_positions(game, prey_position, hunter_1_position, hunter_2_position):
    """
    Set positions of participants.
    """
    game.prey_position = np.array(prey_position)
    game.hunter_1_position = np.array(hunter_1_position)
    game.hunter_2_position = np.array(hunter_2_position)


#########################################################################################
# test case
#########################################################################################


class TestGame(unittest.TestCase):

    def setUp(self):
        """
        Setup a standard game for every test
        """
        self.playing_field = (7, 7)
        self.x_max = self.playing_field[0]
        self.y_max = self.playing_field[1]
        dict_action_to_coord = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0, 0)}
        pray_action_prob = np.array([1 / 3, 1 / 3, 1 / 3, 0, 0])
        self.reward = 1
        self.penalty = -1

        self.game = Game(self.playing_field, self.reward, self.penalty, dict_action_to_coord, pray_action_prob)

        #########################################################################################

    # test 1
    def test_init_prey_position(self):
        """ Test if prey is placed on a valid spot on the board """
        test_init_position(self, self.game.prey_position)

    # test 2
    def test_init_hunter_1_position(self):
        """ Test if hunter_1 is placed on a valid spot on the board """
        test_init_position(self, self.game.hunter_1_position)

    # test 3
    def test_init_hunter_2_position(self):
        """ Test if hunter_2 is placed on a valid spot on the board """
        test_init_position(self, self.game.hunter_2_position)

    #########################################################################################
    # test 4
    def test_get_relative_locations_1(self):
        """
        Test if the shortest relative distance is returned.
        """
        set_positions(self.game, [3, 3], [1, 2], [6, 5])
        pos_hunter_1, pos_hunter_2 = self.game.get_relative_locations()
        np.testing.assert_array_equal(pos_hunter_1, np.array([2, 1]))
        np.testing.assert_array_equal(pos_hunter_2, np.array([-3, -2]))

    # test 5
    def test_get_relative_locations_2(self):
        """
        Test if the shortest relative distance is returned.
        """
        set_positions(self.game, [0, 0], [0, 6], [5, 0])
        pos_hunter_1, pos_hunter_2 = self.game.get_relative_locations()
        np.testing.assert_array_equal(pos_hunter_1, np.array([0, 1]))
        np.testing.assert_array_equal(pos_hunter_2, np.array([2, 0]))

    # test 6
    def test_get_relative_locations_3(self):
        """
        Test if the shortest relative distance is returned.
        """
        set_positions(self.game, [5, 6], [0, 1], [1, 0])
        pos_hunter_1, pos_hunter_2 = self.game.get_relative_locations()
        np.testing.assert_array_equal(pos_hunter_1, np.array([-2, -2]))
        np.testing.assert_array_equal(pos_hunter_2, np.array([-3, -1]))

    #########################################################################################

    # test 7
    def test_check_score_penalty_1(self):
        """
        Test if penalty is returned when prey is not captured.
        """
        set_positions(self.game, [1, 1], [0, 0], [2, 2])
        self.assertEqual(self.game.compute_score(), self.penalty)
        set_positions(self.game, [1, 1], [2, 2], [0, 0])
        self.assertEqual(self.game.compute_score(), self.penalty)
        set_positions(self.game, [1, 1], [1, 0], [2, 2])
        self.assertEqual(self.game.compute_score(), self.penalty)

    # test 8
    def test_check_reward_horizontal(self):
        """
        Test if reward is returned when prey is captured horizontally.
        """
        set_positions(self.game, [1, 1], [1, 0], [1, 2])
        for i in range(0, self.x_max * 2):
            self.game.prey_position[0] += 1
            self.game.hunter_1_position[0] += 1
            self.game.hunter_2_position[0] += 1

            for j in range(0, self.y_max * 2):
                self.game.prey_position[1] += 1
                self.game.hunter_1_position[1] += 1
                self.game.hunter_2_position[1] += 1
                self.assertEqual(self.game.compute_score(), 1)

    # test 9
    def test_check_reward_vertical(self):
        """
        Test if reward is returned when prey is captured vertically.
        """
        set_positions(self.game, [1, 1], [0, 1], [2, 1])
        for i in range(0, self.x_max * 2):
            self.game.prey_position[0] += 1
            self.game.hunter_1_position[0] += 1
            self.game.hunter_2_position[0] += 1

            for j in range(0, self.y_max * 2):
                self.game.prey_position[1] += 1
                self.game.hunter_1_position[1] += 1
                self.game.hunter_2_position[1] += 1
                self.assertEqual(self.game.compute_score(), 1)


if __name__ == '__main__':
    unittest.main()
