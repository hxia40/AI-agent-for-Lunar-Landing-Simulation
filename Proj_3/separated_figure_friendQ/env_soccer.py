import sys
from contextlib import closing

import numpy as np
from six import StringIO, b

import gym
from gym import utils
from gym.envs.toy_text import discrete
from gym import spaces
from gym.utils import seeding

actions = [-4,4,1,-1,0]

class SoccerEnv(gym.Env):
    """
    The soccer game is from A. Greenwald and K. Hall 2002, which is a simplified version of Littman 1994.
    It is a zero-sum two-player game for which there do not exist deterministic equilibrium policies.

    The world is a 2 x 4 grid world:

    0 1 2 3
    4 5 6 7

    2 : Player A's starting point
    1 : Player B's starting point
    0 and 4 : Player A's goal. Player A will score + 100 (and player B -100) if a player with ball move to this point.
    3 and 7 : Player B's goal. Player B will score + 100 (and player A -100) if a player with ball move to this point.
    Other points are flat ground, all players can move through

    There are two players, A and B, whose possible actions are N (-4), S (+4), E (+1), W (-1), and stick (+0).
    The agents' actions are executed in random order. Player B start on point B, and player A start on point A.

    A ball will be randomly generated to one of the players. When a player with ball reach points b, the game ends,
    player B get a reward of 100, and player A gets a reward of -100.

    Visa versa, or when a player with ball reaches points a, the game ends, player A get a reward of 100, and
    player B get a reward of -100.

    If this sequence of actions causes the players to collide, then neither moves. But if the player with the ball moves
    second, then the ball changes possession.

    This env is based on OpenAI Gym, which typically does not typically support two-player game. Thus, we can treat the
    two-player game as one sinle player, by updating the SARSA in a modified manner. That is, instead of updating
    (s, a, r, s, a_) for one player, we can update (s, a1, a2, r1, r2, s1, s2, a1_, a2_) in this algorithm, then
    argmax each of the player's utility.

    """

    def __init__(self):
        # Thus,we shall consider the state as the position of both players, wich means these are 8 * 7 = 56 different
        # cases.
        # Considering that either player can have the ball, the total number of the states will be 56 * 2 = 112, the
        # first value in S indicates who has the ball. 0 = player A has the ball, 1 = player B has the ball. The second
        # and third value in S indicates the location of players A and B, respectively.

        self.S = [(0, 0, 1),
              (0, 0, 2),
              (0, 0, 3),
              (0, 0, 4),
              (0, 0, 5),
              (0, 0, 6),
              (0, 0, 7),
              (0, 1, 0),
              (0, 1, 2),
              (0, 1, 3),
              (0, 1, 4),
              (0, 1, 5),
              (0, 1, 6),
              (0, 1, 7),
              (0, 2, 0),
              (0, 2, 1),
              (0, 2, 3),
              (0, 2, 4),
              (0, 2, 5),
              (0, 2, 6),
              (0, 2, 7),
              (0, 3, 0),
              (0, 3, 1),
              (0, 3, 2),
              (0, 3, 4),
              (0, 3, 5),
              (0, 3, 6),
              (0, 3, 7),
              (0, 4, 0),
              (0, 4, 1),
              (0, 4, 2),
              (0, 4, 3),
              (0, 4, 5),
              (0, 4, 6),
              (0, 4, 7),
              (0, 5, 0),
              (0, 5, 1),
              (0, 5, 2),
              (0, 5, 3),
              (0, 5, 4),
              (0, 5, 6),
              (0, 5, 7),
              (0, 6, 0),
              (0, 6, 1),
              (0, 6, 2),
              (0, 6, 3),
              (0, 6, 4),
              (0, 6, 5),
              (0, 6, 7),
              (0, 7, 0),
              (0, 7, 1),
              (0, 7, 2),
              (0, 7, 3),
              (0, 7, 4),
              (0, 7, 5),
              (0, 7, 6),
              (1, 0, 1),
              (1, 0, 2),
              (1, 0, 3),
              (1, 0, 4),
              (1, 0, 5),
              (1, 0, 6),
              (1, 0, 7),
              (1, 1, 0),
              (1, 1, 2),
              (1, 1, 3),
              (1, 1, 4),
              (1, 1, 5),
              (1, 1, 6),
              (1, 1, 7),
              (1, 2, 0),
              (1, 2, 1),
              (1, 2, 3),
              (1, 2, 4),
              (1, 2, 5),
              (1, 2, 6),
              (1, 2, 7),
              (1, 3, 0),
              (1, 3, 1),
              (1, 3, 2),
              (1, 3, 4),
              (1, 3, 5),
              (1, 3, 6),
              (1, 3, 7),
              (1, 4, 0),
              (1, 4, 1),
              (1, 4, 2),
              (1, 4, 3),
              (1, 4, 5),
              (1, 4, 6),
              (1, 4, 7),
              (1, 5, 0),
              (1, 5, 1),
              (1, 5, 2),
              (1, 5, 3),
              (1, 5, 4),
              (1, 5, 6),
              (1, 5, 7),
              (1, 6, 0),
              (1, 6, 1),
              (1, 6, 2),
              (1, 6, 3),
              (1, 6, 4),
              (1, 6, 5),
              (1, 6, 7),
              (1, 7, 0),
              (1, 7, 1),
              (1, 7, 2),
              (1, 7, 3),
              (1, 7, 4),
              (1, 7, 5),
              (1, 7, 6)]
        self.R = [(100., -100.),
              (100., -100.),
              (100., -100.),
              (100., -100.),
              (100., -100.),
              (100., -100.),
              (100., -100.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (-100., 100.),
              (-100., 100.),
              (-100., 100.),
              (-100., 100.),
              (-100., 100.),
              (-100., 100.),
              (-100., 100.),
              (100., -100.),
              (100., -100.),
              (100., -100.),
              (100., -100.),
              (100., -100.),
              (100., -100.),
              (100., -100.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (0., 0.),
              (-100., 100.),
              (-100., 100.),
              (-100., 100.),
              (-100., 100.),
              (-100., 100.),
              (-100., 100.),
              (-100., 100.),
              (0., 0.),
              (0., 0.),
              (-100., 100.),
              (100., -100.),
              (0., 0.),
              (0., 0.),
              (-100., 100.),
              (100., -100.),
              (0., 0.),
              (-100., 100.),
              (100., -100.),
              (0., 0.),
              (0., 0.),
              (-100., 100.),
              (100., -100.),
              (0., 0.),
              (-100., 100.),
              (100., -100.),
              (0., 0.),
              (0., 0.),
              (-100., 100.),
              (100., -100.),
              (0., 0.),
              (0., 0.),
              (100., -100.),
              (0., 0.),
              (0., 0.),
              (-100., 100.),
              (100., -100.),
              (0., 0.),
              (0., 0.),
              (-100., 100.),
              (0., 0.),
              (0., 0.),
              (-100., 100.),
              (100., -100.),
              (0., 0.),
              (0., 0.),
              (-100., 100.),
              (100., -100.),
              (0., 0.),
              (-100., 100.),
              (100., -100.),
              (0., 0.),
              (0., 0.),
              (-100., 100.),
              (100., -100.),
              (0., 0.),
              (-100., 100.),
              (100., -100.),
              (0., 0.),
              (0., 0.),
              (-100., 100.),
              (100., -100.),
              (0., 0.),
              (0., 0.)]
        self.individual_A = [-4,4,1,-1,0]
        self.nS = 112
        self.reward_range = (-100, 100)
        # These should be five actions for each player, which is go N (-4), S (+4), E (+1), W (-1), and stick (+0 i.e.
        # does not move). All combination of actions of the two players are:
        self.A = [(-4, -4),
              (4, -4),
              (1, -4),
              (-1, -4),
              (0, -4),
              (-4, 4),
              (4, 4),
              (1, 4),
              (-1, 4),
              (0, 4),
              (-4, 1),
              (4, 1),
              (1, 1),
              (-1, 1),
              (0, 1),
              (-4, -1),
              (4, -1),
              (1, -1),
              (-1, -1),
              (0, -1),
              (-4, 0),
              (4, 0),
              (1, 0),
              (-1, 0),
              (0, 0)]
        self.nA = 5

        # Other common parameters for the environments:
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Discrete(112)
        self.seed()

        # WHen restarting, reset bith players' positions with random distribution of ball.
        rdm_init = np.random.rand()
        if rdm_init <= 0.5:
            self.state = [0, 2, 1]
        else:
            self.state = [1, 2, 1]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        ball, me, you = self.state
        action = list(action)
        # Find the actions for both players, define that neither players can go out of the board
        # print("env_soccer.py line 334, action", action)
        if (me + action[0]) > 7 or (me + action[0]) < 0:
            action[0] = 0
        if (you + action[1]) > 7 or (you + action[1]) < 0:
            action[1] = 0

        mynewstate = me + action[0]
        yournewstate = me + action[1]

        s_ind = None
        for i in range(self.nS):
            if np.array_equal(self.S[i], self.state):
                s_ind = i
                # print(342, "i", i)
                # print(343, "self.S[i]", self.S[i])
                # print(344, "self.tate", self.state)
        # print("env_soccer.py, line 345, s_ind", s_ind)
        # Generating new state, take care of the case that players collide into each other
        if mynewstate == yournewstate:
            first = np.random.randint(2)
            if first == 0:  # I'm first
                if ball == 0:  # I have a ball
                    s_prime = [ball, me + action[0], you]  # I move, you don't, ball is still mine
                else:
                    s_prime = [ball - 1, me + action[0], you]  # i move, you bump into me, I get your ball
            else:  # you move first
                if ball == 1:  # you have the ball
                    s_prime = [ball, me, you + action[1]]  # you move, you keep the ball, i stay
                else:  # i have the ball
                    s_prime = [ball - 1, me, you + action[1]]  # you move, I bump into you, you get the ball
        else:
            s_prime = [ball, me + action[0], you + action[1]]
        # If players didn't bump into each other, we will adopt the new self.state. Else they will keep old positions
        if (s_prime[1] != s_prime[2]) and (s_prime[0] != -1):
            self.state = np.array(s_prime)

        # Generating reward and done: get index of the new state, then track on the reward table.
        for i in range(self.nS):
            if np.array_equal(self.S[i], self.state):
                s_ind = i
                # print(361, "i", i)
                # print(362, "self.S[i]", self.S[i])
                # print(363, "self.tate", self.state)

        # print("env_soccer.py line 372 s_ind", s_ind, self.state)
        reward = self.R[s_ind]
        # print("env_soccer.py line 363, reward\n", reward)
        if 0 in reward:
            done = False
        else:
            done = True
        # print("env_soccer.py line 381, self.state", self.state)
        # print("env_soccer.py line 382, done", done)

        return tuple(self.state), reward, done, {}        # <=
        # return s_ind, reward, done, {}

    def reset(self):
        # WHen restarting, reset bith players' positions with random distribution of ball.
        rdm_reset = np.random.rand()
        if rdm_reset <= 0.5:
            self.state = [0, 2, 1]
        else:
            self.state = [1, 2, 1]
        return tuple(self.state)             # <=
        # if rdm_reset <= 0.5:
        #     s_ind = 15
        # else:
        #     s_ind = 71
        # return s_ind



