import gym
from gym import spaces
from gym.utils import seeding

class DieN(gym.Env):
    """
    This game DieN MDP is made for solving the DieN problem in class CS7642-2020 Spring of GIT -Hui Xia

    The game DieN is played in the following way:
    1. You will be given a die with N sides. You will know the size of N , and can assume that N is
        a value greater than 1 and less than or equal to 30.
    2. You will be given a boolean mask vector isBadSide where the value of 1 indicates the
        sides of the die that will make you lose. The vector will be of size N , and 1 indexed. (there
        is no 0 side)
    3. You start with 0 dollars.
    4. At any time you have the option to roll the die or to quit the game
        a. If you decide to roll:
            i. And you roll a number not in isBadSide, you receive that many dollars. (eg.
                if you roll the number 2 and 2 is not a bad side -- meaning the second
                element of the vector is 0 -- in isBadSide , then you receive 2 dollars)
                Repeat step 4.
            ii. And you roll a number in isBadSide, then you lose all money obtained in
                previous rolls and the game ends.
        b. If you decide to quit:
            i. You keep all the money gained from previous rolls and the game ends.

    """
    def __init__(self,  n=6, isBadSide=[1, 1, 1, 0, 0, 0], slip=0):
        self.n = n                  # a die with N sides
        self.isBadSide = isBadSide  # A Boolean mask, value of 1 indicates the sides of the die that will make you lose.
        self.slip = slip  # probability of 'slipping' an action, by default equal to 0 in the homework
        self.state = 0              # State start at 0 dollars
        self.action_space = spaces.Discrete(2)  # roll dice to risk more(action=1) or quit & get all money (action=0)
        self.observation_space = spaces.Discrete(self.n)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if self.np_random.rand() < self.slip:
            action = not action  # agent slipped, reverse action taken
        if action:  # roll dice
            roll_result = int((np.random.randint(n, size=1)))   # roll dice, get a surface on dice

            if (isBadSide[roll_result]) == 0:    # value of 0 on isBadSide will make you win
                reward = roll_result + 1
                self.state += reward
                done = False
            else:  # value of 1 on isBadSide will make you lose all your money
                reward = -self.state
                self.state += reward
                done = True
        else:  # keep previously collected reward (i.e. keep self.state as-is) and quit
            reward = 0
            done = True

        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        return self.state