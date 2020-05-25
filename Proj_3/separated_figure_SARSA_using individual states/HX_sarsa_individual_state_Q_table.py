"""
The origin of this code from morvan zhou's Q learner table
View more on morvan tutorial page: https://morvanzhou.github.io/tutorials/

For this code file, i.e. the double player version of this friendly-Q also referred from Ayazhan:
https://github.com/Ayazhan/Correlated-Q-Learning

Being a friendly Q, each of the player have to assume that their oppoent will pick

"""

import numpy as np
import pandas as pd
import time
from env_soccer import SoccerEnv

class Two_player_SARSA_table:

    def __init__(self,
                 # alpha=0.01, gamma=0.9, epsilon=0.9,
                 verbose=False):

        self.states = [0, 1, 2, 3, 4, 5, 6, 7]
        self.actions = [-4,-1,0,1,4]
        self.S = SoccerEnv().S
        self.A = SoccerEnv().A
        self.q_table_A = pd.DataFrame(0, index=self.states, columns= self.actions, dtype=np.float64)
        self.q_table_B = pd.DataFrame(0, index=self.states, columns=self.actions, dtype=np.float64)
        self.verbose = verbose

    def choose_action(self, s, epsilon):

        if np.random.random() > epsilon:
            # print("self.q_table_A\n", self.q_table_A)
            # print("two_player_srasa.py, line 33, s[1]\n", s[1])
            state_action_A = self.q_table_A.loc[s[1], :]
            state_action_B = self.q_table_B.loc[s[2], :]

            action_A = np.argmax(state_action_A)
            action_B = np.argmax(state_action_B)
        else:
            # choose random action
            action_A = np.random.choice(self.actions)
            action_B = np.random.choice(self.actions)

        return [action_A, action_B]

    def learn(self, s, a, r, s_, a_, alpha, gamma):
        # a[0], a[1] are actions for players A and B
        # s[1], s[2] are states for players A and B. s[0] indicates the ball holding status.
        # print("two_play_SRSA.py, line 48, s, a", s, a)
        q_predict_A = self.q_table_A.loc[s[1], a[0]]
        q_predict_B = self.q_table_B.loc[s[2], a[1]]
        # print("two player SARSA table line 49, r, r[0]", r, r[0])
        if 0 in r:
            q_target_A = r[0] + gamma * self.q_table_A.loc[s_[0], a_[0]]  # next state is not terminal
            q_target_B = r[1] + gamma * self.q_table_B.loc[s_[1], a_[1]]  # next state is not terminal
        else:
            q_target_A = r[0]  # next state is terminal
            q_target_B = r[1]  # next state is terminal

        self.q_table_A.loc[s[1], a[0]] += alpha * (q_target_A - q_predict_A)
        self.q_table_B.loc[s[2], a[1]] += alpha * (q_target_B - q_predict_B)

        if self.verbose >= 2:
            # print("s,a,r,s_,a_:", s,a,r,s_,a_)
            print('\n Q table A is:\n', self.q_table_A)
            print('\n Q table B is:\n', self.q_table_A)
            pass

    def return_Q_table(self):
        return self.q_table_A, self.q_table_B


