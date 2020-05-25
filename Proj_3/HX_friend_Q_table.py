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

class Friend_Q_table:

    def __init__(self,
                 # alpha=0.01, gamma=0.9, epsilon=0.9,
                 verbose=False):

        self.states = [0, 1, 2, 3, 4, 5, 6, 7]
        self.actions = [-4,-1,0,-1,4]
        self.S = SoccerEnv().S
        self.A = SoccerEnv().A
        self.q_table_A = pd.DataFrame(np.random.rand(len(self.S), len(self.A)),
                                      index=pd.MultiIndex.from_tuples(self.S),
                                      columns=pd.MultiIndex.from_tuples(self.A), dtype=np.float64)
        self.q_table_B = pd.DataFrame(np.random.rand(len(self.S), len(self.A)),
                                      index=pd.MultiIndex.from_tuples(self.S),
                                      columns=pd.MultiIndex.from_tuples(self.A), dtype=np.float64)
        self.verbose = verbose

    def choose_action(self, s, epsilon):

        if np.random.random() > epsilon:
            # print("self.q_table_A\n", self.q_table_A)
            # print("friend_Q.py, line 37, s", s)
            state_action_A = self.q_table_A.loc[s, :]
            state_action_B = self.q_table_B.loc[s, :]
            # print("friend_Q.py line 40, argmax, state_action_A")
            # print(state_action_A)
            # print("friend_Q.py line 42, argmax, state_action_B")
            # print(state_action_B)
            action_A = np.argmax(state_action_A)[0]        # expect best reward for the player A himself
            action_B = np.argmax(state_action_B)[1]        # expect best reward for the player B himself
            # print("friend_Q.py line 46, argmax, action A, action B", action_A, action_B)
        else:
            # choose random action

            action_A = np.random.choice(self.actions)
            action_B = np.random.choice(self.actions)

            # print("friend_Q.py line 46, random choose, action A, action B", action_A, action_B)
        return tuple([action_A, action_B])

    def learn(self, s, a, r, s_, a_, alpha, gamma):
        # a[0], a[1] are actions for players A and B
        # s[1], s[2] are states for players A and B. s[0] indicates the ball holding status.
        # print("two_play_SRSA.py, line 59, s, a", s, a)
        # print("two_play_SRSA.py, line 60, s_, a_", s_, a_)
        q_predict_A = self.q_table_A.loc[s, a]
        q_predict_B = self.q_table_B.loc[s, a]
        # print("friend_Q.py line 63, q_predict_A, q_predict_B", q_predict_A, q_predict_B)
        # print("friend_Q.py line 64, r, r[0]", r, r[0])
        if 0 in r:
            q_target_A = r[0] + gamma * self.q_table_A.loc[s_, a_]  # next state is not terminal
            q_target_B = r[1] + gamma * self.q_table_B.loc[s_, a_]  # next state is not terminal
            # print("friend_Q.py line 61, self.q_table_A.loc[s_, a_[0]] ", self.q_table_A.loc[s_, a_])
        else:
            q_target_A = r[0]  # next state is terminal
            q_target_B = r[1]  # next state is terminal

        self.q_table_A.loc[s, a] += alpha * (q_target_A - q_predict_A)
        self.q_table_B.loc[s, a] += alpha * (q_target_B - q_predict_B)
        # print("friend_Q.py line 67, q_target_A ",  q_target_A)
        # print("friend_Q.py line 68, q_predict_A",  q_predict_A)
        # print(" ")
        # print(" ")
        # print("friend_Q.py line 68, q_table_A", self.q_table_A)

        if self.verbose >= 2:
            # print("s,a,r,s_,a_:", s,a,r,s_,a_)
            print('\n Q table A is:\n', self.q_table_A)
            print('\n Q table B is:\n', self.q_table_A)
            pass

    def return_Q_table(self):
        return self.q_table_A, self.q_table_B


