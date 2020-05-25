"""
The original single player version of this SARSA modified from morvan zhou's Q learner table
View more on morvan tutorial page: https://morvanzhou.github.io/tutorials/

For this code file, i.e. the double player version of this SARSA also referred from Ayazhan:
https://github.com/Ayazhan/Correlated-Q-Learning

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
        self.actions = [-4,-1,0,-1,4]
        self.S = SoccerEnv().S
        self.A = SoccerEnv().A
        self.q_table_A = pd.DataFrame(0, index=pd.MultiIndex.from_tuples(self.S),
                                      columns=self.actions, dtype=np.float64)  # <=
        self.q_table_B = pd.DataFrame(0, index=pd.MultiIndex.from_tuples(self.S),
                                      columns=self.actions, dtype=np.float64)  # <=
        self.verbose = verbose

    def choose_action(self, s, epsilon):

        if np.random.random() > epsilon:
            # print("self.q_table_A\n", self.q_table_A)
            # print("two_player_srasa.py, line 33, s[1]\n", s[1])
            state_action_A = self.q_table_A.loc[s, :]               # <=
            state_action_B = self.q_table_B.loc[s, :]               # <=
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
        action_ind_A = [-4, -1, 0, 1, 4].index(a[0])    # weird bug, if I do q_table_A.loc[s, -1], it returns a list..
        action_ind_B = [-4, -1, 0, 1, 4].index(a[1])    # to fix the crazy bug I have to call this dataframe with
        action_ind_A_ = [-4, -1, 0, 1, 4].index(a_[0])  # both iloc and loc
        action_ind_B_ = [-4, -1, 0, 1, 4].index(a_[1])
        # print("two_play_SRSA.py, line 48, s, a, s, a_", s, a, s_, a_)
        # print("two_play_SARSA.py, line 51, self.q_table_A", type(self.q_table_A))
        q_predict_B = self.q_table_B.loc[s, :].iloc[action_ind_B]                # <=
        q_predict_A = self.q_table_A.loc[s, :].iloc[action_ind_A]               # <=

        # print("two player SARSA table line 55, r, r[0], gamma", r, r[0], r[1], gamma)
        # print("two player SARSA table line 56, self.q_table_A.shape(), s_, a_[0]", self.q_table_A.shape, s_, a_[0])
        # print("two player SARSA table line 57, self.q_table_A.loc[s_, :]", self.q_table_A.loc[s_, :].iloc[action_ind_A_])
        if 0 in r:
            q_target_B = r[1] + gamma * self.q_table_B.loc[s_, :].iloc[action_ind_B_]  # <=
            q_target_A = r[0] + gamma * self.q_table_A.loc[s_, :].iloc[action_ind_A_]  # <=

        else:
            q_target_B = r[1]  # next state is terminal
            q_target_A = r[0]  # next state is terminal
        # print("two_play_SARSA.py, line 63, q_target_A", q_target_A)
        # print("two_play_SARSA.py, line 64, q_predict_A", q_predict_A)
        self.q_table_B.loc[s, :].iloc[action_ind_B] += alpha * (q_target_B - q_predict_B)   # <=
        self.q_table_A.loc[s, :].iloc[action_ind_A] += alpha * (q_target_A - q_predict_A)   # <=


        if self.verbose >= 2:
            # print("s,a,r,s_,a_:", s,a,r,s_,a_)
            print('\n Q table A is:\n', self.q_table_A)
            print('\n Q table B is:\n', self.q_table_A)
            pass

    def return_Q_table(self):
        return self.q_table_A, self.q_table_B



