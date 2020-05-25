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
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
from env_soccer import SoccerEnv


class Foe_Q_table:

    def __init__(self,
                 # alpha=0.01, gamma=0.9, epsilon=0.9,
                 verbose=False):

        self.states = [0, 1, 2, 3, 4, 5, 6, 7]
        self.actions = [-4,-1,0,-1,4]
        self.S = SoccerEnv().S
        self.A = SoccerEnv().A
        self.q_table_A = pd.DataFrame(1,
                                      index=pd.MultiIndex.from_tuples(self.S),
                                      columns=pd.MultiIndex.from_tuples(self.A), dtype=np.float64)
        self.q_table_B = pd.DataFrame(1,
                                      index=pd.MultiIndex.from_tuples(self.S),
                                      columns=pd.MultiIndex.from_tuples(self.A), dtype=np.float64)
        self.verbose = verbose
        self.vi_A = pd.DataFrame(1,
                                 index=pd.MultiIndex.from_tuples(self.S),
                                 columns=['value'], dtype=np.float64)
        self.vi_B = pd.DataFrame(1,
                                 index=pd.MultiIndex.from_tuples(self.S),
                                 columns=['value'], dtype=np.float64)

    def maxmin(self, A,
               solver="glpk"
               ):  # by Adam Novotny
        num_vars = len(A)
        # minimize matrix c
        c = [-1] + [0 for i in range(num_vars)]
        c = np.array(c, dtype="float")
        c = matrix(c)
        # constraints G*x <= h
        G = np.matrix(A, dtype="float").T  # reformat each variable is in a row
        G *= -1  # minimization constraint
        G = np.vstack([G, np.eye(num_vars) * -1])  # > 0 constraint for all vars
        new_col = [1 for i in range(num_vars)] + [0 for i in range(num_vars)]
        G = np.insert(G, 0, new_col, axis=1)  # insert utility column
        G = matrix(G)
        h = ([0 for i in range(num_vars)] +
             [0 for i in range(num_vars)])
        h = np.array(h, dtype="float")
        h = matrix(h)
        # contraints Ax = b
        A = [0] + [1 for i in range(num_vars)]
        A = np.matrix(A, dtype="float")
        A = matrix(A)
        b = np.matrix(1, dtype="float")
        b = matrix(b)
        sol = solvers.lp(c=c, G=G, h=h, A=A, b=b
                         , solver=solver, options={'glpk':{'msg_lev':'GLP_MSG_OFF'}}
                         )
        return sol


    def choose_action(self, s, epsilon):

        if np.random.random() > epsilon:
            # print("self.q_table_A\n", self.q_table_A)
            # print("two_player_srasa.py, line 33, s", s)
            state_action_A = self.q_table_A.loc[s, :]
            state_action_B = self.q_table_B.loc[s, :]
            # print("friend_Q.py line 40, argmax, state_action_A, state_action_B", state_action_A, state_action_B)
            # print("foeQ line 80, state_action_A, shape")
            # print(state_action_A.shape)
            # print(state_action_A)
            state_action_A = state_action_A.values.reshape(5, 5)
            state_action_B = state_action_B.values.reshape(5, 5)
            # print("foeQ line 83, state_action_A reshaped, shape")
            # print(state_action_A.shape)
            # print(state_action_A)
            solA = self.maxmin(state_action_A)        # choose the maxmin from Q table
            solB = self.maxmin(state_action_B)       # choose the maxmin from Q table

            prob_A = np.abs(np.array(solA['x'][1:]).reshape((5, -1))) / sum(np.abs(solA['x'][1:]))
            prob_B = np.abs(np.array(solB['x'][1:]).reshape((5, -1))) / sum(np.abs(solB['x'][1:]))
            # print("foe_Q line 96, self.q_table_A.loc[s, :])", self.q_table_A.loc[s, :])

            # q_stack_A = self.q_table_A.loc[s, :].swaplevel().mean(axis = 0, level = 0)
            # print("foe_Q line 99, q_stack_A", q_stack_A)

            ind_A = int(np.random.choice(np.arange(5), 1, p=prob_A.flatten()))
            ind_B = int(np.random.choice(np.arange(5), 1, p=prob_B.flatten()))
            # print("foe_Q.py line 102, minmax, ind_A", ind_A)
            action_A = self.actions[ind_A]
            action_B = self.actions[ind_B]

            # print("foe_Q.py line 106, minmax, action A, action B", action_A, action_B)
        else:
            # choose random action
            action_A = np.random.choice(self.actions)
            action_B = np.random.choice(self.actions)
            # print("friend_Q.py line 96, random choose, action A, action B", action_A, action_B)
        return tuple([action_A, action_B])

    def learn(self, s, a, r, s_, a_, alpha, gamma):
        # a[0], a[1] are actions for players A and B
        # s[1], s[2] are states for players A and B. s[0] indicates the ball holding status.
        # print("foe.py, line 110, s, a", s, a)
        # print("foe.py, line 111, s_, a_", s_, a_)

        q_predict_A = self.q_table_A.loc[s, a]
        q_predict_B = self.q_table_B.loc[s, a]
        # print("two player SARSA table line 49, r, r[0]", r, r[0])
        if 0 in r:
            q_target_A = r[0] + gamma * self.vi_A.loc[s_, :].value # next state is not terminal
            q_target_B = r[1] + gamma * self.vi_B.loc[s_, :].value  # next state is not terminal
            # print("friend_Q.py line 61, self.q_table_A.loc[s_, a_[0]] ", self.q_table_A.loc[s_, a_])
        else:
            q_target_A = r[0]  # next state is terminal
            q_target_B = r[1]  # next state is terminal

        self.q_table_A.loc[s, a] += alpha * (q_target_A - q_predict_A)
        self.q_table_B.loc[s, a] += alpha * (q_target_B - q_predict_B)

        # update vi table
        state_action_A = self.q_table_A.loc[s, :]
        state_action_B = self.q_table_B.loc[s, :]

        state_action_A = state_action_A.values.reshape(5, 5)
        state_action_B = state_action_B.values.reshape(5, 5)

        solA = self.maxmin(state_action_A)  # choose the maxmin from Q table
        solB = self.maxmin(state_action_B)  # choose the maxmin from Q table

        prob_A = np.abs(np.array(solA['x'][1:]).reshape((5, -1))) / sum(np.abs(solA['x'][1:]))
        prob_B = np.abs(np.array(solB['x'][1:]).reshape((5, -1))) / sum(np.abs(solB['x'][1:]))

        self.vi_A.loc[s, :] = np.sum(prob_A.flatten() * self.q_table_A.loc[s, :].mean(axis=0, level=0))
        self.vi_B.loc[s, :] = np.sum(prob_B.flatten() * self.q_table_B.loc[s, :].swaplevel().mean(axis=0, level=0))

        if self.verbose >= 2:
            # print("s,a,r,s_,a_:", s,a,r,s_,a_)
            print('\n Q table A is:\n', self.q_table_A)
            print('\n Q table B is:\n', self.q_table_A)
            pass

    def return_Q_table(self):
        return self.q_table_A, self.q_table_B


