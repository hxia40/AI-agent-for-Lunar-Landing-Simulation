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


class CE_Q_table:

    def __init__(self,
                 # alpha=0.01, gamma=0.9, epsilon=0.9,
                 verbose=False):

        self.states = [0, 1, 2, 3, 4, 5, 6, 7]
        self.actions = [-4,-1,0,-1,4]
        self.S = SoccerEnv().S
        self.A = SoccerEnv().A
        self.q_table_A = pd.DataFrame(1,
                                      # np.random.rand(len(self.S), len(self.A)),
                                      index=pd.MultiIndex.from_tuples(self.S),
                                      columns=pd.MultiIndex.from_tuples(self.A), dtype=np.float64)
        self.q_table_B = pd.DataFrame(1,
                                      # np.random.rand(len(self.S), len(self.A)),
                                      index=pd.MultiIndex.from_tuples(self.S),
                                      columns=pd.MultiIndex.from_tuples(self.A), dtype=np.float64)
        self.verbose = verbose
        self.vi_A = pd.DataFrame(1,
                                 index=pd.MultiIndex.from_tuples(self.S),
                                 columns=['value'], dtype=np.float64)
        self.vi_B = pd.DataFrame(1,
                                 index=pd.MultiIndex.from_tuples(self.S),
                                 columns=['value'], dtype=np.float64)

    def ce(self, A, solver=None):
        # Both build_ce_constraints and this function require the reward matrix of both players to be a nested list.e.g.
        # for a chicken game that each player has two options, a expected A should be like:
        # B = [[6, 6], [2, 7], [7, 2], [0, 0]]
        # thus, here we will convert our 5x5 matrix which only have one player's utility (only have one player's utility
        # because that we are doing zero-sum game here) to a nested list that has a length of 25
        A = A.reshape(1, -1).flatten()
        # print("ce_Q.py line 49, A", A)
        A = [[p, -p] for p in A]
        # print("ce_Q.py line 51, new A", A)

        num_vars = len(A)
        # print("ce_Q.py line 54, num_vars", num_vars)
        # maximize matrix c
        c = [sum(i) for i in A]  # sum of payoffs for both players. should always be list of 0 in zero-sum games.
        c = np.array(c, dtype="float")
        # print("ce_Q.py line 58, c", c)
        c = matrix(c)
        c *= -1  # cvxopt minimizes so *-1 to maximize
        # constraints G*x <= h
        G = self.build_ce_constraints(A=A)
        # print("ce_Q.py line 62, G", G.shape)
        # print("ce_Q.py line 63, np.eye(num_vars)", np.eye(num_vars).shape)
        G = np.vstack([G, np.eye(num_vars) * -1])  # > 0 constraint for all vars
        h_size = len(G)
        G = matrix(G)
        h = [0 for i in range(h_size)]
        h = np.array(h, dtype="float")
        h = matrix(h)
        # contraints Ax = b
        A = [1 for i in range(num_vars)]
        A = np.matrix(A, dtype="float")
        A = matrix(A)
        b = np.matrix(1, dtype="float")
        b = matrix(b)
        sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)
        return sol

    def build_ce_constraints(self, A):

        num_vars = int(len(A) ** 0.5)
        # print("ce_Q.py line 83, A", A)
        # print("ce_Q.py line 84, num_vars", num_vars)
        G = []
        # row player
        for i in range(num_vars):  # action row i
            for j in range(num_vars):  # action row j
                if i != j:
                    constraints = [0 for m in A]
                    base_idx = i * num_vars
                    comp_idx = j * num_vars
                    for k in range(num_vars):
                        constraints[base_idx + k] = (- A[base_idx + k][0]
                                                     + A[comp_idx + k][0])
                    # print("ce_Q.py line 96, constraints", len(constraints), constraints)
                    G += [constraints]
                    # print("ce_Q.py line 98, G", len(G), G)
        # col player
        for i in range(num_vars):  # action column i
            for j in range(num_vars):  # action column j
                if i != j:
                    constraints = [0 for n in A]
                    for k in range(num_vars):
                        constraints[i + (k * num_vars)] = (
                                - A[i + (k * num_vars)][1]
                                + A[j + (k * num_vars)][1])
                    G += [constraints]
        return np.matrix(G, dtype="float")

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
            solA = self.ce(state_action_A)  # choose the ce from Q table
            solB = self.ce(state_action_B)  # choose the ce from Q table

            prob_A = np.abs(np.array(solA['x']).reshape((25, -1))) / sum(np.abs(solA['x']))
            prob_B = np.abs(np.array(solB['x']).reshape((25, -1))) / sum(np.abs(solB['x']))
            # print("ce_Q line 136, prob_A", prob_A)
            # print("ce_Q line 137, prob_A.shape", prob_A.flatten().shape, self.q_table_A.loc[s, :].shape)

            # print("ce_Q line 140, self.vi_A.loc[s, :]", self.vi_A.loc[s, :])

            ind_A = int(np.random.choice(np.arange(25), 1, p=prob_A.flatten()))
            ind_B = int(np.random.choice(np.arange(25), 1, p=prob_B.flatten()))
            # print("ce_Q line 144, ind_A", ind_A)
            action_A = self.A[ind_A][0]
            action_B = self.A[ind_B][1]

        else:
            # choose random action
            action_A = np.random.choice(self.actions)
            action_B = np.random.choice(self.actions)
            # print("friend_Q.py line 96, random choose, action A, action B", action_A, action_B)
        return tuple([action_A, action_B])

    def learn(self, s, a, r, s_, a_, alpha, gamma):
        # a[0], a[1] are actions for players A and B
        # s[1], s[2] are states for players A and B. s[0] indicates the ball holding status.
        # print("ce_Q.py, line 158, s, a, r, s_, a_", s, a, r, s_, a_)

        q_predict_A = self.q_table_A.loc[s, a]
        q_predict_B = self.q_table_B.loc[s, a]

        if 0 in r:
            # print("ce_Q.py line 185, self.vi_A.loc[s_, :] ", self.vi_A.loc[s_, :].value)
            q_target_A = r[0] + gamma * self.vi_A.loc[s_, :].value  # next state is not terminal
            q_target_B = r[1] + gamma * self.vi_A.loc[s_, :].value  # next state is not terminal

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

        solA = self.ce(state_action_A)  # choose the ce from Q table
        solB = self.ce(state_action_B)  # choose the ce from Q table

        prob_A = np.abs(np.array(solA['x']).reshape((25, -1))) / sum(np.abs(solA['x']))
        prob_B = np.abs(np.array(solB['x']).reshape((25, -1))) / sum(np.abs(solB['x']))

        self.vi_A.loc[s, :] = np.sum(prob_A.flatten() * self.q_table_A.loc[s, :])
        self.vi_B.loc[s, :] = np.sum(prob_B.flatten() * self.q_table_B.loc[s, :])

        if self.verbose >= 2:
            # print("s,a,r,s_,a_:", s,a,r,s_,a_)
            print('\n Q table A is:\n', self.q_table_A)
            print('\n Q table B is:\n', self.q_table_A)
            pass

    def return_Q_table(self):
        return self.q_table_A, self.q_table_B


