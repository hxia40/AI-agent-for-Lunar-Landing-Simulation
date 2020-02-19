"""
modified from morvan zhou's Q learner table
View more on morvan tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time


class SARSA_TABLE:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, verbose=False):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.new_state_counter = 0
        self.verbose = verbose

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.randint(len(self.actions))  # using np.random.randint instead of np.random.choice for RL, HW 3
            # action = np.random.choice(self.actions)

        return action

    # def learn(self, s, a, r, s_, a_, alpha):
    #     self.check_state_exist(s_)
    #     q_predict = self.q_table.loc[s, a]
    #     if s_ != 'terminal':
    #         q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
    #     else:
    #         q_target = r  # next state is terminal
    #     # self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update , Morvan's original
    #     self.q_table.loc[s, a] += alpha * (q_target - q_predict)  # update , HX self defined
    #     # print("updating self.q_table.loc[s, a], updated s, a, Q(s,a) is:", s, a, self.q_table.loc[s, a])
    #     if self.verbose >= 2:
    #         print('\n Q table is:\n', self.q_table)

    def learn(self, s, a, r, s_, a_, alpha):  # R(3) Q(S, A) <- Q(S,A) + alpha[R + gamma * Q(S',A') - Q(S,A)
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        # self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update , Morvan's original
        self.q_table.loc[s, a] += alpha * (q_target - q_predict)  # update , HX self defined
        if self.verbose >= 2:
            print('\n Q table is:\n', self.q_table)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.new_state_counter += 1
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
            if self.verbose >= 1:
                print('========adding', self.new_state_counter,'th new state====== : ', state)
            if self.verbose >= 2:
                print('\n Q table added new state:\n', self.q_table)


