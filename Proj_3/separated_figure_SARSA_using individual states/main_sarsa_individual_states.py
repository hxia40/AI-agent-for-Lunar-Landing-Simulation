"""
Reinforcement learning


Reference:
Morzan,
wfeng,
Moustafa Alzantot (malzantot@ucla.edu)

"""
from HX_sarsa_individual_state_Q_table import Two_player_SARSA_table
from env_soccer import SoccerEnv

import gym
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def SARSA(
          num_episode = 10000,
          alpha = 0.5,
          alpha_decay_rate = 0.998,
          alpha_min = 0.001,
          epsilon = 0.5,
          epsilon_decay_rate = 0.998,
          epsilon_min = 0.001,
          gamma = 0.99,
          timeout = 25
              ):

    error_list = []
    delta_list = []
    reward_list_A = []
    reward_list_A_sub = []

    range_end = num_episode
    q_output_A, q_output_B = table_SARSA.return_Q_table()

    # find player A's Q-values, corresponding to state s (A stands on 2, B stands on 1, B has the ball, as S[71])
    # and action S (+4).
    # print("main.py line 47", q_output_A)
    old_q_sa = q_output_A.loc[2, 4]
    old_q = np.array(q_output_A).flatten()
    # print("main.py line 45", old_q_sa)

    for episode in range(range_end):
        if episode % 100 == 0:
            reward_list_A.append(np.mean(reward_list_A_sub))
            reward_list_A_sub = []

        print("episode = ", episode + 1, "alpha = ", alpha, "epsilon =", epsilon)



        # initial observation
        observation = env.reset()   # (1) Initialize S
        # print("main.py line 64, observation\n", observation)
        start_time = time.time()
        # QL choose action based on observation # (2) choose A from S using policy derived Q (e.g. sigma-greedy)
        action = table_SARSA.choose_action(observation, epsilon)
        # print("main.py, line 68, action", action)
        # print("new episode!, observation is:", observation)
        for t in range(timeout):                     # (3) Repeat (for each step of this episode)
            # # fresh env

            # QL take action and get next observation and reward  # R(1) Take action A, observe R, S'
            observation_, reward, done, info = env.step(action)
            # print("main.py line 81, a, obs_, r, done, info:", action, observation_, reward, done, info)

            # QL choose action based on observation  # R(2) choose A' from S' using policy derived Q (e.g. sigma-greedy)
            action_ = table_SARSA.choose_action(observation_, epsilon)
            # print env_HW3.step(action_)

            # QL learn from this transition  # R(3) Q(S, A) <- Q(S,A) + alpha[R + gamma * Q(S',A') - Q(S,A)
            table_SARSA.learn(observation, action, reward, observation_, action_, alpha, gamma)

            # swap observation      #  R(4) S <- S'; A <- A'
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done or t == timeout - 1:
                q_output_A, q_output_B = table_SARSA.return_Q_table()
                # after one episode, find player A's Q-values, corresponding to state s (as point 2) and action S (+4).
                new_q_sa = q_output_A.loc[2, 4]
                new_q = np.array(q_output_A).flatten()
                # print("main.py line 99", new_q_sa, old_q_sa)
                error_list.append(np.abs(new_q_sa - old_q_sa))
                delta_list.append(rmse(old_q, new_q))
                reward_list_A_sub.append(reward[0])
                old_q_sa = new_q_sa
                old_q = new_q
                break
        alpha = max(alpha * alpha_decay_rate, alpha_min)
        epsilon = max(epsilon * epsilon_decay_rate, epsilon_min)
    print("main.py line 105, q_output_A after all iterations", q_output_A)
    print("main.py line 106, q_output_B after all iterations", q_output_B)
    # print("main.py line 106, error_list after all iterations", error_list)
    plt.plot(range(num_episode), error_list,
             label="Q")
    # plt.ylim([0, .5])
    # plt.show()
    plt.savefig("errlarge_SARSA_a_%d_adecay_%d_timeout25.png" % (alpha, alpha_decay_rate))
    plt.clf()

    plt.plot(range(num_episode), delta_list,
             label="Q")
    # plt.ylim([0, .5])
    # plt.show()
    plt.savefig("deltalarge_SARSA_a_%d_adecay_%d_timeout25.png" % (alpha, alpha_decay_rate))
    plt.clf()

    plt.plot(range(num_episode), error_list,
             label="Q")
    plt.ylim([0, .5])
    # plt.show()
    plt.savefig("errsmall_SARSA_a_%d_adecay_%d_timeout25.png" % (alpha, alpha_decay_rate))
    plt.clf()

    plt.plot(range(num_episode), delta_list,
             label="Q")
    plt.ylim([0, .5])
    # plt.show()
    plt.savefig("deltasmall_SARSA_a_%d_adecay_%d_timeout25.png" % (alpha, alpha_decay_rate))
    plt.clf()

    plt.plot(range(num_episode / 100), reward_list_A,
             label="reward for player A")
    plt.savefig("rewards_SARSA_a_%d_adecay_%d_timeout25.png" % (alpha, alpha_decay_rate))
    plt.clf()

    # end of game
    print('game over')
    # env.destroy()



if __name__ == "__main__":

    seed = 0  # seed
    np.random.seed(seed)

    '''soccer env'''
    env = SoccerEnv()
    env.seed(seed)

    for alpha in [0.9,
                  0.7,
                  0.5,
                  0.3]:
        for alpha_decay_rate  in [0.9999,
                                  0.9995,
                                  0.999,
                                  ]:

            table_SARSA = Two_player_SARSA_table(verbose=False)
            SARSA(num_episode = 10000,
                  alpha = alpha,
                  alpha_decay_rate =  alpha_decay_rate,
                  epsilon = 0.5,
                  epsilon_decay_rate =  0.9995,
                  gamma = 0.9,
                  timeout = 25
          )




