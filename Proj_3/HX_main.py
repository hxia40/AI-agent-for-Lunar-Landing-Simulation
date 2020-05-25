"""
Reinforcement learning

"""
from HX_sarsa_Q_table import Two_player_SARSA_table
from HX_friend_Q_table import Friend_Q_table
from HX_foe_Q_table import Foe_Q_table
from HX_ce_Q_table import CE_Q_table
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
pd.set_option('display.max_columns', None)

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
    start_alpha = alpha
    print("alpha_decay_rate", alpha_decay_rate)
    print("epsilon_decay_rate", epsilon_decay_rate)
    error_list = []
    delta_list = []
    time_list = []
    time_list_sub = []
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
            print("episode = ", episode + 1, "alpha = ", alpha, "epsilon =", epsilon)
            reward_list_A.append(np.mean(reward_list_A_sub))
            reward_list_A_sub = []
            time_list.append(np.mean(time_list_sub))
            time_list_sub = []

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
                time_list_sub.append(time.time() - start_time)
                old_q_sa = new_q_sa
                old_q = new_q
                break
        alpha = max(alpha * alpha_decay_rate, alpha_min)
        epsilon = max(epsilon * epsilon_decay_rate, epsilon_min)

    # after all episode done, let us check what the actions will be on the initial state, with epsilon = 0:
    # action_A_list = []
    # action_B_list = []
    # for i in range(100):
    #     action_A_list.append(table_SARSA.choose_action((1, 2, 1), 0)[0])
    #     action_B_list.append(table_SARSA.choose_action((1, 2, 1), 0)[1])
    # with open('sarsa_Q/actions_a_%0.5f_adecay_%0.5f_timeout%f.txt' % (start_alpha, alpha_decay_rate, timeout),
    #           'a') as f:
    #     f.write("actions_A\n")
    #     for itemA in action_A_list:
    #         f.write("%s\n" % itemA)
    #     f.write("actions_B\n")
    #     for itemB in action_B_list:
    #         f.write("%s\n" % itemB)
    # f.close()

    # plotting everything
    # plt.plot(range(num_episode), error_list,
    #          label="Q")
    # # plt.ylim([0, .5])
    # # plt.show()
    # plt.savefig(
    #     "sarsa_Q/errlarge_friend_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()
    #
    # plt.plot(range(num_episode), delta_list,
    #          label="Q")
    # # plt.ylim([0, .5])
    # # plt.show()
    # plt.savefig(
    #     "sarsa_Q/deltalarge_friend_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()

    plt.plot(range(num_episode), error_list,
             label="Q")
    plt.ylim([0, .5])
    # plt.show()
    plt.savefig(
        "sarsa.png")
    plt.clf()

    # plt.plot(range(num_episode), delta_list,
    #          label="Q")
    # plt.ylim([0, .5])
    # # plt.show()
    # plt.savefig(
    #     "sarsa_Q/deltasmall_friend_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()
    #
    # plt.plot(range(num_episode / 100), reward_list_A,
    #          label="reward for player A")
    # plt.savefig(
    #     "sarsa_Q/rewards_friend_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()
    #
    # plt.plot(range(num_episode / 100), time_list,
    #          label="Q")
    # plt.savefig("sarsa_Q/time_ce_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()
    #
    # q_output_A.to_csv(
    #     'sarsa_Q/friend_a_%0.5f_adecay_%0.5f_timeout%f.csv' % (start_alpha, alpha_decay_rate, timeout))
    # q_output_B.to_csv(
    #     'sarsa_Q/friend_a_%0.5f_adecay_%0.5f_timeout%f.csv' % (start_alpha, alpha_decay_rate, timeout))

    # end of game
    print('game over')
    # env.destroy()


def friend_Q(
          num_episode = 10000,
          alpha = 0.5,
          alpha_decay_rate = 0.998,
          alpha_min = 0.001,
          epsilon = 0.5,
          epsilon_decay_rate = 0.998,
          epsilon_min = 0.001,
          gamma = 0.99,
          timeout = 10000
              ):
    start_alpha = alpha
    print("alpha_decay_rate", alpha_decay_rate)
    print("epsilon_decay_rate", epsilon_decay_rate)
    error_list = []
    delta_list = []
    reward_list_A = []
    reward_list_A_sub = []
    time_list = []
    time_list_sub = []

    range_end = num_episode
    q_output_A, q_output_B = table_friend_Q.return_Q_table()

    # find player A's Q-values, corresponding to state s (A stands on 2, B stands on 1, B has the ball, i.e. S[(1,2,1)])
    # and A takes action S, B stick, which is  (+4, 0).

    old_q_sa = q_output_A.loc[(1,2,1), (4,0)]
    old_q = np.array(q_output_A).flatten()
    # print("main.py line 171", old_q_sa)

    for episode in range(range_end):
        if episode % 100 == 0:
            print("episode = ", episode + 1, "alpha = ", alpha, "epsilon =", epsilon)
            reward_list_A.append(np.mean(reward_list_A_sub))
            reward_list_A_sub = []
            time_list.append(np.mean(time_list_sub))
            time_list_sub = []

        ''' for each episode, the algorithm on Figure 6.5 of Sutton book is executed:
        Initialize S
        Choose A from S using policy derived Q (e.g. sigma-greedy)
        Repeat (for each step of this episode)
            Take action A, observe R, S'
            choose A' from S' using policy derived Q (e.g. sigma-greedy)
            Q(S, A) <- Q(S,A) + alpha[R + gamma * Q(S',A') - Q(S,A)
            S <- S'; A <- A'
        until S is terminal 
        '''

        # initial observation
        observation = env.reset()   # (1) Initialize S
        # print("main.py line 192, observation\n", observation)
        start_time = time.time()
        # QL choose action based on observation # (2) choose A from S using policy derived Q (e.g. sigma-greedy)
        action = table_friend_Q.choose_action(observation, epsilon)
        # print("main.py, line 198, action", action)
        # print("new episode!, observation is:", observation)
        for t in range(timeout):                     # (3) Repeat (for each step of this episode)
            # # fresh env

            # QL take action and get next observation and reward  # R(1) Take action A, observe R, S'
            observation_, reward, done, info = env.step(action)
            # print("main.py line 205, a, obs_, r, done, info:", action, observation_, reward, done, info)

            # QL choose action based on observation  # R(2) choose A' from S' using policy derived Q (e.g. sigma-greedy)
            action_ = table_friend_Q.choose_action(observation_, epsilon)
            # print("main.py line 209, action_:", action_)

            # QL learn from this transition  # R(3) Q(S, A) <- Q(S,A) + alpha[R + gamma * Q(S',A') - Q(S,A)
            table_friend_Q.learn(observation, action, reward, observation_, action_, alpha, gamma)

            # swap observation      #  R(4) S <- S'; A <- A'
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done or t == timeout - 1:
                q_output_A, q_output_B = table_friend_Q.return_Q_table()
                # after one episode, find player A's Q-values, corresponding to state s (as point 2) and action S (+4).
                new_q_sa = q_output_A.loc[(1,2,1), (4,0)]
                new_q = np.array(q_output_A).flatten()
                # print("main.py line 45", new_q_sa)
                error_list.append(np.abs(new_q_sa - old_q_sa))
                delta_list.append(rmse(old_q, new_q))
                reward_list_A_sub.append(reward[0])
                time_list_sub.append(time.time() - start_time)
                old_q_sa = new_q_sa
                old_q = new_q
                break
        alpha = max(alpha * alpha_decay_rate, alpha_min)
        epsilon = max(epsilon * epsilon_decay_rate, epsilon_min)

    # after all episode done, let us check what the actions will be on the initial state, with epsilon = 0:
    # action_A_list = []
    # action_B_list = []
    # for i in range(100):
    #     action_A_list.append(table_friend_Q.choose_action((1, 2, 1), 0)[0])
    #     action_B_list.append(table_friend_Q.choose_action((1, 2, 1), 0)[1])
    # with open('foe_Q/actions_a_%0.5f_adecay_%0.5f_timeout%f.txt' % (start_alpha, alpha_decay_rate, timeout), 'a') as f:
    #     f.write("actions_A\n")
    #     for itemA in action_A_list:
    #         f.write("%s\n" % itemA)
    #     f.write("actions_B\n")
    #     for itemB in action_B_list:
    #         f.write("%s\n" % itemB)
    # f.close()
    #
    # # plotting everything
    # plt.plot(range(num_episode), error_list,
    #          label="Q")
    # # plt.ylim([0, .5])
    # # plt.show()
    # plt.savefig("friend_Q/errlarge_friend_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()
    #
    # plt.plot(range(num_episode), delta_list,
    #          label="Q")
    # # plt.ylim([0, .5])
    # # plt.show()
    # plt.savefig("friend_Q/deltalarge_friend_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()

    plt.plot(range(num_episode), error_list,
             label="Q")
    plt.ylim([0, .5])
    # plt.show()
    plt.savefig("friend_Q.png")
    plt.clf()

    # plt.plot(range(num_episode), delta_list,
    #          label="Q")
    # plt.ylim([0, .5])
    # # plt.show()
    # plt.savefig("friend_Q/deltasmall_friend_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()
    #
    # plt.plot(range(num_episode / 100), reward_list_A,
    #          label="reward for player A")
    # plt.savefig("friend_Q/rewards_friend_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()
    #
    # plt.plot(range(num_episode / 100), time_list,
    #          label="Q")
    # plt.savefig("friend_Q/time_ce_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()
    #
    # q_output_A.to_csv('friend_Q/friend_a_%0.5f_adecay_%0.5f_timeout%f.csv' % (start_alpha, alpha_decay_rate, timeout))
    # q_output_B.to_csv('friend_Q/friend_a_%0.5f_adecay_%0.5f_timeout%f.csv' % (start_alpha, alpha_decay_rate, timeout))
    # # end of game
    # print("reward A mean:", np.mean(reward_list_A))
    # # print("reward B mean:", np.mean(reward_list_B))
    # print("reward A std:", np.std(reward_list_A))
    # # print("reward B std:", np.std(reward_list_B))
    print('game over')
    # env.destroy()

def foe_Q(
          num_episode = 10000,
          alpha = 0.5,
          alpha_decay_rate = 0.998,
          alpha_min = 0.001,
          epsilon = 0.5,
          epsilon_decay_rate = 0.998,
          epsilon_min = 0.001,
          gamma = 0.99,
          timeout = 10000
              ):
    start_alpha = alpha
    print("alpha_decay_rate", alpha_decay_rate)
    print("epsilon_decay_rate", epsilon_decay_rate)
    error_list = []
    delta_list = []
    time_list = []
    time_list_sub = []
    reward_list_A = []
    reward_list_A_sub = []

    range_end = num_episode
    q_output_A, q_output_B = table_foe_Q.return_Q_table()

    # find player A's Q-values, corresponding to state s (A stands on 2, B stands on 1, B has the ball, i.e. S[(1,2,1)])
    # and A takes action S, B stick, which is  (+4, 0).
    # print("=========main.py line 277, q_output_A===========")
    # print(q_output_A)
    old_q_sa = q_output_A.loc[(1,2,1), (4,0)]
    old_q = np.array(q_output_A).flatten()

    for episode in range(range_end):
        if episode % 100 == 0:
            reward_list_A.append(np.mean(reward_list_A_sub))
            reward_list_A_sub = []
            time_list.append(np.mean(time_list_sub))
            time_list_sub = []
            print("episode = ", episode + 1, "alpha = ", alpha, "epsilon =", epsilon)

        ''' for each episode, the algorithm on Figure 6.5 of Sutton book is executed:
        Initialize S
        Choose A from S using policy derived Q (e.g. sigma-greedy)
        Repeat (for each step of this episode)
            Take action A, observe R, S'
            choose A' from S' using policy derived Q (e.g. sigma-greedy)
            Q(S, A) <- Q(S,A) + alpha[R + gamma * Q(S',A') - Q(S,A)
            S <- S'; A <- A'
        until S is terminal 
        '''

        # initial observation
        observation = env.reset()   # (1) Initialize S
        # print("main.py line 64, observation\n", observation)
        start_time = time.time()
        # QL choose action based on observation # (2) choose A from S using policy derived Q (e.g. sigma-greedy)
        action = table_foe_Q.choose_action(observation, epsilon)
        # print("main.py, line 68, action", action)
        # print("new episode!, observation is:", observation)
        for t in range(timeout):                     # (3) Repeat (for each step of this episode)
            # # fresh env

            # QL take action and get next observation and reward  # R(1) Take action A, observe R, S'
            observation_, reward, done, info = env.step(action)
            # print("main.py line 182, a, obs_, r, done, info:", action, observation_, reward, done, info)

            # QL choose action based on observation  # R(2) choose A' from S' using policy derived Q (e.g. sigma-greedy)
            action_ = table_foe_Q.choose_action(observation_, epsilon)
            # print("main.py line 186, action_:", action_)

            # QL learn from this transition  # R(3) Q(S, A) <- Q(S,A) + alpha[R + gamma * Q(S',A') - Q(S,A)
            table_foe_Q.learn(observation, action, reward, observation_, action_, alpha, gamma)

            # swap observation      #  R(4) S <- S'; A <- A'
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done or t == timeout - 1:
                q_output_A, q_output_B = table_foe_Q.return_Q_table()
                # after one episode, find player A's Q-values, corresponding to state s (as point 2) and action S (+4).
                new_q_sa = q_output_A.loc[(1,2,1), (4,0)]
                new_q = np.array(q_output_A).flatten()
                # print("main.py line 45", new_q_sa)
                error_list.append(np.abs(new_q_sa - old_q_sa))
                delta_list.append(rmse(old_q, new_q))
                reward_list_A_sub.append(reward[0])
                time_list_sub.append(time.time() - start_time)
                old_q_sa = new_q_sa
                old_q = new_q
                break
        alpha = max(alpha * alpha_decay_rate, alpha_min)
        epsilon = max(epsilon * epsilon_decay_rate, epsilon_min)

    # after all episode done, let us check what the actions will be on the initial state, with epsilon = 0:
    # action_A_list = []
    # action_B_list = []
    # for i in range(100):
    #     action_A_list.append(table_foe_Q.choose_action((1, 2, 1), 0)[0])
    #     action_B_list.append(table_foe_Q.choose_action((1, 2, 1), 0)[1])
    # with open('friend_Q/actions_a_%0.5f_adecay_%0.5f_timeout%f.txt' % (start_alpha, alpha_decay_rate, timeout), 'a') as f:
    #     f.write("actions_A\n")
    #     for itemA in action_A_list:
    #         f.write("%s\n" % itemA)
    #     f.write("actions_B\n")
    #     for itemB in action_B_list:
    #         f.write("%s\n" % itemB)
    # f.close()

    # plotting everything
    # plt.plot(range(num_episode), error_list,
    #          label="Q")
    # # plt.ylim([0, .5])
    # # plt.show()
    # plt.savefig(
    #     "foe_Q/errlarge_foe_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()
    #
    # plt.plot(range(num_episode), delta_list,
    #          label="Q")
    # # plt.ylim([0, .5])
    # # plt.show()
    # plt.savefig(
    #     "foe_Q/deltalarge_foe_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()

    plt.plot(range(num_episode), error_list,
             label="Q")
    plt.ylim([0, .5])
    # plt.show()
    plt.savefig(
        "foe_Q.png")
    plt.clf()

    # plt.plot(range(num_episode), delta_list,
    #          label="Q")
    # plt.ylim([0, .5])
    # # plt.show()
    # plt.savefig(
    #     "foe_Q/deltasmall_foe_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()
    #
    # plt.plot(range(num_episode / 100), reward_list_A,
    #          label="reward for player A")
    # plt.savefig("foe_Q/rewards_foe_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()
    #
    # plt.plot(range(num_episode / 100), time_list,
    #          label="Q")
    # plt.savefig(
    #     "foe_Q/time_ce_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()
    #
    # q_output_A.to_csv('foe_Q/foe_a_%0.5f_adecay_%0.5f_timeout%f.csv' % (start_alpha, alpha_decay_rate, timeout))
    # q_output_B.to_csv('foe_Q/foe_a_%0.5f_adecay_%0.5f_timeout%f.csv' % (start_alpha, alpha_decay_rate, timeout))
    # # end of game
    # print("reward A mean:", np.mean(reward_list_A))
    # # print("reward B mean:", np.mean(reward_list_B))
    # print("reward A std:", np.std(reward_list_A))
    # # print("reward B std:", np.std(reward_list_B))
    print('game over')
    # env.destroy()


def ce_Q(
          num_episode = 10000,
          alpha = 0.5,
          alpha_decay_rate = 0.998,
          alpha_min = 0.001,
          epsilon = 0.5,
          epsilon_decay_rate = 0.998,
          epsilon_min = 0.001,
          gamma = 0.99,
          timeout = 10000
              ):
    start_alpha = alpha
    print("alpha_decay_rate", alpha_decay_rate)
    print("epsilon_decay_rate", epsilon_decay_rate)
    error_list = []
    delta_list = []
    time_list = []
    time_list_sub = []
    reward_list_A = []
    reward_list_A_sub = []

    range_end = num_episode
    q_output_A, q_output_B = table_ce_Q.return_Q_table()

    old_q_sa = q_output_A.loc[(1,2,1), (4,0)]
    old_q = np.array(q_output_A).flatten()

    for episode in range(range_end):
        if episode % 100 == 0:
            reward_list_A.append(np.mean(reward_list_A_sub))
            reward_list_A_sub = []
            time_list.append(np.mean(time_list_sub))
            time_list_sub = []
            print("episode = ", episode + 1, "alpha = ", alpha, "epsilon =", epsilon)

        # initial observation
        observation = env.reset()   # (1) Initialize S
        # print("main.py line 64, observation\n", observation)
        start_time = time.time()
        # QL choose action based on observation # (2) choose A from S using policy derived Q (e.g. sigma-greedy)
        action = table_ce_Q.choose_action(observation, epsilon)
        # print("main.py, line 68, action", action)
        # print("new episode!, observation is:", observation)
        for t in range(timeout):                     # (3) Repeat (for each step of this episode)
            # # fresh env

            # QL take action and get next observation and reward  # R(1) Take action A, observe R, S'
            observation_, reward, done, info = env.step(action)
            # print("main.py line 182, a, obs_, r, done, info:", action, observation_, reward, done, info)

            # QL choose action based on observation  # R(2) choose A' from S' using policy derived Q (e.g. sigma-greedy)
            action_ = table_ce_Q.choose_action(observation_, epsilon)
            # print("main.py line 186, action_:", action_)

            # QL learn from this transition  # R(3) Q(S, A) <- Q(S,A) + alpha[R + gamma * Q(S',A') - Q(S,A)
            table_ce_Q.learn(observation, action, reward, observation_, action_, alpha, gamma)

            # swap observation      #  R(4) S <- S'; A <- A'
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done or t == timeout - 1:
                q_output_A, q_output_B = table_ce_Q.return_Q_table()
                # after one episode, find player A's Q-values, corresponding to state s (as point 2) and action S (+4).
                new_q_sa = q_output_A.loc[(1,2,1), (4,0)]
                new_q = np.array(q_output_A).flatten()
                # print("main.py line 45", new_q_sa)
                error_list.append(np.abs(new_q_sa - old_q_sa))
                delta_list.append(rmse(old_q, new_q))
                time_list_sub.append(time.time() - start_time)
                reward_list_A_sub.append(reward[0])
                old_q_sa = new_q_sa
                old_q = new_q

                break
        alpha = max(alpha * alpha_decay_rate, alpha_min)
        epsilon = max(epsilon * epsilon_decay_rate, epsilon_min)

    # after all episode done, let us check what the actions will be on the initial state, with epsilon = 0:
    # action_A_list = []
    # action_B_list = []
    # for i in range(100):
    #     action_A_list.append(table_ce_Q.choose_action((1, 2, 1), 0)[0])
    #     action_B_list.append(table_ce_Q.choose_action((1, 2, 1), 0)[1])
    # with open('ce_Q/actions_a_%0.5f_adecay_%0.5f_timeout%f.txt' % (start_alpha, alpha_decay_rate, timeout), 'a') as f:
    #     f.write("actions_A\n")
    #     for itemA in action_A_list:
    #         f.write("%s\n" % itemA)
    #     f.write("actions_B\n")
    #     for itemB in action_B_list:
    #         f.write("%s\n" % itemB)
    # f.close()

    # plotting everything

    # plt.plot(range(num_episode), error_list,
    #          label="Q")
    # # plt.ylim([0, .5])
    # # plt.show()
    # plt.savefig(
    #     "ce_Q/errlarge_ce_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()
    #
    # plt.plot(range(num_episode), delta_list,
    #          label="Q")
    # # plt.ylim([0, .5])
    # # plt.show()
    # plt.savefig(
    #     "ce_Q/deltalarge_ce_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()

    plt.plot(range(num_episode), error_list,
             label="Q")
    plt.ylim([0, .5])
    # plt.show()
    plt.savefig(
        "ce_Q.png")
    plt.clf()

    # plt.plot(range(num_episode), delta_list,
    #          label="Q")
    # plt.ylim([0, .5])
    # # plt.show()
    # plt.savefig(
    #     "ce_Q/deltasmall_ce_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()
    #
    # plt.plot(range(num_episode / 100), reward_list_A,
    #          label="reward for player A")
    # plt.savefig("ce_Q/rewards_ce_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()
    #
    # plt.plot(range(num_episode / 100), time_list,
    #          label="Q")
    # plt.savefig(
    #     "ce_Q/time_ce_a_%0.5f_adecay_%0.5f_timeout%f.png" % (start_alpha, alpha_decay_rate, timeout))
    # plt.clf()
    #
    # q_output_A.to_csv('ce_Q/ce_A_a_%0.5f_adecay_%0.5f_timeout%f.csv' % (start_alpha, alpha_decay_rate, timeout))
    # q_output_B.to_csv('ce_Q/ce_B_a_%0.5f_adecay_%0.5f_timeout%f.csv' % (start_alpha, alpha_decay_rate, timeout))

    # end of game
    print('game over')
    # env.destroy()

if __name__ == "__main__":
    seed = 1  # seed
    np.random.seed(seed)

    '''soccer env'''
    env = SoccerEnv()
    env.seed(seed)

    for alpha in [
                  # 1,
                  # 0.75,
                  0.5,
                  # 0.25
                  ]:
        for alpha_decay_rate in [
                                # 1,
                                # 0.99999,
                                # # 0.9999,
                                   0.9995,
                                #    0.999,
                                 ]:
            table_SARSA = Two_player_SARSA_table(verbose=False)
            SARSA(num_episode=10000,
                  alpha=alpha,
                  alpha_decay_rate = alpha_decay_rate,
                  alpha_min=0.001,
                  epsilon=0.5,
                  # epsilon_decay_rate=0.999982730761,
                  epsilon_decay_rate=0.9995,
                  epsilon_min=0.00,
                  gamma=0.9,
                  timeout=1000
                  )

            table_friend_Q = Friend_Q_table(verbose=False)
            friend_Q(num_episode=10000,
                  alpha=alpha,
                  alpha_decay_rate = alpha_decay_rate,
                  alpha_min=0.001,
                  epsilon=0.5,
                  # epsilon_decay_rate=0.999982730761,
                  epsilon_decay_rate=0.9995,
                  epsilon_min=0.00,
                  gamma=0.9,
                  timeout=1000
                  )

            table_foe_Q = Foe_Q_table(verbose=False)
            foe_Q(num_episode=10000,
                  alpha=alpha,
                  alpha_decay_rate = alpha_decay_rate,
                  alpha_min=0.001,
                  epsilon=0.5,
                  # epsilon_decay_rate=0.999982730761,
                  epsilon_decay_rate=0.9995,
                  epsilon_min=0.00,
                  gamma=0.9,
                  timeout=1000
                  )

            table_ce_Q = CE_Q_table(verbose=False)
            ce_Q(num_episode=10000,
                  alpha=alpha,
                  alpha_decay_rate = alpha_decay_rate,
                  alpha_min=0.001,
                  epsilon=0.5,
                  # epsilon_decay_rate=0.999982730761,
                  epsilon_decay_rate=0.9995,
                  epsilon_min=0.00,
                  gamma=0.9,
                  timeout=1000
                  )



