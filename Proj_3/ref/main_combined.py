"""
Reinforcement learning


Reference:
Morzan,
wfeng,
Moustafa Alzantot (malzantot@ucla.edu)

"""
from HX_two_player_SARSA_table import Two_player_SARSA_table
# from HX_friend_Q_table import Friend_Q_table
# from HX_foe_Q_table import Foe_Q_table
# from HX_ce_Q_table import CE_Q_table
from HX_friend_Q_table_combinedQ import Friend_Q_table
from HX_foe_Q_table_combinedQ import Foe_Q_table
from HX_ce_Q_table_combinedQ import CE_Q_table
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
    S_file = open('SARSALearner.txt', 'a')
    print("alpha_decay_rate", alpha_decay_rate)
    print("epsilon_decay_rate", epsilon_decay_rate)
    error_list = []
    delta_list = []
    reward_list_A = []
    reward_list_A_sub = []

    range_end = num_episode
    q_output_A, q_output_B = table_SARSA.return_Q_table()

    # find player A's Q-values, corresponding to state s (A stands on 2, B stands on 1, B has the ball, as S[71])
    # and action S (+4).
    print("main.py line 61", q_output_A)
    old_q_sa = q_output_A.loc[(1,2,1), 4]
    old_q = np.array(q_output_A).flatten()

    for episode in range(range_end):

        print("episode = ", episode + 1, "alpha = ", alpha, "epsilon =", epsilon)
        if episode % 100 == 0:
            reward_list_A.append(np.mean(reward_list_A_sub))
            reward_list_A_sub = []
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
                new_q_sa = q_output_A.loc[(1,2,1), 4]
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
    # print("main.py line 105, q_output_A after all iterations", q_output_A)
    # print("main.py line 106, q_output_B after all iterations", q_output_B)
    print("main.py line 106, error_list after all iterations", error_list)

    plt.plot(range(num_episode), error_list,
             label="Q")
    # plt.ylim([0, .5])
    # plt.show()
    plt.savefig("sarsaQ_all_0.9995_0.9995_timeout25.png")
    plt.clf()

    plt.plot(range(num_episode), delta_list,
             label="Q")
    # plt.ylim([0, .5])
    # plt.show()
    plt.savefig("delta_sarsaQ_all_0.9995_0.9995_timeout25.png")
    plt.clf()

    plt.plot(range(num_episode), error_list,
             label="Q")
    plt.ylim([0, .5])
    # plt.show()
    plt.savefig("sarsaQ_small_0.9995_0.9995_timeout25.png")
    plt.clf()

    plt.plot(range(num_episode), delta_list,
             label="Q")
    plt.ylim([0, .5])
    # plt.show()
    plt.savefig("delta_sarsaQ_small_0.9995_0.9995_timeout25.png")
    plt.clf()

    plt.plot(range(num_episode / 100), reward_list_A,
             label="reward for player A")
    plt.savefig("rewards_sarsaQ_small_0.9995_0.9995_timeout25.png")
    plt.clf()

    # end of game
    print("reward A mean:", np.mean(reward_list_A))
    # print("reward B mean:", np.mean(reward_list_B))
    print("reward A std:", np.std(reward_list_A))
    # print("reward B std:", np.std(reward_list_B))
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
    S_file = open('Friend_Q_Learner.txt', 'a')
    print("alpha_decay_rate", alpha_decay_rate)
    print("epsilon_decay_rate", epsilon_decay_rate)
    error_list = []
    delta_list = []
    reward_list_A = []
    reward_list_A_sub = []

    range_end = num_episode
    q_output = table_friend_Q.return_Q_table()
    print("main.py line 170, q_output ", type(q_output))

    # find player A's Q-values, corresponding to state s (A stands on 2, B stands on 1, B has the ball, i.e. S[(1,2,1)])
    # and A takes action S, B stick, which is  (+4, 0).
    # print("=========main.py line 164, q_output_A===========")
    # print(q_output_A)
    old_q_sa = q_output.loc[(1,2,1), (4,0)]
    old_q = np.array(q_output).flatten()
    print("main.py line 152, old_q", old_q)

    for episode in range(range_end):
        if episode % 100 == 0:
            reward_list_A.append(np.mean(reward_list_A_sub))
            reward_list_A_sub = []

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
        action = table_friend_Q.choose_action(observation, epsilon)
        # print("main.py, line 68, action", action)
        # print("new episode!, observation is:", observation)
        for t in range(timeout):                     # (3) Repeat (for each step of this episode)
            # # fresh env

            # QL take action and get next observation and reward  # R(1) Take action A, observe R, S'
            observation_, reward, done, info = env.step(action)
            # print("main.py line 182, a, obs_, r, done, info:", action, observation_, reward, done, info)

            # QL choose action based on observation  # R(2) choose A' from S' using policy derived Q (e.g. sigma-greedy)
            action_ = table_friend_Q.choose_action(observation_, epsilon)
            # print("main.py line 186, action_:", action_)

            # QL learn from this transition  # R(3) Q(S, A) <- Q(S,A) + alpha[R + gamma * Q(S',A') - Q(S,A)
            table_friend_Q.learn(observation, action, reward, observation_, action_, alpha, gamma)

            # swap observation      #  R(4) S <- S'; A <- A'
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done or t == timeout - 1:
                q_output = table_friend_Q.return_Q_table()

                # after one episode, find player A's Q-values, corresponding to state s (as point 2) and action S (+4).
                new_q_sa = q_output.loc[(1,2,1), (4,0)]
                new_q = np.array(q_output).flatten()
                # print("main.py line 45", new_q_sa)
                error_list.append(np.abs(new_q_sa - old_q_sa))
                delta_list.append(rmse(old_q, new_q))
                # print("main.py line 233, rmse(old_q, new_q)", rmse(old_q, new_q))
                reward_list_A_sub.append(reward[0])
                old_q_sa = new_q_sa
                old_q = new_q
                break
        alpha = max(alpha * alpha_decay_rate, alpha_min)
        epsilon = max(epsilon * epsilon_decay_rate, epsilon_min)

    print("main.py line 103, error_list after all iterations", error_list)
    plt.plot(range(num_episode), error_list,
             label="Q")
    # plt.ylim([0, .5])
    # plt.show()
    plt.savefig("friendQ_all_0.9995_0.9995_timeout25.png")
    plt.clf()

    plt.plot(range(num_episode), delta_list,
             label="Q")
    # plt.ylim([0, .5])
    # plt.show()
    plt.savefig("delta_friendQ_all_0.9995_0.9995_timeout25.png")
    plt.clf()

    plt.plot(range(num_episode), error_list,
             label="Q")
    plt.ylim([0, .5])
    # plt.show()
    plt.savefig("friendQ_small_0.9995_0.9995_timeout25.png")
    plt.clf()

    plt.plot(range(num_episode), delta_list,
             label="Q")
    plt.ylim([0, .5])
    # plt.show()
    plt.savefig("delta_friendQ_small_0.9995_0.9995_timeout25.png")
    plt.clf()

    plt.plot(range(num_episode/100), reward_list_A,
             label="reward for player A")
    plt.savefig("rewards_friendQ_small_0.9995_0.9995_timeout25.png")
    plt.clf()

    # end of game
    print("reward A mean:", np.mean(reward_list_A))
    # print("reward B mean:", np.mean(reward_list_B))
    print("reward A std:", np.std(reward_list_A))
    # print("reward B std:", np.std(reward_list_B))
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
    S_file = open('Friend_Q_Learner.txt', 'a')
    print("alpha_decay_rate", alpha_decay_rate)
    print("epsilon_decay_rate", epsilon_decay_rate)
    error_list = []
    delta_list = []
    reward_list_A = []
    reward_list_A_sub = []

    range_end = num_episode
    q_output = table_foe_Q.return_Q_table()
    print("main.py line 418, q_output", q_output)

    # find player A's Q-values, corresponding to state s (A stands on 2, B stands on 1, B has the ball, i.e. S[(1,2,1)])
    # and A takes action S, B stick, which is  (+4, 0).
    # print("=========main.py line 164, q_output_A===========")
    # print(q_output_A)
    old_q_sa = q_output.loc[(1,2,1), (4,0)]
    old_q = np.array(q_output).flatten()
    print("main.py line 152, old_q", old_q)

    for episode in range(range_end):
        if episode % 100 == 0:
            reward_list_A.append(np.mean(reward_list_A_sub))
            reward_list_A_sub = []

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
                q_output = table_foe_Q.return_Q_table()

                # after one episode, find player A's Q-values, corresponding to state s (as point 2) and action S (+4).
                new_q_sa = q_output.loc[(1,2,1), (4,0)]
                new_q = np.array(q_output).flatten()
                # print("main.py line 45", new_q_sa)
                error_list.append(np.abs(new_q_sa - old_q_sa))
                delta_list.append(rmse(old_q, new_q))
                # print("main.py line 233, rmse(old_q, new_q)", rmse(old_q, new_q))
                reward_list_A_sub.append(reward[0])
                old_q_sa = new_q_sa
                old_q = new_q
                break
        alpha = max(alpha * alpha_decay_rate, alpha_min)
        epsilon = max(epsilon * epsilon_decay_rate, epsilon_min)

    print("main.py line 377, error_list after all iterations", error_list)
    plt.plot(range(num_episode), error_list,
             label="Q")
    # plt.ylim([0, .5])
    # plt.show()
    plt.savefig("foeQ_combined_all_0.9995_0.9995_timeout25.png")
    plt.clf()

    plt.plot(range(num_episode), delta_list,
             label="Q")
    # plt.ylim([0, .5])
    # plt.show()
    plt.savefig("delta_foeQ_combined_all_0.9995_0.9995_timeout25.png")
    plt.clf()

    plt.plot(range(num_episode), error_list,
             label="Q")
    plt.ylim([0, .5])
    # plt.show()
    plt.savefig("foeQ_combined_small_0.9995_0.9995_timeout25.png")
    plt.clf()

    plt.plot(range(num_episode), delta_list,
             label="Q")
    plt.ylim([0, .5])
    # plt.show()
    plt.savefig("delta_foeQ_combined_small_0.9995_0.9995_timeout25.png")
    plt.clf()

    plt.plot(range(num_episode / 100), reward_list_A,
             label="reward for player A")
    plt.savefig("rewards_foeQ_combined_small_0.9995_0.9995_timeout25.png")
    plt.clf()

    # end of game
    print("reward A mean:", np.mean(reward_list_A))
    # print("reward B mean:", np.mean(reward_list_B))
    print("reward A std:", np.std(reward_list_A))
    # print("reward B std:", np.std(reward_list_B))
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
    S_file = open('Friend_Q_Learner.txt', 'a')
    print("alpha_decay_rate", alpha_decay_rate)
    print("epsilon_decay_rate", epsilon_decay_rate)
    error_list = []
    delta_list = []
    reward_list_A = []
    reward_list_A_sub = []

    range_end = num_episode
    q_output = table_ce_Q.return_Q_table()
    print("main.py line 418, q_output", q_output)

    # find player A's Q-values, corresponding to state s (A stands on 2, B stands on 1, B has the ball, i.e. S[(1,2,1)])
    # and A takes action S, B stick, which is  (+4, 0).
    # print("=========main.py line 164, q_output_A===========")
    # print(q_output_A)
    old_q_sa = q_output.loc[(1,2,1), (4,0)]
    old_q = np.array(q_output).flatten()
    print("main.py line 152, old_q", old_q)

    for episode in range(range_end):
        if episode % 100 == 0:
            reward_list_A.append(np.mean(reward_list_A_sub))
            reward_list_A_sub = []

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
                q_output = table_ce_Q.return_Q_table()

                # after one episode, find player A's Q-values, corresponding to state s (as point 2) and action S (+4).
                new_q_sa = q_output.loc[(1,2,1), (4,0)]
                new_q = np.array(q_output).flatten()
                # print("main.py line 45", new_q_sa)
                error_list.append(np.abs(new_q_sa - old_q_sa))
                delta_list.append(rmse(old_q, new_q))
                # print("main.py line 233, rmse(old_q, new_q)", rmse(old_q, new_q))
                reward_list_A_sub.append(reward[0])
                old_q_sa = new_q_sa
                old_q = new_q
                break
        alpha = max(alpha * alpha_decay_rate, alpha_min)
        epsilon = max(epsilon * epsilon_decay_rate, epsilon_min)

    print("main.py line 453, error_list after all iterations", error_list)
    plt.plot(range(num_episode), error_list,
             label="Q")
    # plt.ylim([0, .5])
    # plt.show()
    plt.savefig("ceQ_combined_all_0.9995_0.9995_timeout25.png")
    plt.clf()

    plt.plot(range(num_episode), delta_list,
             label="Q")
    # plt.ylim([0, .5])
    # plt.show()
    plt.savefig("delta_ceQ_combined_all_0.9995_0.9995_timeout25.png")
    plt.clf()

    plt.plot(range(num_episode), error_list,
             label="Q")
    plt.ylim([0, .5])
    # plt.show()
    plt.savefig("ceQ_combined_small_0.9995_0.9995_timeout25.png")
    plt.clf()

    plt.plot(range(num_episode), delta_list,
             label="Q")
    plt.ylim([0, .5])
    # plt.show()
    plt.savefig("delta_ceQ_combined_small_0.9995_0.9995_timeout25.png")
    plt.clf()

    plt.plot(range(num_episode / 100), reward_list_A,
             label="reward for player A")
    plt.savefig("rewards_ceQ_combined_small_0.9995_0.9995_timeout25.png")
    plt.clf()

    # end of game
    print("reward A mean:", np.mean(reward_list_A))
    # print("reward B mean:", np.mean(reward_list_B))
    print("reward A std:", np.std(reward_list_A))
    # print("reward B std:", np.std(reward_list_B))
    print('game over')
    # env.destroy()


if __name__ == "__main__":

    seed = 0  # seed
    np.random.seed(seed)

    '''soccer env'''
    env = SoccerEnv()
    env.seed(seed)

    # table_SARSA = Two_player_SARSA_table(verbose=False)
    # SARSA(num_episode = 10000,
    #       alpha = 0.5,
    #       alpha_decay_rate =  0.9995,
    #       epsilon = 0.5,
    #       epsilon_decay_rate =  0.9995,
    #       gamma = 0.99,
    #       timeout = 25
    #       )

    table_friend_Q = Friend_Q_table(verbose=False)
    friend_Q(num_episode = 10000,
          alpha = 0.5,
          alpha_decay_rate =  0.9995,
          epsilon = 0.5,
          epsilon_decay_rate =  0.9995,
          gamma = 0.99,     # gamma 0.9???
          timeout = 25
          )

    table_foe_Q = Foe_Q_table(verbose=False)
    foe_Q(num_episode = 10000,
          alpha = 0.5,
          alpha_decay_rate =  0.9995,
          epsilon = 0.5,
          epsilon_decay_rate =  0.9995,
          gamma = 0.99,
          timeout = 25
          )

    table_ce_Q = CE_Q_table(verbose=False)
    ce_Q(num_episode=10000,
          alpha=0.5,
          alpha_decay_rate=0.9995,
          epsilon=0.5,
          epsilon_decay_rate=0.9995,
          gamma=0.99,
          timeout=25
          )




