"""
Reinforcement learning


Reference:
Morzan,
wfeng,
Moustafa Alzantot (malzantot@ucla.edu)

"""
import numpy as np
# from HX_policy_iteration import PI
# from HX_value_iteration import VI
from HX_QLearner import QLearningTable, QLearningTableNC
from HX_SARSA import SARSA_TABLE
# from stocks_env import StocksEnv
from HX_maze import generate_random_map, FrozenLakeEnv
import gym
import time
import math
import random
import matplotlib.pyplot as plt
import pandas as pd



def Q_FL0(learning_rate = 0.01):            # learning fromzen lake using Q-leaner
    Q_file = open('QLearner.txt', 'a')

    episode_list = []
    reward_list = []
    reward_list_jr = []
    time_list = []
    time_list_jr = []

    range_end = 10000
    for episode in range(range_end):
        # alpha = (1 - math.log(episode+1, 10) / math.log(range_end, 10))/10
        alpha = learning_rate
        if (episode + 1) % (range_end / 100) == 0:
            print("episode = ", episode + 1, "learnng rate = ", alpha, "reward = ", np.mean(reward_list_jr))
            episode_list.append(episode + 1)
            time_list.append(np.mean(time_list_jr))
            time_list_jr = []
            reward_list.append(np.mean(reward_list_jr))

            reward_list_jr = []
        # initial observation
        observation = env_FL0.reset()
        start_time = time.time()
        while True:
            # # fresh env
            # env_FL0.render()

            # QL choose action based on observation
            action = QL_FL0.choose_action(str(observation))
            # print env_FL0.step(action)

            # QL take action and get next observation and reward
            observation_, reward, done, info = env_FL0.step(action)

            # QL learn from this transition
            QL_FL0.learn(str(observation), action, reward, str(observation_), alpha)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                time_list_jr.append(time.time()-start_time)
                reward_list_jr.append(reward)
                break

    # Q_file.write('episodes:')
    # Q_file.write(str(episode_list))
    # Q_file.write('\n')
    # Q_file.write('rewards:')
    Q_file.write(str(reward_list))
    Q_file.write('\n')
    # Q_file.write('time_consumption:')
    # Q_file.write(str(time_list))
    Q_file.close()

    # end of game
    print('game over')
    # env.destroy()
def Q_NC(random_seed=1):
    Q_file = open('QLearnerNC.txt', 'w')
    episode_list = []
    reward_list = []
    reward_list_jr = []
    time_list = []
    time_list_jr = []

    range_end = 100000
    env_NC.seed(random_seed)
    for episode in range(range_end):
        # alpha = (1 - math.log(episode+1, 10) / math.log(range_end, 10))/10
        alpha = 0.01
        if (episode + 1) % (range_end / 100) == 0:
            print("episode = ", episode + 1, "learnng rate = ", alpha, "reward = ", np.mean(reward_list_jr))
            episode_list.append(episode + 1)
            time_list.append(np.mean(time_list_jr))
            time_list_jr = []
            reward_list.append(np.mean(reward_list_jr))

            reward_list_jr = []
        # initial observation
        observation = env_NC.reset()
        start_time = time.time()
        while True:
            # # fresh env
            # env_NC.render()

            # QL choose action based on observation
            action = QL_NC.choose_action(observation)
            # print env_NC.step(action)

            # QL take action and get next observation and reward
            observation_, reward, done, info = env_NC.step(action)
            # print action
            # print env_NC.step(action)

            # QL learn from this transition
            QL_NC.learn(observation, action, reward, observation, alpha)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                time_list_jr.append(time.time()-start_time)
                reward_list_jr.append(reward)
                break

    Q_file.write('episodes:')
    Q_file.write(str(episode_list))
    Q_file.write('\n')
    Q_file.write('rewards:')
    Q_file.write(str(reward_list))
    Q_file.write('\n')
    Q_file.write('time_consumption:')
    Q_file.write(str(time_list))

    # end of game
    print('game over')
    # env.destroy()


def Q_AnyTrading(learning_rate = 0.01):
    print("==========env information:========")
    print("> shape:", env_AT.shape)
    print("> df.shape:", env_AT.df.shape)
    print("> prices.shape:", env_AT.prices.shape)
    print("> signal_features.shape:", env_AT.signal_features.shape)
    print("> max_possible_profit:", env_AT.max_possible_profit())

    Q_file = open('AnyTrading_QLearner.txt', 'a')

    episode_list = []
    reward_list = []
    reward_list_jr = []
    time_list = []
    time_list_jr = []

    range_end = 10
    for episode in range(range_end):
        # alpha = (1 - math.log(episode+1, 10) / math.log(range_end, 10))/10
        alpha = learning_rate
        if (episode + 1) % (range_end / 5) == 0:
            print("episode = ", episode + 1, "learnng rate = ", alpha, "reward = ", np.mean(reward_list_jr))

            episode_list.append(episode + 1)
            time_list.append(np.mean(time_list_jr))
            time_list_jr = []
            reward_list.append(np.mean(reward_list_jr))

            reward_list_jr = []
        # initial observation
        observation = env_AT.reset()
        start_time = time.time()
        while True:
            # # fresh env
            # env_AT.render()

            # QL choose action based on observation
            action = QL_AnyTrading.choose_action(str(observation))
            # print env_FL0.step(action)

            # QL take action and get next observation and reward
            observation_, reward, done, info = env_AT.step(action)
            # print("info:", action, reward)

            # QL learn from this transition
            QL_AnyTrading.learn(str(observation), action, reward, str(observation_), alpha)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                time_list_jr.append(time.time()-start_time)
                reward_list_jr.append(reward)
                # print("===========DONE==========:", env_AT.step(action))
                break
    print("===========DONE==========:", env_AT.step(action))
    plt.cla()
    env_AT.render_all()
    plt.show()


    # Q_file.write('episodes:')
    # Q_file.write(str(episode_list))
    # Q_file.write('\n')
    # Q_file.write('rewards:')
    Q_file.write(str(reward_list))
    Q_file.write('\n')
    # Q_file.write('time_consumption:')
    # Q_file.write(str(time_list))
    Q_file.close()

    # end of game
    print('game over')
    # env.destroy()


if __name__ == "__main__":

    '''You must train your agent with an epsilon-greedy exploration strategy, using NumPy's numpy.random.randint
function to select random actions'''

    '''FromzenLake env'''
    # env_FL0 = FrozenLakeEnv(desc=generate_random_map(size=32, p=0.99), map_name=None, is_slippery=False)
    env_FL0 = FrozenLakeEnv(desc=None, map_name='4x4', is_slippery=False)
    # env_HW3 = gym.envs.toy_text.frozen_lake.FrozenLakeEnv().unwrapped

    '''FrozenLake - Q-learning'''

    print("QLearningTable")
    for i in range(1):
        QL_FL0 = QLearningTable(actions=list(range(env_FL0.nA)),
                                # learning_rate=0.1,
                                reward_decay=0.99,
                                e_greedy=0.9,
                                verbose = True)
        Q_FL0(learning_rate = 0.1)


