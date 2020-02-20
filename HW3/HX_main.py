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


def Q_FL0(learning_rate = 0.01):  # learning fromzen lake using Q-leaner
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
        ''' for each episode, the algorithm on Figure 6.7 of Sutton book is executed:
        Initialize S
        Repeat (for each step of this episode)
            choose A from S using policy derived Q (e.g. sigma-greedy)
            Take action A, observe R, S'
            Q(S, A) <- Q(S,A) + alpha[R + gamma * max_for_aQ(S',a) - Q(S,A)
            S <- S'
        until S is terminal 
        '''
        # initial observation
        observation = env_FL0.reset()   # (1) Initialize S
        start_time = time.time()
        while True:         # (2) Repeat (for each step of this episode)
            # # fresh env
            # env_FL0.render()

            # QL choose action based on observation # R(1) choose A from S using policy derived Q (e.g. sigma-greedy)
            action = QL_FL0.choose_action(str(observation))
            # print env_FL0.step(action)

            # QL take action and get next observation and reward # R(2) Take action A, observe R, S'
            observation_, reward, done, info = env_FL0.step(action)

            # QL learn from this transition # R(3) Q(S, A) <- Q(S,A) + alpha[R + gamma * max_for_aQ(S',a) - Q(S,A)
            QL_FL0.learn(str(observation), action, reward, str(observation_), alpha)

            # swap observation   # R(4) S <- S'
            observation = observation_

            # break while loop when end of this episode    # (3) until S is terminal
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

def Q_HW3(learning_rate = 0.01):   # learning fromzen lake using Q-leaner
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
        observation = env_HW3.reset()
        start_time = time.time()
        while True:
            # # fresh env
            # env_HW3.render()

            # QL choose action based on observation
            action = QL_HW3.choose_action(str(observation))
            # print env_HW3.step(action)

            # QL take action and get next observation and reward
            observation_, reward, done, info = env_HW3.step(action)

            # QL learn from this transition
            QL_HW3.learn(str(observation), action, reward, str(observation_), alpha)

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

def SARSA_HW3(num_episode = 10000,
              learning_rate = 0.01,
              ):  # learning fromzen lake using SARSA
    S_file = open('SLearner.txt', 'a')
    print(num_episode,learning_rate)
    episode_list = []
    reward_list = []
    reward_list_jr = []
    time_list = []
    time_list_jr = []

    range_end = num_episode
    for episode in range(range_end):
        # print("episode:", episode)
        # alpha = (1 - math.log(episode+1, 10) / math.log(range_end, 10))/10
        alpha = learning_rate
        if (episode + 1) % (int(range_end / 10)) == 0:
            print("episode = ", episode + 1, "learnng rate = ", alpha, "reward = ", np.mean(reward_list_jr))
            episode_list.append(episode + 1)
            time_list.append(np.mean(time_list_jr))
            time_list_jr = []
            reward_list.append(np.mean(reward_list_jr))
            reward_list_jr = []

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
        observation = env_HW3.reset()   # (1) Initialize S
        start_time = time.time()
        # QL choose action based on observation # (2) choose A from S using policy derived Q (e.g. sigma-greedy)
        action = SARSA.choose_action(str(observation))
        # print("new episode!, observation is:", observation)
        while True:                     # (3) Repeat (for each step of this episode)
            # # fresh env
            # env_HW3.render()

            # QL take action and get next observation and reward  # R(1) Take action A, observe R, S'
            observation_, reward, done, info = env_HW3.step(action)
            # print("a, obs_, r, done, info:", action, observation_, reward, done, info)

            # QL choose action based on observation  # R(2) choose A' from S' using policy derived Q (e.g. sigma-greedy)
            action_ = SARSA.choose_action(str(observation_))
            # print env_HW3.step(action_)

            # QL learn from this transition  # R(3) Q(S, A) <- Q(S,A) + alpha[R + gamma * Q(S',A') - Q(S,A)
            SARSA.learn(str(observation), action, reward, str(observation_), action_, alpha)

            # swap observation      #  R(4) S <- S'; A <- A'
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:

                time_list_jr.append(time.time()-start_time)
                reward_list_jr.append(reward)
                break

    q_output = SARSA.return_Q_table()
    print(type(q_output), "=======q_output========\n", q_output)

    q_output.index = q_output.index.astype(int)
    q_output = q_output.sort_index()
    print("=======sorted========\n", q_output)

    policy = np.argmax(np.array(q_output), axis=1)
    print(type(q_output), "=======policy========\n", policy)

    answer_list = []
    for i in policy:
        if i == 0:
            answer_list.append('<')
        elif i == 1:
            answer_list.append('v')
        elif i == 2:
            answer_list.append('>')
        elif i == 3:
            answer_list.append('^')
    answer_list = ','.join(answer_list)
    print("====answer:====\n",answer_list)
     # S_file.write('episodes:')
    # S_file.write(str(episode_list))
    # S_file.write('\n')
    # S_file.write('rewards:')
    S_file.write(str(reward_list))
    S_file.write('\n')
    # S_file.write('time_consumption:')
    # S_file.write(str(time_list))
    S_file.close()

    # end of game
    print('game over')
    # env.destroy()


def map_reshape(map_string):
    n = int(len(amap) ** 0.5)
    return [amap[i:i + n] for i in range(0, len(amap), n)]


if __name__ == "__main__":

    '''You must train your agent with an epsilon-greedy exploration strategy, using NumPy's numpy.random.randint 
    function to select random actions'''
    ''' HW3 example 2'''
    seed = 983459                                 # seed
    np.random.seed(seed)
    amap = 'SFFFFFFFFFFFFFFFFFFHFFFFG'                      # map

    amap_reshaped = map_reshape(amap)
    print(amap_reshaped)
    '''FromzenLake env'''
    env_HW3 = gym.envs.toy_text.frozen_lake.FrozenLakeEnv(desc=amap_reshaped).unwrapped
    # env_HW3 = gym.make('FrozenLake-v0', desc=amap_reshaped).unwrapped
    env_HW3.seed(seed)
    '''FrozenLake unwrapped - SARSA'''

    SARSA = SARSA_TABLE(actions=list(range(env_HW3.nA)),
                                reward_decay=0.91,    # gamma
                                e_greedy=0.13,     # epslion
                                verbose = 1)
    SARSA_HW3(num_episode = 42271,
              learning_rate = 0.12)                # alpha

    # ''' HW3 example 1'''
    # seed = 741684                                 # seed
    # np.random.seed(seed)
    # amap = 'SFFFHFFFFFFFFFFG'                      # map
    # amap_reshaped = map_reshape(amap)
    # print(amap_reshaped)
    # '''FromzenLake env'''
    # env_HW3 = gym.envs.toy_text.frozen_lake.FrozenLakeEnv(desc=amap_reshaped).unwrapped
    # # env_HW3 = gym.make('FrozenLake-v0', desc=amap_reshaped).unwrapped
    # env_HW3.seed(seed)
    # '''FrozenLake unwrapped - SARSA'''
    #
    # SARSA = SARSA_TABLE(actions=list(range(env_HW3.nA)),
    #                             reward_decay=1.0,    # gamma
    #                             e_greedy=0.29,     # epslion
    #                             verbose = 2)
    # SARSA_HW3(num_episode = 14697,
    #           learning_rate = 0.25)                # alpha



    '''FrozenLake - Q-learning'''

    # print("QLearningTable")
    # for i in range(1):
    #     QL_FL0 = QLearningTable(actions=list(range(env_FL0.nA)),
    #                             # learning_rate=0.1,
    #                             reward_decay=0.99,
    #                             e_greedy=0.9,
    #                             verbose = 1)
    #     Q_FL0(learning_rate = 0.1)

    '''FrozenLake unwrapped- Q-learning'''

    # print("QLearningTable")
    # for i in range(1):
    #     QL_HW3 = QLearningTable(actions=list(range(env_HW3.nA)),
    #                             # learning_rate=0.1,
    #                             reward_decay=0.99,
    #                             e_greedy=0.9,
    #                             verbose =1)
    #     Q_HW3(learning_rate = 0.1)







