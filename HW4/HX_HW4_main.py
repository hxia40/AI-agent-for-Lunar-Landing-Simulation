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
from HX_QLearner import QLearningTable
from HX_SARSA import SARSA_TABLE
# from stocks_env import StocksEnv
from HX_maze import generate_random_map, FrozenLakeEnv
import gym
import time
import math
import random
import matplotlib.pyplot as plt
import pandas as pd

def Q_HW4(num_episode = 10000,
          learning_rate = 0.01,
          ):  # learning fromzen lake using SARSA
    S_file = open('QLearner.txt', 'a')
    print(num_episode,learning_rate)
    episode_list = []
    reward_list = []
    reward_list_jr = []
    time_list = []
    time_list_jr = []

    range_end = num_episode

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
        observation = env_HW4.reset()
        start_time = time.time()
        while True:
            # # fresh env
            # env_HW4.render()

            # QL choose action based on observation
            action = QL_HW4.choose_action(str(observation))
            # print env_HW4.step(action)

            # QL take action and get next observation and reward
            observation_, reward, done, info = env_HW4.step(action)

            # QL learn from this transition
            QL_HW4.learn(str(observation), action, reward, str(observation_), alpha)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                time_list_jr.append(time.time()-start_time)
                reward_list_jr.append(reward)
                break


    q_output = QL_HW4.return_Q_table()
    print(type(q_output), "=======q_output========\n", q_output)
    print('game over')
    # env.destroy()
    return q_output


if __name__ == "__main__":

    '''Implement a basic version of the Q-learning algorithm and use it to solve the taxi domain. The agent should
    explore the MDP, collect data to learn the optimal policy and the optimal Q-value function. (Be mindful of
    how you handle terminal states, typically if St is a terminal state, V (St+1) = 0). Use gamma = 0.90. 
    
    Also, you will
    see how an Epsilon-Greedy strategy can find the optimal policy despite finding sub-optimal Q-values. As we
    are looking for optimal Q-values you will have to carefully consider your exploration strategy.
    Evaluate your agent using the OpenAI gym 0.14.0 Taxi-v2 environment. Install OpenAI Gym 0.14.0 with
    pip install gym==0.14.0'''
    env_HW4 = gym.make('Taxi-v2').unwrapped

    '''Taxi-v2 - Q-learning'''
    print("Taxi-v2")
    for i in range(1):
        QL_HW4 = QLearningTable(actions=list(range(env_HW4.nA)),
                                  # learning_rate=0.1,
                                  reward_decay=0.90,   # gamma
                                  epsilon=0.1,
                                  verbose=True)
        Q_output = Q_HW4(num_episode = 10,
                         learning_rate=0.1)     # function to execute the q-learner, shown above
        print(Q_output)
        Q_output.to_csv('test.csv')
        print(Q_output.iloc[1,1],
              # Q_output[462,4], # 11.374402515
              # Q_output[398, 3], # 4.348907
              # Q_output[253, 0],  # 0.5856821173
              # Q_output[377, 1],  # 9.683
              # Q_output[83, 5]  # 12.8232660372
              )



