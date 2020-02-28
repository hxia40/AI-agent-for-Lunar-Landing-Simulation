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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from taxi import TaxiEnv

def Q_HW4(num_episode = 10000,
          learning_rate = 0.01,
          epsilon = 0.1
          ):  # learning fromzen lake using SARSA
    S_file = open('QLearner.txt', 'a')
    print(num_episode,learning_rate)
    episode_list = []
    reward_list = []
    reward_list_jr = []
    time_list = []
    time_list_jr = []

    range_end = num_episode
<<<<<<< HEAD
    print("range_end:", range_end)
    for episode in range(range_end):

        # epsilon = (1 - math.log(episode+1, 10) / math.log(range_end, 10))
        # alpha = learning_rate

        # if episode < (range_end/2):  # for the first 50000 samples, decaying the epsilon. the rest will con constant epsilon of
        #     epsilon = (1 - math.log(episode+1, 10) / math.log(range_end/2, 10))
        # else:
        #     pass

        if episode < 500000:  # for the first 50000 samples, decaying the epsilon. the rest will con constant epsilon of
            epsilon = 1
            alpha = 0.25
        elif episode < 1000000:
            epsilon = 0.9
            alpha = 0.1
        elif episode < 1100000:
            epsilon = 0.8
            alpha = 0.05
        elif episode < 1200000:
            epsilon = 0.7
            alpha = 0.04
        elif episode < 1300000:
            epsilon = 0.6
            alpha = 0.03
        elif episode < 1400000:
            epsilon = 0.5
            alpha = 0.02
        elif episode < 1500000:
            epsilon = 0.4
            alpha = 0.01
        elif episode < 1600000:
            epsilon = 0.3
            alpha = 0.01
        elif episode < 1700000:
            epsilon = 0.2
            alpha = 0.01
        elif episode < 1800000:
            epsilon = 0.1
            alpha = 0.01
        elif episode < 1900000:
            epsilon = 0.01
            alpha = 0.01
        else:
            epsilon = 0.001
        if (episode == 1000) or ((episode + 1) % (range_end / 100)) == 0:
=======
    old_epsilon = 1

    for episode in range(range_end):

        # alpha = (1 - math.log(episode+1, 10) / math.log(range_end, 10))/10
        alpha = learning_rate
        # epsilon = (1 - math.log(episode + 1, 10) / math.log(range_end, 10)) * 2
        # if num_episode <= range_end/2:         # first drop the learning rate:
        #     epsilon = (1 - math.log(episode+1, 10) / math.log(range_end/2, 10)) * 2
        #     old_epsilon = epsilon
        # else:
        #     epsilon = old_epsilon

        # epsilon = epsilon
        if (episode + 1) % (range_end / 100) == 0:
>>>>>>> 5e761dfdaf2a0b31a1953e31ff471bcb1a9f26ae
            print("episode = ", episode + 1, "learnng rate = ", alpha, "epsilon = ", epsilon, "reward = ", np.mean(reward_list_jr))
            episode_list.append(episode + 1)
            time_list.append(np.mean(time_list_jr))
            time_list_jr = []
            reward_list.append(np.mean(reward_list_jr))
            reward_list_jr = []

            # q_output = QL_HW4.return_Q_table()
            # q_output.index = q_output.index.astype(int)
            # q_output = q_output.sort_index()
            # q_output.to_csv('alpha_%d_epsilon_%d_episode_%d.csv' % (alpha, epsilon, episode))
        # initial observation
        observation = env_HW4.reset()
        # print("line 65 obs after reset:", observation)
        start_time = time.time()
        while True:

            # if (episode + 1) % (range_end / 100) == 0:
            #     env_HW4.render()

            # QL choose action based on observation
            action = QL_HW4.choose_action(str(observation), epsilon)
            # print env_HW4.step(action)

            # QL take action and get next observation and reward
            observation_, reward, done, info = env_HW4.step(action)
<<<<<<< HEAD
            reward_list_jr.append(reward)
=======
            # print(observation_)
            reward_list_jr.append(reward)

>>>>>>> 5e761dfdaf2a0b31a1953e31ff471bcb1a9f26ae
            # QL learn from this transition
            QL_HW4.learn(str(observation), action, reward, str(observation_), alpha)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                time_list_jr.append(time.time()-start_time)

<<<<<<< HEAD
                break

    q_output = QL_HW4.return_Q_table()
    print(type(q_output), "=======q_output========\n", q_output)

=======
    q_output = QL_HW4.return_Q_table()
    print(type(q_output), "=======q_output========\n", q_output)
>>>>>>> 5e761dfdaf2a0b31a1953e31ff471bcb1a9f26ae
    q_output.index = q_output.index.astype(int)
    q_output = q_output.sort_index()
    print("=======sorted========\n", q_output)
    print('game over')
    # env.destroy()
    return q_output


def SARSA_HW4(num_episode = 10000,
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
        observation = env_HW4.reset()   # (1) Initialize S
        start_time = time.time()
        # QL choose action based on observation # (2) choose A from S using policy derived Q (e.g. sigma-greedy)
        action = SARSA.choose_action(str(observation))
        # print("new episode!, observation is:", observation)
        while True:                     # (3) Repeat (for each step of this episode)
            # # fresh env
            # env_HW3.render()

            # QL take action and get next observation and reward  # R(1) Take action A, observe R, S'
            observation_, reward, done, info = env_HW4.step(action)
            # print("a, obs_, r, done, info:", action, observation_, reward, done, info)
            reward_list_jr.append(reward)

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

                break

    q_output = SARSA.return_Q_table()
    print(type(q_output), "=======q_output========\n", q_output)

    q_output.index = q_output.index.astype(int)
    q_output = q_output.sort_index()
    print("=======sorted========\n", q_output)


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

if __name__ == "__main__":

    '''Implement a basic version of the Q-learning algorithm and use it to solve the taxi domain. The agent should
    explore the MDP, collect data to learn the optimal policy and the optimal Q-value function. (Be mindful of
    how you handle terminal states, typically if St is a terminal state, V (St+1) = 0). Use gamma = 0.90. 
    
    Also, you will
    see how an Epsilon-Greedy strategy can find the optimal policy despite finding sub-optimal Q-values. As we
    are looking for optimal Q-values you will have to carefully consider your exploration strategy.
    Evaluate your agent using the OpenAI gym 0.14.0 Taxi-v2 environment. Install OpenAI Gym 0.14.0 with
    pip install gym==0.14.0'''
<<<<<<< HEAD
    env_HW4 = gym.make('Taxi-v2')
=======
    # env_HW4 = gym.make('Taxi-v2').unwrapped
    env_HW4 = TaxiEnv()
>>>>>>> 5e761dfdaf2a0b31a1953e31ff471bcb1a9f26ae

    '''Taxi-v2 - Q-learning'''
    print("Taxi-v2")
    for i in range(1):
        QL_HW4 = QLearningTable(actions=list(range(env_HW4.nA)),
                                  # learning_rate=0.1,
                                  reward_decay=0.90,   # gamma
                                  # epsilon=0.2,
                                  verbose=True)
<<<<<<< HEAD
        Q_output = Q_HW4(num_episode = 2500000,         # Sammy runed 10000000 episodes to get stable updates.
                                                        # Q table only updated for 2500000 episodes
                                                        # sometimes only makes 90% score
                         learning_rate=0.01)     # function to execute the q-learner, shown above
        print(Q_output)
        Q_output.to_csv('test.csv')


        # print(Q_output.iloc[1,1],
        #       Q_output.iloc[462,4], # 11.374402515
        #       Q_output.iloc[398, 3], # 4.348907
        #       Q_output.iloc[253, 0],  # 0.5856821173
        #       Q_output.iloc[377, 1],  # 9.683
        #       Q_output.iloc[83, 5]  # 12.8232660372
        #       )
=======
        Q_output = Q_HW4(num_episode = 200000,
                         learning_rate=0.1,
                         epsilon = 0.05)     # function to execute the q-learner, shown above
        print(Q_output)
        Q_output.to_csv('test.csv')
        print(Q_output.iloc[1,1],
              Q_output.iloc[462,4], # 11.374402515
              Q_output.iloc[398, 3], # 4.348907
              Q_output.iloc[253, 0],  # 0.5856821173
              Q_output.iloc[377, 1],  # 9.683
              Q_output.iloc[83, 5]  # 12.8232660372
              )
>>>>>>> 5e761dfdaf2a0b31a1953e31ff471bcb1a9f26ae

    '''taxi v2 - SARSA'''

    # SARSA = SARSA_TABLE(actions=list(range(env_HW4.nA)),
    #                         reward_decay=0.9,  # gamma
    #                         e_greedy=0.1,  # epslion
    #                         verbose=1)
    # SARSA_HW4(num_episode=10000,
    #               learning_rate=0.1)  # alpha\



