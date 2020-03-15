import gym
import numpy as np
import random
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from _DQN import DQN



def plott(df, figure_name, x_label, y_label):
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = df.plot(linewidth=1, figsize=(15, 8))
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)
    plt.ylim((-500, 300))
    plt.savefig(figure_name)


def gamma_scan():
    print('gamma scan starts')
    env = gym.make('LunarLander-v2')
    env.seed(0)
    np.random.seed(0)

    alpha = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma_list = [0.999, 0.99, 0.9, 0.8]
    num_episodes = 1000
    mem_size=100000
    mem_batch_size=100
    rewards_list = []
    
    for gamma in gamma_list:
        model = DQN(env, alpha, gamma, epsilon, epsilon_decay, mem_size, mem_batch_size)
        print("Training model for epsilon decay: {}".format(epsilon_decay))
        model.train(num_episodes=num_episodes, stop_good_enough=False)
        rewards_list.append(model.rewards_list)

    stacked_reward_df = pd.DataFrame(index=pd.Series(range(1, num_episodes+1)))
    for i in range(len(gamma_list)):
        col_name = "epsilon_decay = "+ str(gamma_list[i])
        stacked_reward_df[col_name] = rewards_list[i]
    plott(stacked_reward_df, "Figure5_gamma_scan", "Episodes", "Reward")
    

    
def alpha_scan():
    print('alpha_scan starts')
    env = gym.make('LunarLander-v2')
    env.seed(0)
    np.random.seed(0)

    alpha_list = [0.0001, 0.001, 0.01, 0.1]
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    num_episodes = 1000
    mem_size=100000
    mem_batch_size=100
    rewards_list = []
    
    for alpha in alpha_list:
        model = DQN(env, alpha, gamma, epsilon, epsilon_decay, mem_size, mem_batch_size)
        print("Training model for epsilon decay: {}".format(epsilon_decay))
        model.train(num_episodes=num_episodes, stop_good_enough=False)
        rewards_list.append(model.rewards_list)

    stacked_reward_df = pd.DataFrame(index=pd.Series(range(1, num_episodes+1)))
    for i in range(len(alpha_list)):
        col_name = "epsilon_decay = "+ str(alpha_list[i])
        stacked_reward_df[col_name] = rewards_list[i]
    plott(stacked_reward_df, "Figure3_alpha_scan", "Episodes", "Reward")
    
    
def epsilon_decay_scan():
    print('epsilon_decay_scan starts')
    env = gym.make('LunarLander-v2')
    env.seed(0)
    np.random.seed(0)

    alpha = 0.001
    epsilon = 1.0
    epsilon_decay_list = [0.999, 0.995, 0.99, 0.9]
    gamma = 0.99
    num_episodes = 1000
    mem_size=100000
    mem_batch_size=100
    rewards_list = []
    
    for epsilon_decay in epsilon_decay_list:
        model = DQN(env, alpha, gamma, epsilon, epsilon_decay, mem_size, mem_batch_size)
        print("Training model for epsilon decay: {}".format(epsilon_decay))
        model.train(num_episodes=num_episodes, stop_good_enough=False)
        rewards_list.append(model.rewards_list)

    stacked_reward_df = pd.DataFrame(index=pd.Series(range(1, num_episodes+1)))
    for i in range(len(epsilon_decay_list)):
        col_name = "epsilon_decay = "+ str(epsilon_decay_list[i])
        stacked_reward_df[col_name] = rewards_list[i]
    plott(stacked_reward_df, "Figure4_epsilon_decay_scan", "Episodes", "Reward")

if __name__ == '__main__':
   
    alpha_scan()
    epsilon_decay_scan()
    gamma_scan()
    


