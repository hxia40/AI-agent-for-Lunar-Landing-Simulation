import gym
import numpy as np
import random
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from _DQN import DQN


def plot_figure1(df, chart_name, title, x_axis_label, y_axis_label):
    plt.rcParams.update({'font.size': 17})
    df['Rolling mean'] = df[df.columns[0]].rolling(100).mean()
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_xlabel(x_axis_label)
    plot.set_ylabel(y_axis_label)
    plt.ylim((-500, 300))
    fig = plot.get_figure()
    plt.legend().set_visible(False)
    plt.savefig(chart_name)

    
def test_already_trained_model(trained_model):
    rewards_list = []
    num_test_episode = 100
    env = gym.make("LunarLander-v2")
    print("Starting Testing of the trained model...")

    step_count = 1000

    for test_episode in range(num_test_episode):
        state = env.reset()
        state_dimension = env.observation_space.shape[0]
        state = np.reshape(state, [1, num_observation_space])
        episode_reward = 0
        for step in range(step_count):
#             env.render()
            predicted_action = np.argmax(trained_model.predict(state)[0])
            state_, reward, done, info = env.step(predicted_action)
            state_ = np.reshape(new_state, [1, state_dimension])
            state = state_
            episode_reward += reward
            if done:
                break
        rewards_list.append(episode_reward)
        print("Episode =", episode, "Episode Reward = ", episode_reward)

    return rewards_list

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    # set seeds
    env.seed(0)
    np.random.seed(0)

    # setting up params
    alpha = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    training_episodes = 1000
    mem_size=100000
    mem_batch_size=100

    model = DQN(env, alpha, gamma, epsilon, epsilon_decay, mem_size, mem_batch_size)
    model.train(training_episodes, stop_good_enough=False)

    # Save trained model
    model.save("trained_model_mem_1000000_batch_100_singleNN_10000.h5")

    # Save Rewards list
    pickle.dump(model.rewards_list, open("list_mem_1000000_batch_100_singleNN_10000.p", "wb"))
    rewards_list = pickle.load(open("list_mem_1000000_batch_100_singleNN_10000.p", "rb"))

    # plot reward in graph
    reward_df = pd.DataFrame(rewards_list)
    plot_figure1(reward_df, "Figure_mem_1000000_batch_100_singleNN_10000", "Reward for each training episode", "Episode","Reward")



