import gym
import numpy as np
import random
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from _DQN import DQN
from keras.models import load_model

def plot_figure2(df, figure_name, x_label, y_label):
    df['mean'] = df[df.columns[0]].mean()
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)
    plt.ylim((0, 300))
    plt.xlim((0, 100))
    plt.legend().set_visible(False)
    plt.savefig(figure_name)
    
def test_trained_model(trained_model):
    rewards_list = []
    num_test_episode = 1000
    env = gym.make("LunarLander-v2")
    env.seed(0)
    np.random.seed(0)
    print("Starting Testing of the trained model...")

    step_count = 1000

    for episode in range(num_test_episode):
        state = env.reset()
        state_dimension = env.observation_space.shape[0]
        state = np.reshape(state, [1, state_dimension])
        episode_reward = 0
        for step in range(step_count):
#             env.render()
            predicted_action = np.argmax(trained_model.predict(state)[0])
            state_, reward, done, info = env.step(predicted_action)
            state_ = np.reshape(state_, [1, state_dimension])
            state = state_
            episode_reward += reward
            if done:
                break
        rewards_list.append(episode_reward)
        print("Episode =", episode, "Episode Reward = ", episode_reward)

    return rewards_list

if __name__ == '__main__':


    trained_model = load_model("train_model_mem_1000_batch_100_doubleNN_100.h5")
    test_rewards = test_trained_model(trained_model)

    plot_figure2(pd.DataFrame(test_rewards), "Figure2_mem_1000_batch_100_doubleNN_100","Episode", "Reward")
    print("Game over")


