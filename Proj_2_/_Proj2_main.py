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

def plot_experiments(df, chart_name, title, x_axis_label, y_axis_label, y_limit):
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = df.plot(linewidth=1, figsize=(15, 8), title=title)
    plot.set_xlabel(x_axis_label)
    plot.set_ylabel(y_axis_label)
    plt.ylim(y_limit)
    fig = plot.get_figure()
    fig.savefig(chart_name)


def run_experiment_for_gamma():
    print('Running Experiment for gamma...')
    env = gym.make('LunarLander-v2')

    # set seeds
    env.seed(21)
    np.random.seed(21)

    # setting up params
    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma_list = [0.99, 0.9, 0.8, 0.7]
    training_episodes = 1000
    mem_size=100000
    mem_batch_size=100

    rewards_list_for_gammas = []
    for gamma_value in gamma_list:
        # save_dir = "hp_gamma_"+ str(gamma_value) + "_"
        model = DQN(env, lr, gamma_value, epsilon, epsilon_decay, mem_size, mem_batch_size)
        print("Training model for Gamma: {}".format(gamma_value))
        model.train(training_episodes, False)
        rewards_list_for_gammas.append(model.rewards_list)

    pickle.dump(rewards_list_for_gammas, open("rewards_list_for_gammas.p", "wb"))
    rewards_list_for_gammas = pickle.load(open("rewards_list_for_gammas.p", "rb"))

    gamma_rewards_pd = pd.DataFrame(index=pd.Series(range(1, training_episodes + 1)))
    for i in range(len(gamma_list)):
        col_name = "gamma=" + str(gamma_list[i])
        gamma_rewards_pd[col_name] = rewards_list_for_gammas[i]
    plot_experiments(gamma_rewards_pd, "Figure 4: Rewards per episode for different gamma values",
                     "Figure 4: Rewards per episode for different gamma values", "Episodes", "Reward", (-600, 300))


def run_experiment_for_lr():
    print('Running Experiment for learning rate...')
    env = gym.make('LunarLander-v2')

    # set seeds
    env.seed(21)
    np.random.seed(21)

    # setting up params
    lr_values = [0.0001, 0.001, 0.01, 0.1]
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    training_episodes = 1000
    mem_size=100000
    mem_batch_size=100
    
    rewards_list_for_lrs = []
    for lr_value in lr_values:
        model = DQN(env, lr_value, gamma, epsilon, epsilon_decay, mem_size, mem_batch_size)
        print("Training model for LR: {}".format(lr_value))
        model.train(training_episodes, False)
        rewards_list_for_lrs.append(model.rewards_list)

    pickle.dump(rewards_list_for_lrs, open("rewards_list_for_lrs.p", "wb"))
    rewards_list_for_lrs = pickle.load(open("rewards_list_for_lrs.p", "rb"))

    lr_rewards_pd = pd.DataFrame(index=pd.Series(range(1, training_episodes + 1)))
    for i in range(len(lr_values)):
        col_name = "lr="+ str(lr_values[i])
        lr_rewards_pd[col_name] = rewards_list_for_lrs[i]
    plot_experiments(lr_rewards_pd, "Figure 3: Rewards per episode for different learning rates", "Figure 3: Rewards per episode for different learning rates", "Episodes", "Reward", (-2000, 300))


def run_experiment_for_ed():
    print('Running Experiment for epsilon decay...')
    env = gym.make('LunarLander-v2')

    # set seeds
    env.seed(21)
    np.random.seed(21)

    # setting up params
    lr = 0.001
    epsilon = 1.0
    ed_values = [0.999, 0.995, 0.990, 0.9]
    gamma = 0.99
    training_episodes = 1000
    mem_size=100000
    mem_batch_size=100

    rewards_list_for_ed = []
    for ed in ed_values:
        save_dir = "hp_ed_"+ str(ed) + "_"
        model = DQN(env, lr, gamma, epsilon, ed, mem_size, mem_batch_size)
        print("Training model for ED: {}".format(ed))
        model.train(training_episodes, False)
        rewards_list_for_ed.append(model.rewards_list)

    pickle.dump(rewards_list_for_ed, open("rewards_list_for_ed.p", "wb"))
    rewards_list_for_ed = pickle.load(open("rewards_list_for_ed.p", "rb"))

    ed_rewards_pd = pd.DataFrame(index=pd.Series(range(1, training_episodes+1)))
    for i in range(len(ed_values)):
        col_name = "epsilon_decay = "+ str(ed_values[i])
        ed_rewards_pd[col_name] = rewards_list_for_ed[i]
    plot_experiments(ed_rewards_pd, "Figure 5: Rewards per episode for different epsilon decay", "Figure 5: Rewards per episode for different epsilon decay values", "Episodes", "Reward", (-600, 300))

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
    model.save("trained_model_mem_100000_batch_100_doubleNN_10000.h5")

    # Save Rewards list
    pickle.dump(model.rewards_list, open("list_mem_100000_batch_100_doubleNN_10000.p", "wb"))
    rewards_list = pickle.load(open("list_mem_100000_batch_100_doubleNN_10000.p", "rb"))

    # plot reward in graph
    reward_df = pd.DataFrame(rewards_list)
    plot_figure1(reward_df, "Figure_mem_100000_batch_100_doubleNN_10000", "Reward for each training episode", "Episode","Reward")

# #     Run experiments for hyper-parameter
#     run_experiment_for_lr()
#     run_experiment_for_ed()
#     run_experiment_for_gamma()


