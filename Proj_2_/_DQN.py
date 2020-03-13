import gym
import numpy as np
from collections import deque
import random

from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import Adam
from keras.losses import mean_squared_error


class DQN:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_decay, mem_size, mem_batch_size):

        self.env = env
        self.counter = 0
        # somehow, if i dont replace the state dimension and action numbers with a sefl.* superparameter, 
        # the q-learner converge toward worse performance
        self.state_dimension = env.observation_space.shape[0]
        self.action_num = env.action_space.n

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=mem_size)
        self.batch_size = mem_batch_size
        
        self.model = self.create_model()
        
        self.rewards_list = []

    def create_model(self):
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_dimension, activation=relu))
        model.add(Dense(100, activation=relu))
#         model.add(Dense(100, activation=relu))
        model.add(Dense(self.action_num, activation=linear))
        model.compile(loss=mean_squared_error,optimizer=Adam(learning_rate=self.alpha))
        print(model.summary())
        return model
    
    def remember(self, state, action, reward, state_, done):
        self.memory.append([state, action, reward, state_, done])
        
    def learn_from_replay(self):
        # memory have to be larger than batch_size
        if len(self.memory) < self.batch_size or self.counter != 0: 
            return
        # from Gupta: Early Stopping is the practice to stop the neural networks from overtraining. 
        # Avoid training the model for a specific episode if the average of the last 10 rewards is more than 180.
        if np.mean(self.rewards_list[-10:]) > 180:   
            return
        samples = random.sample(self.memory, self.batch_size)
        states, actions, rewards, state_s, dones = self.get_attribues_from_samples(samples)
        # This calculate the target part of the Q learner (i.e. the "r + gamma * argmax" part). 
        # Using (1- dones), rather than if, to perform vectorized conditional test.
        targets = \
            rewards + self.gamma * (np.amax(self.model.predict_on_batch(state_s), axis=1)) * (1 - dones)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets
        self.model.fit(states, target_vec, epochs=1, verbose=0)
                
    # from Gupta. By attaching all samples into a numpy array, calculations are vectorized and thus save computation power.
    def get_attribues_from_samples(self, samples):
        states = np.array([i[0] for i in samples])
        actions = np.array([i[1] for i in samples])
        rewards = np.array([i[2] for i in samples])
        state_s = np.array([i[3] for i in samples])
        dones = np.array([i[4] for i in samples])
        states = np.squeeze(states)
        state_s = np.squeeze(state_s)
        return np.squeeze(states), actions, rewards, state_s, dones

    def perform_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_num)
        else:
            return np.argmax(self.model.predict(state)[0])

    def update_counter(self):
        self.counter += 1
        step_size = 5
        self.counter = self.counter % step_size

    def save(self, name):
        self.model.save(name)

    def train(self, num_episodes=1000, stop_good_enough=True):
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            num_steps = 1000
            state = np.reshape(state, [1, self.state_dimension])
            for step in range(num_steps):
                # env.render()
                received_action = self.perform_action(state)
                
                state_, reward, done, info = self.env.step(received_action)
                state_ = np.reshape(state_, [1, self.state_dimension])
                
                
                self.remember(state, received_action, reward, state_, done)
                
                episode_reward += reward
                state = state_
                self.update_counter()
                self.learn_from_replay()
                if done:
                    break
                    
            self.rewards_list.append(episode_reward)

            # Epsilon decay 
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

            # Check for breaking condition
            mean_score_last_hundred_episode = np.mean(self.rewards_list[-100:])
            if mean_score_last_hundred_episode > 200 and stop_good_enough:
                print("Training Complete!")
                break
            print("Episode =", episode, "Episode Reward = ", episode_reward, \
                  "Reward Rolling Mean = ",mean_score_last_hundred_episode, "Epsilon = ", self.epsilon )
            



