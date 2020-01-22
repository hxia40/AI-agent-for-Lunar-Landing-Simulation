"""
Solving environment using Value Itertion.
Author : Wei Feng
"""
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import gym
from gym import wrappers
from HX_maze import generate_random_map, FrozenLakeEnv
from HX_DieN import DieNEnv
import time

def run_episode(env, policy, gamma = 1.0, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()  # Resetting the environment will return an integer. This number will be our initial state.
    total_reward = 0
    step_idx = 0

    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        print(obs, reward)

        # total_reward += (gamma ** step_idx * reward)
        # the above code is from Moustafa Alzantot , which this is problematic.
        # As the policy's target here is never to finish in shortest time. Rather,
        # the only thing matters is that if u can successfully recover ur stuff, or drop into one of the ice-hole.
        total_reward += reward    # HX
        step_idx += 1
        if done:
            print('done')
            break
    # print "total_reward:", total_reward
    return total_reward

def evaluate_policy(env, policy, gamma = 1.0,  n = 10):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    start_time = time.time()
    scores = []
    for _ in range(n):
        if _ % (n/100) == 0:
            print("=====sample went through =====", _)
        scores.append(run_episode(env, policy, gamma = gamma, render = False))


    # # scores = [
    # #         run_episode(env, policy, gamma = gamma, render = False)
    # #         for _ in range(n)]
    end_time = time.time() - start_time
    print("time consumed for n=%d is %d") %(n, end_time)
    return np.mean(scores)

def run_episode_stock(env, policy, gamma = 1.0, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()  # Resetting the environment will return an integer. This number will be our initial state.
    obs = 11  # that's what usually happens. for PI/VI for the daily trading, obs has to be set to 0
    total_reward = 0
    step_idx = 0

    while True:
        if render:
            env.render()
        actual_obs_not_using, reward, done , _ = env.step(int(policy[obs]))
        obs += 1
        # print('obs:', obs, 'r:', reward, 'done:', done)
        # time.sleep(0.3)

        # total_reward += (gamma ** step_idx * reward)
        # the above code is from Moustafa Alzantot , which this is problematic.
        # As the policy's target here is never to finish in shortest time. Rather,
        # the only thing matters is that if u can successfully recover ur stuff, or drop into one of the ice-hole.
        total_reward += reward    # HX
        step_idx += 1
        if done:

            break
    # print "total_reward:", total_reward
    return total_reward

def evaluate_policy_stock(env, policy, gamma = 1.0,  n = 1000):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
            run_episode_stock(env, policy, gamma = gamma, render = False)
            for _ in range(n)]
    return np.mean(scores)

class VI:
    def __init__(self, env):
        self.env = env
        self.V = np.zeros(env.nS)

    def next_best_action(self, s, V, gamma=1):
        # print("\ns =" , s)
        action_values = np.zeros(self.env.nA)
        for a in range(self.env.nA):
            for prob, next_state, reward, done in self.env.P[s][a]:
                # print('prob:', prob, 's_:', next_state, 'r:', reward, 'done:', done)
                action_values[a] += prob * (reward + gamma * V[next_state])
        return np.argmax(action_values), np.max(action_values)

    def optimize(self, gamma =1):
<<<<<<< HEAD
        # THETA = 12.3636363637
=======
        THETA = 13.23543
>>>>>>> 820eea75b3e5db389b79c0166f572c317849e5a3
        delta = float("inf")
        last_delta = float("inf")
        round_num = 0

        while delta:
            start_time = time.time()
            delta = 0
            # print("\nValue Iteration: Round " + str(round_num))
            # print(np.reshape(self.V,(8,8)))
            for s in range(self.env.nS):
                best_action, best_action_value = self.next_best_action(s, self.V, gamma)
                delta = max(delta, np.abs(best_action_value - self.V[s]))
                self.V[s] = best_action_value
            if last_delta/delta < 1.000000000000000000001:
                break
            else:
                round_num += 1
                last_delta = delta
                print('round_num/delta/time:', round_num, delta, time.time()-start_time)

        policy = np.zeros(self.env.nS)
        for s in range(self.env.nS):
            best_action, best_action_value = self.next_best_action(s, self.V, gamma)
            policy[s] = best_action
        print('VI policy:', policy)
        # print('VI table:', self.V)
        return policy


if __name__ == '__main__':
    '''===========DieN==========='''
    '''obs, reward, done , _ = env.step(int(policy[obs]))'''
    # isBadSide = [1, 1, 1, 0, 0, 0]
<<<<<<< HEAD
    # isBadSide = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    isBadSide = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
    n = len(isBadSide)
    slip=0
    env_DN = DieNEnv(n=n, isBadSide=isBadSide, slip=slip)
    env_DN.slip = 0

    # print(env_DN.nA)
    # print(env_DN.nS)
    # print(env_DN.isBadSide)
=======
    isBadSide = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
    numbb = len(isBadSide)
    env_DN = DieNEnv(numbb=numbb,
                     isBadSide=isBadSide,
                     slip=0)


    print(env_DN.numbb)
    print(env_DN.nS)
    print(env_DN.isBadSide)
>>>>>>> 820eea75b3e5db389b79c0166f572c317849e5a3
    # print("==========================")
    # print(env_DN.P)
    # print("==========================")
    vi_DN = VI(env_DN)
    optimal_policy_DN = vi_DN.optimize(gamma=1)

<<<<<<< HEAD
    policy_score = evaluate_policy(env_DN, optimal_policy_DN,
                                   n = 1000
                                   # n=600000000
                                   )
=======
    policy_score = evaluate_policy(env_DN, optimal_policy_DN, n=1000)
>>>>>>> 820eea75b3e5db389b79c0166f572c317849e5a3
    print('Policy average score = ', policy_score)
    # '''===========Frozenlake==========='''
    # env_name  = 'FrozenLake8x8-v0'
    # env = gym.make(env_name)
    # vi = VI(env)
    # optimal_policy = vi.optimize(gamma=1)
    # policy_score = evaluate_policy(env, optimal_policy, n=1)
    # print('Policy average score = ', policy_score)
    # print('end')
    # '''===========stocks==========='''
    # '''obs, reward, done , _ = env.step(int(policy[obs]))'''
    #
    # env_AT = StocksEnv(df=pd.read_csv('IBM.csv'),frame_bound=(50, 100), window_size=10)
    # print(env_AT.nA)
    # print(env_AT.nS)
    # print("==========================")
    # print(env_AT.P)
    # print("==========================")
    # vi_AT = VI(env_AT)
    # optimal_policy_AT = vi_AT.optimize(gamma=1)
    # policy_score = evaluate_policy_stock(env_AT, optimal_policy_AT, n=1000)
    # print('Policy average score = ', policy_score)

