"""
Homework #2
TD( λ )
Problem
Description
Recall that the TD( λ ) estimator for an MDP can be thought of as a weighted combination of the
k-step estimators Ek for k ≥ 1.

        ∞
TD(λ) = Σ(1-λ)λ^(k-1)Ek
       k=1



Consider the MDP described by the following state diagram. (Assume the discount factor is γ = 1.)

Procedure
● Find a value of λ , strictly less than 1, such that the TD estimate for λ equals that of the
TD(1) estimate. Round your answer for λ to three decimal places.
● This HW is designed to help solidify your understanding of the Temporal Difference
algorithms and k-step estimators. You will be given the probability to State 1 and a vector
of rewards [r0, r1, r2, r3, r4, r5, r6]
● You will be given 10 test cases for which you will return the best lambda value for each.
Your answer must be correct to 3 decimal places. You may use any programming
language and libraries you wish.

Examples
The following examples can be used to verify your calculation is correct.
● Input: probToState = 0.81, valueEstimates = [0.0, 4.0, 25.7, 0.0, 20.1, 12.2, 0.0],
rewards = [7.9, -5.1, 2.5, -7.2, 9.0, 0.0, 1.6], Output: 0.6227
● Input: probToState = 0.22, valueEstimates = [0.0, -5.2, 0.0, 25.4, 10.6, 9.2, 12.3],
rewards = [-2.4, 0.8, 4.0, 2.5, 8.6, -6.4, 6.1], Output: 0.4956
● Input: probToState = 0.64, valueEstimates = [0.0, 4.9, 7.8, -2.3, 25.5, -10.2, -6.5],
rewards = [-2.4, 9.6, -7.8, 0.1, 3.4, -2.1, 7.9], Output: 0.2055
"""
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import time
from sympy import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def cal_TD(probToState = 0.6227,
            valueEstimates = [0.0, 4.0, 25.7, 0.0, 20.1, 12.2, 0.0],
            rewards = [7.9, -5.1, 2.5, -7.2, 9.0, 0.0, 1.6],
            lambd = 0.6227,
            gamma = 1
            ):
    # play with lambda
    lambd = lambd
    initial_t = 0
    # for a MDP with a the target (total time) of T, t = 0 , 1 ... T-1 = 4, as a total of 5 turns
    T = 5
    lambda_list = []
    for t in range(initial_t, T):
        # print(t)
        lambda_list.append((1 - lambd) * (lambd ** t))
    final_item = lambd ** (T - initial_t)
    print('lambda =', lambd)
    print("lambda_list = ", lambda_list)
    print("sum lambda_list, final_item =", sum(lambda_list), ",", final_item)
    print("total weight =", sum(lambda_list) + final_item)

    probToState = probToState
    rewards = rewards
    # calculate conventional_return, the total return of the whole MDP, which is referred as Gt in equ 12.3,
    # considering gamma = 1
    conventional_return = (rewards[0] + rewards[2]) * probToState + \
                          (rewards[1] + rewards[3]) * (1 - probToState) + \
                          rewards[4] + rewards[5] + rewards[6]

    print('conventional_return = ', conventional_return)  # which is also referred as Gt in equ 12.3

    stepped_reward_list = []
    stepped_reward_list.append(rewards[0] * probToState + rewards[1] * (1 - probToState))
    stepped_reward_list.append(rewards[2] * probToState + rewards[3] * (1 - probToState))
    stepped_reward_list.append(rewards[4])
    stepped_reward_list.append(rewards[5])
    stepped_reward_list.append(rewards[6])

    print('stepped_reward_list = ', stepped_reward_list)

    # now we can implement equ. 12.3 in Sultan RL book. here, t = 0 (as we start t at value of 0)
    # this is the shortened version, assuming gamma = 1
    G_t_lambda = (lambd) ** 0 * stepped_reward_list[0] + \
                 (lambd) ** 1 * stepped_reward_list[1] + \
                 (lambd) ** 2 * stepped_reward_list[2] + \
                 (lambd) ** 3 * stepped_reward_list[3] + \
                 (lambd) ** 4 * stepped_reward_list[4]

    print("when lambda =", lambd, ", G_t_lambda = ",  G_t_lambda)

    return G_t_lambda

if __name__ == '__main__':
    # cal_TD(probToState=0.81,
    #        valueEstimates=[0.0, 4.0, 25.7, 0.0, 20.1, 12.2, 0.0],
    #        rewards=[7.9, -5.1, 2.5, -7.2, 9.0, 0.0, 1.6],
    #        lambd=0.6227,
    #        gamma=1)

    temp_list = []
    alter_list = []
    for i in np.linspace(-1, 1, num=50):
        alter_list.append(i)
        temp_list.append(cal_TD(probToState=0.64,
                                valueEstimates=[0.0, 4.9, 7.8, -2.3, 25.5, -10.2, -6.5],
                                rewards=[-2.4, 9.6, -7.8, 0.1, 3.4, -2.1, 7.9],
                                lambd=i,
                                gamma=1))

    plt.grid()
    # ylim = (0, 1.1)
    # plt.ylim(*ylim)

    plt.plot(alter_list, temp_list, color="r")

    plt.savefig('short.png')
    plt.gcf().clear()


    #
    # x, y = symbols('x y')
    # gfg_exp = x ** 2 - 4
    #
    # print("Before Integration : {}".format(gfg_exp))
    #
    # # Use sympy.integrate() method
    # intr = solve(gfg_exp, x)
    #
    # print("After Integration : {}".format(intr))












