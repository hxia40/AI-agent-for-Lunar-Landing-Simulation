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
from scipy.optimize import fsolve
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def cal_TD(lambd,
           probToState = 0.81,
           valueEstimates = [0.0, 4.0, 25.7, 0.0, 20.1, 12.2, 0.0],
           rewards = [7.9, -5.1, 2.5, -7.2, 9.0, 0.0, 1.6],
           gamma = 1
           ):
    valueEstimates = valueEstimates
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

    stepped_esti_list = []
    stepped_esti_list.append(valueEstimates[0])
    stepped_esti_list.append(valueEstimates[1] * probToState + valueEstimates[2] * (1 - probToState))
    stepped_esti_list.append(valueEstimates[3])
    stepped_esti_list.append(valueEstimates[4])
    stepped_esti_list.append(valueEstimates[5])
    stepped_esti_list.append(valueEstimates[6])

    print('stepped_esti_list = ', stepped_esti_list)

    # now we can implement equ. 12.3 in Sultan RL book. here, t = 0 (as we start t at value of 0)
    G_0_1 =  stepped_reward_list[0] + \
                            (gamma ** 1) * stepped_esti_list[1] # as n = 1,  "G sub t:t+n" become "G sub 0:1",
    # which is noted as G_0_1, using equ 12.1,  G_0_1 = lambda^0 * R1 (first item in the reward list))
    G_0_2 = (stepped_reward_list[0] +
                            (gamma ** 1) * stepped_reward_list[1]) + \
                            (gamma ** 2) * stepped_esti_list[2]

    G_0_3 = (stepped_reward_list[0] +
                            (gamma ** 1) * stepped_reward_list[1] +
                            (gamma ** 2) * stepped_reward_list[2])+ \
                            (gamma ** 3) * stepped_esti_list[3]
    G_0_4 =  (stepped_reward_list[0] +
                            (gamma ** 1) * stepped_reward_list[1] +
                            (gamma ** 2) * stepped_reward_list[2] +
                            (gamma ** 3) * stepped_reward_list[3])+ \
                            (gamma ** 4) * stepped_esti_list[4]
    G_0_5 =  (stepped_reward_list[0] +
                            (gamma ** 1) * stepped_reward_list[1] +
                            (gamma ** 2) * stepped_reward_list[2] +
                            (gamma ** 3) * stepped_reward_list[3] +
                            (gamma ** 4) * stepped_reward_list[4])+ \
                            (gamma ** 5) * stepped_esti_list[5]

    # putting "G sub 0:1" to "G sub 0:5", and the conventional return Gt back to equ 12.3:
    G_t_lambda = (1 - lambd) * (lambd ** 0 * G_0_1 +
                                lambd ** 1 * G_0_2 +
                                lambd ** 2 * G_0_3 +
                                lambd ** 3 * G_0_4 +
                                lambd ** 4 * G_0_5) + lambd ** 5 * conventional_return

    print("when lambda =", lambd, ", G_t_lambda = ",  G_t_lambda)
    # return G_t_lambda
    return G_t_lambda - conventional_return   # actually this function should only return G_t_lambda. returnning
    # G_t_lambda - conventional_return will make sure when lambda == 1, that is, TD(1), equal to 0, facilitating
    # the solution of HW 2.

if __name__ == '__main__':
    # cal_TD(lambd=1,
    #               probToState=0.81,
    #               valueEstimates=[0.0, 4.0, 25.7, 0.0, 20.1, 12.2, 0.0],
    #               rewards=[7.9, -5.1, 2.5, -7.2, 9.0, 0.0, 1.6],
    #               gamma=1)

    # '''example set 1'''
    # # Use scipy.optimize.fslove to calculate the numerical solution on what value can make cal_TD = 0.
    # print("============start finding lambda to make TD(lambda) = TD(1)===============")
    # result = fsolve(cal_TD,
    #                 x0=0.5,
    #                 args=(0.81,
    #                       [0.0, 4.0, 25.7, 0.0, 20.1, 12.2, 0.0],
    #                       [7.9, -5.1, 2.5, -7.2, 9.0, 0.0, 1.6],
    #                       1)  # the four args are: probToState, valueEstimates, rewards, and gamma, respectively.
    #                 )
    # print('fsolve result =', result)

    # '''example set 2'''
    # # Use scipy.optimize.fslove to calculate the numerical solution on what value can make cal_TD = 0.
    # print("============start finding lambda to make TD(lambda) = TD(1)===============")
    # result = fsolve(cal_TD,
    #                 x0=0.5,
    #                 args=(0.22,
    #                     [0.0, -5.2, 0.0, 25.4, 10.6, 9.2, 12.3],
    #                     [-2.4, 0.8, 4.0, 2.5, 8.6, -6.4, 6.1],
    #                     1)    # the four args are: probToState, valueEstimates, rewards, and gamma, respectively.
    #                 )
    # print('fsolve result =',result)
    #
    # '''example set 3'''
    # # Use scipy.optimize.fslove to calculate the numerical solution on what value can make cal_TD = 0.
    # print("============start finding lambda to make TD(lambda) = TD(1)===============")
    # result = fsolve(cal_TD,
    #                 x0=0.5,
    #                 args=(0.64,
    #                       [0.0, 4.9, 7.8, -2.3, 25.5, -10.2, -6.5],
    #                       [-2.4, 9.6, -7.8, 0.1, 3.4, -2.1, 7.9],
    #                       1)  # the four args are: probToState, valueEstimates, rewards, and gamma, respectively.
    #                 )
    # print('fsolve result =', result)

    # '''HW set 1'''
    # # Use scipy.optimize.fslove to calculate the numerical solution on what value can make cal_TD = 0.
    # print("============start finding lambda to make TD(lambda) = TD(1)===============")
    # result = fsolve(cal_TD,
    #                 x0=0.5,
    #                 args=(0.52,
    #                       [0.0,0,4,15.3,11,24.7,0.0],
    #                       [1.3,-1,-2.7,7.2,-3.3,10,1.4],
    #                       1)  # the four args are: probToState, valueEstimates, rewards, and gamma, respectively.
    #                 )
    # print('fsolve result =', result)    # 0.56805852

    # '''HW set 2'''
    # # Use scipy.optimize.fslove to calculate the numerical solution on what value can make cal_TD = 0.
    # print("============start finding lambda to make TD(lambda) = TD(1)===============")
    # result = fsolve(cal_TD,
    #                 x0=0.5,
    #                 args=(0.23,
    #                       [0.0,1.1,5.3,3.8,0,-2.1,9.6],
    #                       [1.2,4.7,7.8,4.2,1.5,-0.1,0.0],
    #                       1)  # the four args are: probToState, valueEstimates, rewards, and gamma, respectively.
    #                 )
    # print('fsolve result =', result)  # 0.57894715

    # '''HW set 3'''
    # # Use scipy.optimize.fslove to calculate the numerical solution on what value can make cal_TD = 0.
    # print("============start finding lambda to make TD(lambda) = TD(1)===============")
    # result = fsolve(cal_TD,
    #                 x0=0.5,
    #                 args=(0.53,
    #                       [0.0,-1.3,0,8.3,4.6,17.8,5.3],
    #                       [8.1,0,3.3,9.1,4.5,-0.1,-1.6],
    #                       1)  # the four args are: probToState, valueEstimates, rewards, and gamma, respectively.
    #                 )
    # print('fsolve result =', result)  # 0.57505781
    #
    # '''HW set 4'''
    # # Use scipy.optimize.fslove to calculate the numerical solution on what value can make cal_TD = 0.
    # print("============start finding lambda to make TD(lambda) = TD(1)===============")
    # result = fsolve(cal_TD,
    #                 x0=0.5,
    #                 args=(1.0,
    #                       [0.0,0,0,18.2,12.1,14,20.0],
    #                       [7.6,-3.8,-0.1,5.8,0,-4.9,8.9],
    #                       1)  # the four args are: probToState, valueEstimates, rewards, and gamma, respectively.
    #                 )
    # print('fsolve result =', result)   # 0.23442123

    # '''HW set 5'''
    # # Use scipy.optimize.fslove to calculate the numerical solution on what value can make cal_TD = 0.
    # print("============start finding lambda to make TD(lambda) = TD(1)===============")
    # result = fsolve(cal_TD,
    #                 x0=0.5,
    #                 args=(0.63,
    #                       [0.0,0,-1.5,0,12.4,7.9,13.3],
    #                       [-3.0,-3.5,4.8,8.7,0,-1.7,-1.3],
    #                       1)  # the four args are: probToState, valueEstimates, rewards, and gamma, respectively.
    #                 )
    # print('fsolve result =', result)  # 0.36343052

    # '''HW set 6'''
    # # Use scipy.optimize.fslove to calculate the numerical solution on what value can make cal_TD = 0.
    # print("============start finding lambda to make TD(lambda) = TD(1)===============")
    # result = fsolve(cal_TD,
    #                 x0=0.5,
    #                 args=(0.81,
    #                       [0.0,-2.7,0,0,1.9,21.3,15.8],
    #                       [2.7,0,6.7,-2.8,0.1,0.3,-4.3],
    #                       1)  # the four args are: probToState, valueEstimates, rewards, and gamma, respectively.
    #                 )
    # print('fsolve result =', result)  # 0.3377388

    # '''HW set 7'''
    # # Use scipy.optimize.fslove to calculate the numerical solution on what value can make cal_TD = 0.
    # print("============start finding lambda to make TD(lambda) = TD(1)===============")
    # result = fsolve(cal_TD,
    #                 x0=0.5,
    #                 args=(0.0,
    #                       [0.0,1.5,0,23.7,19.8,21.2,0.7],
    #                       [8.6,0,4.1,-3,1.5,-2,4.1],
    #                       1)  # the four args are: probToState, valueEstimates, rewards, and gamma, respectively.
    #                 )
    # print('fsolve result =', result)  # 0.02908487

    # '''HW set 8'''
    # # Use scipy.optimize.fslove to calculate the numerical solution on what value can make cal_TD = 0.
    # print("============start finding lambda to make TD(lambda) = TD(1)===============")
    # result = fsolve(cal_TD,
    #                 x0=0.5,
    #                 args=(0.92,
    #                       [0.0,13.1,1.3,24.5,19.9,21.5,0.0],
    #                       [5.3,0.6,6.7,-2.2,8.6,4.1,-3.2],
    #                       1)  # the four args are: probToState, valueEstimates, rewards, and gamma, respectively.
    #                 )
    # print('fsolve result =', result)  # 0.17469671

    # '''HW set 9'''
    # # Use scipy.optimize.fslove to calculate the numerical solution on what value can make cal_TD = 0.
    # print("============start finding lambda to make TD(lambda) = TD(1)===============")
    # result = fsolve(cal_TD,
    #                 x0=0.5,
    #                 args=(0.37,
    #                       [0.0,21.6,17.2,3.1,11.2,9.1,0.0],
    #                       [0.0,4.2,8.7,-3.8,-4.8,9.8,8.4],
    #                       1)  # the four args are: probToState, valueEstimates, rewards, and gamma, respectively.
    #                 )
    # print('fsolve result =', result)  # 0.36135704

    # '''HW set 10'''
    # # Use scipy.optimize.fslove to calculate the numerical solution on what value can make cal_TD = 0.
    # print("============start finding lambda to make TD(lambda) = TD(1)===============")
    # result = fsolve(cal_TD,
    #                 x0=0.5,
    #                 args=(0.37,
    #                       [0.0,0.4,-3.6,19.2,19.2,-4.8,0.0],
    #                       [6.1,9.6,2.6,0,2.1,0.2,-1.1],
    #                       1)  # the four args are: probToState, valueEstimates, rewards, and gamma, respectively.
    #                 )
    # print('fsolve result =', result)  # 0.1963809

    '''for exmaple 1, drawing the TD(lambda) curve when lambda has the value between 0 and 1'''
    temp_list = []
    alter_list = []
    for i in np.linspace(0, 1, num=50):
        alter_list.append(i)
        temp_list.append(cal_TD(probToState=0.81,
                                valueEstimates=[0.0, 4.0, 25.7, 0.0, 20.1, 12.2, 0.0],
                                rewards=[7.9, -5.1, 2.5, -7.2, 9.0, 0.0, 1.6],
                                lambd=i,
                                gamma=1))

    plt.grid()
    # ylim = (0, 1.1)
    # plt.ylim(*ylim)

    plt.plot(alter_list, temp_list, color="r")

    plt.savefig('example1.png')
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












