"""
Project #1
"""
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import time
from scipy.optimize import fsolve
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



'''
For generating Figure 3:

Generate 100 training set, Each training set has 10 sequences.
 
For each training set:
  Repeat cyclically (1,2, ..., 9,10, 1,2 ...) its ten sequences, update weight every 10 sequences, until the weight is converged.
  Calculated RMS error based on this converged weights vector
 
Final RMS (point on the graph) is the average of the 100 above rms measures.

'''

def make_seqs(num_train_set=100, num_sequences=10):
    ls_seq = [
        [4, 5, 6, 7],
        [4, 5, 6, 7],
        [4, 5, 6, 7],
        [4, 5, 6, 7],
        [4, 5, 6, 7],
        [4, 5, 6, 7],
        [4, 5, 6, 7],
        [4, 5, 6, 7],
        [4, 5, 6, 7],
        [4, 5, 6, 7],
    ]

    return ls_seq

def cal_TD(lambd,
           sequence = ['D', 'C', 'D', 'E', 'F', 'G'],
           valueEstimates = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # Pt here
           rewards = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
           gamma = 1,
           ):
    valueEstimates = valueEstimates
    # play with lambda
    lambd = lambd
    initial_t = 1
    # for a MDP with a the target (total time) of T, t = 0 , 1 ... T-1 = 4, as a total of 5 turns


    stepped_eligibility_list = []
    for step in sequence:
        print("current step is at:", step)
        # stepped_esti_list *= lambd
        stepped_eligibility_list = [i * lambd for i in stepped_eligibility_list]
        if step == 'B':
            temp_omega = (np.array([1,0,0,0,0])).T
            stepped_eligibility_list.append(temp_omega)
        elif step == 'C':
            temp_omega = (np.array([0,1,0,0,0])).T
            stepped_eligibility_list.append(temp_omega)
        elif step == 'D':
            temp_omega = (np.array([0,0,1,0,0])).T
            stepped_eligibility_list.append(temp_omega)
        elif step == 'E':
            temp_omega = (np.array([0,0,0,1,0])).T
            stepped_eligibility_list.append(temp_omega)
        elif step == 'F':
            temp_omega = (np.array([0,0,0,0,1])).T
            stepped_eligibility_list.append(temp_omega)
        else:
            pass

    stepped_eligibility_list = (np.array(stepped_eligibility_list)).sum(axis = 0)


    print('stepped_eligibility_list = \n', stepped_eligibility_list)
    '''
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
    '''



if __name__ == '__main__':
    # seq = make_sample(num_train_set=100, num_sequences=10)
    # print(seq)
    cal_TD(lambd = 1)

