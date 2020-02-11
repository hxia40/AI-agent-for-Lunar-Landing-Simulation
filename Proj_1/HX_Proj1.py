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


def make_train_sets(num_train_set=100,num_sequences=10, random_seed=1):
    # generate sequences like "454567", instead of "DEDEFG", as the former is way easier to code.
    np.random.seed(random_seed)
    all_sets = []
    for each_train_set in range(num_train_set):
        train_set = []
        for each_seq in range(num_sequences):
            sequence = [4]
            temp = 4
            while (temp < 7) and (temp > 1):
                temp += np.random.choice([-1, 1])
                sequence.append(temp)
            train_set.append(sequence)
        all_sets.append(train_set)

    return all_sets


def FindMaxLength(listt):
    max_length = 0
    for i in listt:
        for j in i:
            max_length = max(max_length, len(j))
    print("max_length of all sequences is:", max_length)


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def cal_TD(lambd,
           alpha,
           sequence = [4, 5, 6, 7],    # t = 1, 2, 3, 4, respectively for states (4, 5, 6, 7), or ('D', 'E', 'F', 'G')
           valueEstimates = [0.5, 0.5, 0.5, 0.5, 0.5],  # omega here
           # valueEstimates = [1, 1, 1, 1, 1],  # omega here
           gamma = 1,
           verbose = 0
           ):  # returns "delta omegaT" for the whole sequence. i,e, for here, "omega t=1" + "omega t=2" + "omega t=3"
    omega = np.array(valueEstimates)    # omega is initiated with [0.5, 0.5, 0.5, 0.5, 0.5], representing states B, C, D, E, and F
    # convert sequence into xt (i.e. state) matrix
    for step in range(len(sequence)):
        print("sequence[step]", sequence[step])
        if sequence[step] == 2:
            sequence[step] = (np.array([1,0,0,0,0])).T
        elif sequence[step] == 3:
            sequence[step] = (np.array([0, 1, 0, 0, 0])).T
        elif sequence[step] == 4:
            sequence[step] = (np.array([0, 0, 1, 0, 0])).T
        elif sequence[step] == 5:
            sequence[step] = (np.array([0, 0, 0, 1, 0])).T
        elif sequence[step] == 6:
            sequence[step] = (np.array([0, 0, 0, 0, 1])).T
        elif sequence[step] == 7:
            sequence[step] = 1
        elif sequence[step] == 1:
            sequence[step] = 0
    if verbose == 1:
        print(sequence, "\n===============")

    Pt = 0
    eligibility_list = []
    delta_omega_t_list = []
    for step in range(len(sequence)-1):
        t = step + 1

        # calculate eligibility into combined_eligibility_list:
        eligibility_list = [i * lambd for i in eligibility_list]
        eligibility_list.append(sequence[step])
        combined_eligibility_list = (np.array(eligibility_list)).sum(axis=0)

        # calculate Pt and "Pt+1"
        Pt = omega.T * sequence[step]
        if step == len(sequence)-2:
            if verbose == 1:
                print("last number!")
            P_t_plus_1 = sequence[step+1]
        else:
            P_t_plus_1 = omega.T * sequence[step+1]

        # calculate "delta omega t":
        delta_omega_t = alpha * (P_t_plus_1 - Pt) * combined_eligibility_list
        delta_omega_t_list.append(delta_omega_t)
        if verbose == 1:
            print("for current step, t=", t,
                  ", the eligibility_list = ", combined_eligibility_list,
                  "Pt=", Pt,
                  "P_t_plus_1 =", P_t_plus_1,
                  "delta_omega_t=", delta_omega_t)
    combined_delta_omega_t_list = (np.array(delta_omega_t_list)).sum(axis=0)
    if verbose == 1:
        print("=================\ndelta_omega_t_list", delta_omega_t_list)
        print("=================\ncombined_delta_omega_t_list",combined_delta_omega_t_list)
    return combined_delta_omega_t_list

    # stepped_eligibility_list = (np.array(stepped_eligibility_list)).sum(axis = 0)
    #
    # print('stepped_eligibility_list = \n', stepped_eligibility_list)

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
    all_sets = make_train_sets(num_train_set=1,num_sequences=10, random_seed=1)
    # print(all_sets)
    # FindMaxLength(all_sets)

    targets = [0, 1/6, 1/3, 1/2, 2/3, 5/6, 1]
    train_set_enum = 0
    for each_train_set in all_sets:
        train_set_enum += 1
        print("this is the ", train_set_enum, "th train set")
        valueEstimates = [0.5, 0.5, 0.5, 0.5, 0.5]  # delta_omega_T should start 0.5 for all B,C,D,E,and F,until updated
        omega_t_list = []
        seq_enum = 0
        while True:
            for each_seq in each_train_set:
                seq_enum += 1
                print("this is the ", seq_enum, "th seq")
                delta_omega_T = cal_TD(lambd=0.1,
                                       alpha=0.01,
                                       sequence=each_seq,
                                       valueEstimates=valueEstimates,  # which is delta_omega_T
                                       gamma=1,
                                       verbose=0,
                                       )
                omega_t_list.append(delta_omega_T)
            print("omega_t_list:",omega_t_list)
            combined_omega_t_list = (np.array(omega_t_list)).sum(axis=0)
            print("combined_omega_t_list:", combined_omega_t_list)
            valueEstimates += combined_omega_t_list
            print("valueEstimates:", valueEstimates)


        # print("converged valueEstimates is:", valueEstimates)
        # RMS_error = rmse(predictions, targets)



