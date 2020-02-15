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


def make_train_sets(num_train_set=100,num_sequences=10, random_seed=1, length_limit = 9999):
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
                if len(sequence) > length_limit:
                    sequence = [4]
                    temp = 4

            train_set.append(sequence)
        all_sets.append(train_set)

    return all_sets


def make_exact_same_train_sets(num_train_set=100,num_sequences=10, random_seed=1, length_limit = 9999):
    # generate sequences like "454567", instead of "DEDEFG", as the former is way easier to code.
    np.random.seed(random_seed)
    all_sets = []
    for each_train_set in range(num_train_set):
        train_set = []
        for each_seq in range(num_sequences):
            sequence = [4,3,2,1]

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
    omega = np.array(valueEstimates)  # omega is initiated with [0.5, 0.5, 0.5, 0.5, 0.5], representing states B, C, D, E, and F
    # convert sequence into xt (i.e. state) matrix
    for step in range(len(sequence)):
        # print("sequence[step]", sequence[step])
        if sequence[step] == 2:
            sequence[step] = (np.array([1, 0, 0, 0, 0])).T
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
        print("this new sequence is:\n", sequence,"\n")

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
        # print("calculate Pt and Pt+1: omega =", omega, "sequence[step]=", sequence[step])
        Pt = np.dot(omega.T , sequence[step])
        if step == len(sequence)-2:
            if verbose == 1:
                print("last number!")
            P_t_plus_1 = sequence[step+1]
        else:
            P_t_plus_1 = np.dot(omega.T , sequence[step+1])

        # calculate "delta omega t":
        delta_omega_t = alpha * (P_t_plus_1 - Pt) * combined_eligibility_list
        delta_omega_t_list.append(delta_omega_t)
        if verbose == 1:
            print("for current step, t=", t,
                  # ", the eligibility_list = ", combined_eligibility_list, "||",
                  "Pt=", Pt, "||",
                  "P_t_plus_1 =", P_t_plus_1, "||",
                  "delta_omega_t=", delta_omega_t)
    combined_delta_omega_t_list = (np.array(delta_omega_t_list)).sum(axis=0)
    if verbose == 1:
        print("\ncombined_delta_omega_t_list in function",combined_delta_omega_t_list, "\n-----------end---------")
    return combined_delta_omega_t_list

    # stepped_eligibility_list = (np.array(stepped_eligibility_list)).sum(axis = 0)
    #
    # print('stepped_eligibility_list = \n', stepped_eligibility_list)


# trying to figure out on why cant loop again and again, not useful anymore
def GenerateCombinedWreights(train_set, lambd, alpha):
    omega_t_list = []
    seq_enum = 0
    valueEstimates = [0.5, 0.5, 0.5, 0.5, 0.5]  # delta_omega_T should start 0.5 for all B,C,D,E,and F,until updated
    for each_seq in train_set:
        seq_enum += 1
        print("this is the ", seq_enum, "th seq")
        delta_omega_T = cal_TD(lambd=lambd,
                               alpha=alpha,
                               sequence=each_seq,
                               valueEstimates=valueEstimates,  # which is delta_omega_T
                               gamma=1,
                               verbose=0,
                               )
        omega_t_list.append(delta_omega_T)
    print("omega_t_list:", omega_t_list)
    combined_omega_t_list = (np.array(omega_t_list)).sum(axis=0)
    print("combined_omega_t_list:", combined_omega_t_list)
    valueEstimates += combined_omega_t_list
    print("valueEstimates:", valueEstimates)


def Figure3Iterator(train_set, lambd, alpha):   # Generate one point of data for figure 3
    omega_t_list = []
    seq_enum = 0
    valueEstimates = [0.5, 0.5, 0.5, 0.5, 0.5]  # delta_omega_T should start 0.5 for all B,C,D,E,and F,until updated
    for each_seq in train_set:
        seq_enum += 1
        print("this is the ", seq_enum, "th seq")
        delta_omega_T = cal_TD(lambd=lambd,
                               alpha=alpha,
                               sequence=each_seq,
                               valueEstimates=valueEstimates,  # which is delta_omega_T
                               gamma=1,
                               verbose=0,
                               )
        omega_t_list.append(delta_omega_T)
    print("omega_t_list:", omega_t_list)
    combined_omega_t_list = (np.array(omega_t_list)).sum(axis=0)
    print("combined_omega_t_list:", combined_omega_t_list)
    valueEstimates += combined_omega_t_list
    print("valueEstimates:", valueEstimates)

if __name__ == '__main__':
    all_sets = make_train_sets(num_train_set=100,num_sequences=10, random_seed=1, length_limit = 40)  # somehow then num_sequence larger than 100, say 1000, the value estimate will go crazy. using a smaller alhpa helps.
    # same_sets = make_exact_same_train_sets(num_train_set=100,num_sequences=10, random_seed=1, length_limit = 9999)

    # print(all_sets)
    # all_sets = [[[4, 5, 6, 7],[4, 5, 6, 7],[4, 5, 6, 7],[4, 5, 6, 7],[4, 5, 6, 7]]]
    # FindMaxLength(all_sets)
    lam_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alph_list = [0.0, 0.05,  0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    alter_lam_list = []
    for lam_value in lam_list:
        alter_alpha_list = []
        for alph_value in alph_list:
            print("lambda = ", lam_value, "alpha = ", alph_value)
            '''new posisiton of valueestimates for resetting onlhy att he beginning - seems not working, 
            also refer to piazza @291_f2, i should just reset weight after each trainning set'''
            # valueEstimates = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # delta_omega_T should start 0.5 for all B,C,D,E,and F,until updated
            train_set_enum = 0
            for each_train_set in all_sets:

                num_update = 0

                trainset_lvl_valueEstimates_list = []   # reset the rainset_lvl_valueEstimates_list for each train set
                omega_t_list = []  # need to reset omega list to empty each time all of the trainset is done.
                current_train_set = each_train_set.copy()
                train_set_enum += 1
                # print("this is the ", train_set_enum, "th train set, full set is:\n", each_train_set)
                '''original, and adopted posisiton of valueestimates for resetting every dataset'''
                valueEstimates = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # delta_omega_T should start 0.5 for all B,C,D,E,and F,until updated

                seq_enum = 0
                circle_enum = 0

                '''updateing valueEstimates once every 1 sequence, 10 updates per trainning set'''
                for each_seq in current_train_set:
                    # print(each_seq)
                    temp = each_seq.copy()
                    # seq_enum += 1
                    # print("this is the ", seq_enum, "th seq")
                    delta_omega_T = cal_TD(lambd=lam_value,
                                           alpha=alph_value,
                                           sequence=temp,
                                           valueEstimates=valueEstimates,  # which is delta_omega_T
                                           gamma=1,
                                           verbose=0,
                                           )
                    # print("combined_delta_omega_t_list out of function:", delta_omega_T)

                    old_valueEstimates = valueEstimates.copy()
                    # print("valueEstimates", valueEstimates)
                    valueEstimates += delta_omega_T
                    num_update += 1
                    # print("num_update", num_update)
                    # print("valueEstimates after adding:", valueEstimates)

                trainset_lvl_valueEstimates_list.append(valueEstimates)

            combined_trainset_lvl_valueEstimates_list = (np.array(trainset_lvl_valueEstimates_list)).mean(axis=0)
            print(" combined_trainset_lvl_valueEstimates_list:", combined_trainset_lvl_valueEstimates_list)

            targets = [1/6, 1/3, 1/2, 2/3, 5/6]
            point_error = rmse(targets, combined_trainset_lvl_valueEstimates_list)
            print("for a total of ", len(all_sets), "sets, the error is:", point_error)
            alter_alpha_list.append(point_error)
            # print("alter_alpha_list:", alter_alpha_list)
        alter_lam_list.append(alter_alpha_list)
    print("alter_lam_list:", alter_lam_list)

    '''plotting the whole alter_lam_list as Figure 4'''

    plt.grid()
    ylim = (0, 0.7)
    plt.ylim(*ylim)

    fully_nested = [list(zip(*[(ix + 1, y) for ix, y in enumerate(x)])) for x in alter_lam_list]
    names = ['lambda=%s' % (i) for i in lam_list]

    for l in fully_nested:
        plt.plot(*l)
    plt.ylabel("Error")
    plt.xlabel("alpha")
    plt.legend(names, fontsize=7, loc='upper left')
    plt.savefig('F4_Maxlength_40_10updatepertrainningset_rdomseed1_seq_of_10.png')
    plt.gcf().clear()

    '''plotting the error value at best alpha for each lambda as Figure 5'''

    plt.grid()

    min_error_list = []
    for each_alter_alpha_list in alter_lam_list:
        min_error = min(each_alter_alpha_list)
        print("alpha list is:", each_alter_alpha_list)
        print("min is:", min_error, "\n")
        min_error_list.append(min_error)

    plt.plot(lam_list, min_error_list)

    plt.xlabel('λ')
    plt.ylabel("ERROR USING BEST α")
    plt.legend(loc="best")
    plt.legend(names, fontsize=7, loc='upper left')
    plt.savefig('F5_Maxlength_40_10updatepertrainningset_rdomseed1_seq_of_10.png')





