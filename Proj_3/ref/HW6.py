'''
ref:
https://medium.com/@excitedAtom/linear-programming-in-python-cvxopt-and-game-theory-8626a143d428
by Adam Novotny
'''


from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import numpy as np


def maxmin(A, solver="glpk"):        # by Adam Novotny
    num_vars = len(A)
    # minimize matrix c
    c = [-1] + [0 for i in range(num_vars)]
    c = np.array(c, dtype="float")
    c = matrix(c)
    # constraints G*x <= h
    G = np.matrix(A, dtype="float").T # reformat each variable is in a row
    G *= -1 # minimization constraint
    G = np.vstack([G, np.eye(num_vars) * -1]) # > 0 constraint for all vars
    new_col = [1 for i in range(num_vars)] + [0 for i in range(num_vars)]
    G = np.insert(G, 0, new_col, axis=1) # insert utility column
    G = matrix(G)
    h = ([0 for i in range(num_vars)] + 
         [0 for i in range(num_vars)])
    h = np.array(h, dtype="float")
    h = matrix(h)
    # contraints Ax = b
    A = [0] + [1 for i in range(num_vars)]
    A = np.matrix(A, dtype="float")
    A = matrix(A)
    b = np.matrix(1, dtype="float")
    b = matrix(b)
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)
    return sol


def ce(A, solver=None):
    num_vars = len(A)
    print("line 42, num_vars", num_vars)
    # maximize matrix c
    c = [sum(i) for i in A] # sum of payoffs for both players
    c = np.array(c, dtype="float")
    c = matrix(c)
    c *= -1 # cvxopt minimizes so *-1 to maximize
    # constraints G*x <= h
    G = build_ce_constraints(A=A)
    print("line 48, G", G) # G size 4 by 25
    print("line 50, num_vars", num_vars)
    print("line_51, np.eye(num_vars)", np.eye(num_vars))  # np.eye(num_vars) size 25 by 25
    G = np.vstack([G, np.eye(num_vars) * -1]) # > 0 constraint for all vars
    print("line 53, G", len(G), G)             # G size 29 by 25
    h_size = len(G)
    G = matrix(G)
    h = [0 for i in range(h_size)]
    h = np.array(h, dtype="float")
    h = matrix(h)
    # contraints Ax = b
    A = [1 for i in range(num_vars)]
    A = np.matrix(A, dtype="float")
    A = matrix(A)
    b = np.matrix(1, dtype="float")
    b = matrix(b)
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)
    return sol

def build_ce_constraints(A):
    print("A", A)
    num_vars = int(len(A) ** 0.5)
    # num_vars = 3
    print("line 69, num_vars", num_vars)
    G = []
    # row player
    for i in range(num_vars): # action row i
        for j in range(num_vars): # action row j
            if i != j:
                constraints = [0 for m in A]
                print("line 78, constraints, i, num_vars", constraints, i, num_vars)
                base_idx = i * num_vars
                print("line 80, base_idx", base_idx)
                comp_idx = j * num_vars
                for k in range(num_vars):

                    print("line 82, comp_idx", comp_idx)
                    print("line 82, k", k)
                    constraints[base_idx+k] = (- A[base_idx+k][0] + A[comp_idx+k][0])

                print("ce_Q.py line 88, constraints", len(constraints), constraints)
                G += [constraints]
                print("ce_Q.py line 90, G", len(G), G)
    # col player
    for i in range(num_vars): # action column i
        for j in range(num_vars): # action column j
            if i != j:
                constraints = [0 for n in A]
                for k in range(num_vars):
                    constraints[i + (k * num_vars)] = (
                        - A[i + (k * num_vars)][1]
                        + A[j + (k * num_vars)][1])
                G += [constraints]
                print("line101, G plus some contrains", constraints)
    print("line102, G", len(G), G)
    return np.matrix(G, dtype="float")



def rock_scissor_paper():        # idea from Ron Parr's ppt
    # when playing rock scissor paper, the player's utility is U, and the probability of player choosing Rock,
    # Scissor, and Paper, are R,S, and P, respectively. THen,  (U, R, S, P)  should satisfy:
    # (1) if the player's opponent always chooses Rock, then the player will surely benefit from choosing paper but
    #     suffer loss from choosing scissors. Thus, U = P - S. However, if the opponent chooses something else than Rock,
    #     the player's utility described by P - S will not be as high as when the opponent chooses Rock. Thus, in general,
    #     it is reasonable to say U is always equal to or smaller than P - S, or, U <= P - S
    # (2) Similarly, U <= S - R
    # (3) Similarly, U <= R - P
    # (4) We also have R + S + P = 1. THis is acutally quite informative if written to fit cvxopt.solver.lp, which requires
    #     equatiions with "<=". First, R + S + P <=1
    # (5) We als have R + S + P >= 1. however, since that cvxopt.solver.lp asks for equation written with "<=", this
    #     could be converted into -R -S -P <= -1
    # Thus, we have our G,h, and c, listed below: (Parr refers them as A, b, and c in his slides, and he made a small mistake)
    G = matrix([[1, 0 ,-1, 1], [1, 1, 0, -1], [1, -1, 1, 0], [0, 1, 1, 1], [0, -1, -1, -1]]).T
    h = matrix([0, 0, 0, 1, -1])
    c = matrix([1, 0 , 0, 0])    # because that we only care about U

    # putting G, H, C back to solver.lp:
    sol = solvers.lp(c, G, h)
    print(sol['x'])


if __name__ == '__main__':
    # A = [[6, -6], [3, -3], [7, -7], [10, -10], [6, -6], [3, -3], [7, -7], [10, -10], [6, -6],
         # [6, -6], [3, -3], [7, -7], [10, -10], [6, -6], [3, -3], [7, -7], [10, -10], [6, -6],
         # [6, -6], [3, -3], [7, -7], [10, -10], [6, -6], [3, -3], [7, -7]]
    A = [[6, 6], [2, 7], [7, 2], [0, 0], [1, -3], [4, 2], [0, 0], [1, -3], [4, 2]]
    # A = [[6, 6], [2, 7], [7, 2], [0, 0]]
    # sol = maxmin(A=A, solver='glpk')
    # probs = sol['x']
    # print(probs)

    sol = ce(A=A, solver="glpk")
    probs = sol["x"]
    print(probs)


    # # rock_scissor_paper()
    # # c = matrix([-4., -5.])
    # # G = matrix([[2., 1., -1., 0.], [1., 2., 0., -1.]])
    # # h = matrix([3., 3., 0., 0.])
    #
    # c = matrix([-1., 0., 0., 0.])
    # G = matrix([[1., 0.,-1., 1.], [1., 1., 0., -1.], [1., -1., 1., 0.], [0., 1., 1., 1.], [0., -1., -1., -1.]]).T
    # h = matrix([0., 0., 0., 1., -1.])
    #
    # print("=========G==========\n", G)
    # print(G)
    # print("=========h==========\n", h)
    # print(h)
    # print("=========c==========\n", c)
    # print(c)
    # sol = solvers.lp(c, G, h)
    # print(sol['x'])

