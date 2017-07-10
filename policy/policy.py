import numpy as np

# MINIMAL MDP
# -----
# |0|1|
# -----
# policy: pi = {0: ">", 1: "<"}
#
# coefficients: [ a  b ]  <==> [ R(0,0)  R(0,1) ]
#               [ c  d ]       [ R(1,0)  R(1,1) ]

P_pi = np.array([[0, 1], [1, 0]])
P_stop = np.array([[1, 0], [0, 1]])
P_left = np.array([[1, 0], [1, 0]])
P_right = np.array([[0, 1], [0, 1]])
gamma = 0.5

I = np.eye(2)
inv_part = np.linalg.inv((I - gamma*P_pi))

# SETTING UP THE LINEAR PROGRAM
#

def variable_layout_shard(N):
    """
    N is the number of states.
    num_actions is the number of actions.
    """
    virtual_X = numpy.zeros(N*N + N*N + N)
    # The first N*N variables are the rewards R(i, j).
    # The next N*N variables are the aux variables abs(R(i, j)).
    # The late N variables are the aux variables max(over a\a*, Q_pi(state i, a)).

def reward_max_constraints_shard(N, bounds, R_max):
    """ Let's make the first N^2 variables the reward parameters. """
    bounds = np.zeros(N*N + N*N + N)
    for i in range(N*N):
        bounds[i] = -R_max, R_max

def irl_optimal_policy_constraints_shard(N, gamma, P_pi, list_P_a, inv_result=None):
    """
    N is the number of states.
    gamma is the discount factor.
    P_pi is an N x N matrix where P_pi[i, j] is the probability of
        transitioning from state i to state j when following the
        optimal policy.
    list_P_a is an list where list_P_a[i] is the N x N transition
        matrix for the policy of always taking action i. The list
        has an entry for every action, and actions have a preset order.
    (optional) inv_result is the precomputed value of the commonly used matrix
        inversion (I - gamma*P_pi)^-1
    """
    num_actions = len(list_P_a)
    A_ub = np.zeros([N*num_actions, N*N])
    b_ub = np.zeros(N*num_actions)

    if not inv_result:
        I = np.eye(N)
        inv_result = np.linalg.inv(I - gamma*P_pi)

    for act_idx, P_a in enumerate(list_P_a):
        P_diff = P_pi - P_a
        temp = gamma * numpy.dot(P_diff, inv_result)
        for z in range(N):
            for i in range(N):
                for j in range(N):
                    # sum >= 0   <==>  -sum <= 0
                    #
                    # 0 is the lower bound for this linear combination. However,
                    # scipy uses upper bounds. So we must take the negative of each
                    # entry in A_ub.
                    row = z + act_idx * N
                    A_ub[row, i*N + j] = -(temp[z, i] * P_pi[i, j] + P_diff[i, j])

def abs_aux_variables_shard(N, gamma, P_pi, list_P_a, inv_result=None):
    """
    N is the number of states.
    """
    A_ub = np.zeros([2*N, N*N])
    b_ub = np.zeros(N*N)

    for i in range(N):
        for j in range(N):
            row = i*N + j
            # abs(R(i,j)) is at least R(i, j)
            # implies -abs(R(i,j)) is at most -R(i, j)
            # So -abs(R(i,j)) + R(i,j) <= 0
            A_ub[row, N*N + (i*N + j)] = -1
            A_ub[row, i*N + j] = 1

            # abs(R(i,j)) is at least -R(i, j)
            # implies -abs(R(i,j)) is at most R(i, j)
            # So -abs(R(i, j)) - R(i,j) <= 0
            A_ub[row + 1, N*N + (i*N + j)] = -1
            A_ub[row + 1, i*N + j] = -1

def max_Q_aux_variables_shard(N, gamma, pi, P_pi, list_P_a, inv_result=None):
    """
    N is the number of states.
    gamma is the discount factor.
    pi, the optimal policy, is a N-lengthed list where pi[i] is the optimal action
        at state i.
    P_pi is an N x N matrix where P_pi[i, j] is the probability of
        transitioning from state i to state j when following the
        optimal policy.
    list_P_a is an list where list_P_a[i] is the N x N transition
        matrix for the policy of always taking action i. The list
        has an entry for every action, and actions have a preset order.
    (optional) inv_result is the precomputed value of the commonly used matrix
        inversion (I - gamma*P_pi)^-1
    """
    num_actions = len(list_P_a)
    A_ub = np.zeros([N, N*(num_actions - 1)])
    b_ub = np.zeros(N * num_actions)

    if not inv_result:
        I = np.eye(N)
        inv_result = np.linalg.inv(I - gamma*P_pi)

    for s in range(N):
        for a in range(num_actions):
            if pi[s] == a:
                continue
            P_a = list_P_a[a]
            temp_q = gamma * numpy.dot(P_a[i, :], inv_result)

            row = s * num_actions + row_a
            col = (N*N + N) + s
            #  max_Q(s) >= Q_pi(s, a).
            #  max_Q(s) - Q_pi(s, a) >= 0
            # -max_Q(s) + Q_pi(s, a) <= 0
            A_ub[row, col] = -1  # max_Q(s)

            # Q_pi is a linear product of R(i, j)s
            for i in range(N):
                for j in range(N):
                    A_ub[row, i*N + j] = temp_q[1, i] * P_pi[i, j]
                    if i == s:
                        A_ub[row, i*N + j] += P_a[i, j]

def objective_coefficients_shard(N, lbda, P_pi, gamma=None, inv_result=None):
    C = np.zeros(N*N + N*N + N)

    if not inv_result:
        I = np.eye(N)
        inv_result = np.linalg.inv(I - gamma*P_pi)

    # Add each V(s)
    for i in range(N):
        for j in range(N):
            C[i*N + j] = inv_result[i, j] * P_pi[i, j]

    # L1 norm penalty over R
    for i in range(N*N, 2*N*N):
        C[i] = -lbda

    # subtract each max(over a\pi(a), Q(s, a))
    for i in range(2*N*N, 2*N*N + N):
        C[i] = -1

    # The original problem was formulated as a maximization.
    # To transform into a minimization, multiply every entry by -1.
    for i in len(C):
        C[i] *= -1
