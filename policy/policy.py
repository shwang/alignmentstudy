import numpy as np
import scipy.optimize

# SETTING UP THE LINEAR PROGRAM
#
def inverse_reinforcement_learn(N, gamma, lbda, pi, P_pi, list_P_a, R_max=1):
    """
    N is the number of states.
    gamma is the discount factor.
    lbda is the L1-norm penalty factor on the reward space.
    pi, the optimal policy, is a N-lengthed list where pi[i] is the optimal action
        at state i.
    P_pi is an N x N matrix where P_pi[i, j] is the probability of
        transitioning from state i to state j when following the
        optimal policy.
    list_P_a is an list where list_P_a[i] is the N x N transition
        matrix for the policy of always taking action i. The list
        has an entry for every action, and actions have a preset order.
    """
    bounds = reward_max_constraints(N, R_max)
    A_ub, b_ub = irl_optimal_policy_constraints(N, gamma, P_pi, list_P_a)
    def concat(A, b):
        nonlocal A_ub, b_ub
        A_ub = np.concatenate([A_ub, A])
        b_ub = np.concatenate([b_ub, b])
    concat(*abs_aux_variables(N))
    concat(*max_Q_aux_variables(N, gamma, pi, P_pi, list_P_a))

    C = objective_coefficients(N, lbda, P_pi, gamma)
    return scipy.optimize.linprog(C, A_ub, b_ub, bounds=bounds)

def variable_layout_shard(N):
    """
    N is the number of states.
    num_actions is the number of actions.
    """
    virtual_X = np.zeros(N*N + N*N + N)
    # The first N*N variables are the rewards R(i, j).
    # The next N*N variables are the aux variables abs(R(i, j)).
    # The late N variables are the aux variables max(over a\a*, Q_pi(state i, a)).

def reward_max_constraints(N, R_max):
    """ Let's make the first N^2 variables the reward parameters. """
    assert R_max > 0
    bounds = [(None, None)] * (2*N*N + N)
    for i in range(N*N):
        bounds[i] = (-R_max, R_max)
    return bounds

def irl_optimal_policy_constraints(N, gamma, P_pi, list_P_a, inv_result=None):
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
    A_ub = np.zeros([N*num_actions, 2*N*N + N])
    b_ub = np.zeros(N*num_actions)

    if not inv_result:
        I = np.eye(N)
        inv_result = np.linalg.inv(I - gamma*P_pi)

    for act_idx, P_a in enumerate(list_P_a):
        P_diff = P_pi - P_a
        temp = gamma * np.dot(P_diff, inv_result)
        for z in range(N):
            row = z + act_idx * N
            for i in range(N):
                for j in range(N):
                    col = i*N + j
                    A_ub[row, col] = temp[z, i] * P_pi[i, j]
                    if z == i:
                        A_ub[row, col] += P_diff[i, j]
                    # sum >= 0   <==>  -sum <= 0
                    #
                    # 0 is the lower bound for this linear combination. However,
                    # scipy uses upper bounds. So we must take the negative of each
                    # entry in A_ub.
                    A_ub[row, col] *= -1

    return A_ub, b_ub

def abs_aux_variables(N):
    """
    N is the number of states.
    """
    A_ub = np.zeros([2*N*N, 2*N*N + N])
    b_ub = np.zeros(2*N*N)

    for i in range(N):
        for j in range(N):
            row = 2*(i*N + j)
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

    return A_ub, b_ub

def max_Q_aux_variables(N, gamma, pi, P_pi, list_P_a, inv_result=None):
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
    A_ub = np.zeros([N*(num_actions - 1), 2*N*N + N], dtype='float')
    b_ub = np.zeros(N*(num_actions - 1))

    if not inv_result:
        I = np.eye(N)
        inv_result = np.linalg.inv(I - gamma*P_pi)

    nonpolicy_list = []
    for s in range(N):
        for a in range(num_actions):
            if pi[s] != a:
                nonpolicy_list.append((s, a))

    for row, (s, a) in enumerate(nonpolicy_list):
            P_a = list_P_a[a]
            temp_q = gamma * np.dot(P_a[s, :], inv_result)

            col = (N*N + N*N) + s
            #  max_Q(s) >= Q_pi(s, a).
            #  max_Q(s) - Q_pi(s, a) >= 0
            # -max_Q(s) + Q_pi(s, a) <= 0
            A_ub[row, col] = -1  # max_Q(s)

            # Q_pi is a linear product of R(i, j)s
            for i in range(N):
                for j in range(N):
                    A_ub[row, i*N + j] = temp_q[i] * P_pi[i, j]
                    if i == s:
                        A_ub[row, i*N + j] += P_a[i, j]

    return A_ub, b_ub

def objective_coefficients(N, lbda, P_pi, gamma=None, inv_result=None):
    C = np.zeros(N*N + N*N + N)

    if not inv_result:
        I = np.eye(N)
        inv_result = np.linalg.inv(I - gamma*P_pi)

    # Add each V(s)
    for i in range(N):
        for j in range(N):
            for z in range(N):
                C[i*N + j] += inv_result[z, i] * P_pi[i, j]

    # L1 norm penalty over R
    for i in range(N*N, 2*N*N):
        C[i] = -lbda

    # subtract each max(over a\pi(a), Q(s, a))
    for i in range(2*N*N, 2*N*N + N):
        C[i] = -1

    # The original problem was formulated as a maximization.
    # To transform into a minimization, multiply every entry by -1.
    for i in range(len(C)):
        C[i] *= -1

    return C
