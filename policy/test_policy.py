import unittest
import numpy as np
from policy import inverse_reinforcement_learn, reward_max_constraints, \
    irl_optimal_policy_constraints, abs_aux_variables, max_Q_aux_variables, \
    objective_coefficients

class TestRewardMaxConstraints(unittest.TestCase):
    def test_tiny(self):
        bounds = reward_max_constraints(2, 3)
        expected = [(-3, 3)] * 4 + [(None, None)] * (4 + 2)
        self.assertEqual(bounds, expected)

def constraint_sanity_check(t, N, A_ub, b_ub):
    t.assertEqual(len(A_ub.shape), 2)
    t.assertEqual(A_ub.shape[1], 2*N*N + N)
    t.assertEqual(A_ub.shape[0], b_ub.shape[0])
    for x in b_ub:
        t.assertEqual(x, 0)

class TestAbsAux(unittest.TestCase):
    def test_tiny(self):
        A_ub, b_ub = abs_aux_variables(2)
        constraint_sanity_check(self, 2, A_ub, b_ub)
        self.assertTrue(np.array_equal(b_ub, np.zeros(2*2*2)))

        expected_A_ub = [
            [ 1, 0, 0, 0,-1, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0,-1, 0, 0, 0, 0, 0],
            [ 0, 1, 0, 0, 0,-1, 0, 0, 0, 0],
            [ 0,-1, 0, 0, 0,-1, 0, 0, 0, 0],
            [ 0, 0, 1, 0, 0, 0,-1, 0, 0, 0],
            [ 0, 0,-1, 0, 0, 0,-1, 0, 0, 0],
            [ 0, 0, 0, 1, 0, 0, 0,-1, 0, 0],
            [ 0, 0, 0,-1, 0, 0, 0,-1, 0, 0],
                ]
        self.assertTrue(np.array_equal(A_ub, expected_A_ub))

class TestOptimalPolicyConstraints(unittest.TestCase):
    def test_tiny(self):
        P_left = np.array([[1, 0], [1, 0]], dtype='float')
        P_right = np.array([[0, 1], [0, 1]], dtype='float')
        P_stop = np.array([[1, 0], [0, 1]], dtype='float')
        P_pi = np.array([[0, 1], [1, 0]], dtype='float')
        list_P_a = (P_left, P_right, P_stop)
        A_ub, b_ub = irl_optimal_policy_constraints(2, 0.5, P_pi, list_P_a)
        constraint_sanity_check(self, 2, A_ub, b_ub)

        expected_A_ub = -1 * np.array([
            # Coeff(i, j) is at index i*N + j
            # {0: C(0,0), 1: C(0,1), 2: C(1,0), 3: C(1,1)]
            [-1, 2/3, 1/3, 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 0, 1/3, 2/3, -1, 0, 0, 0, 0, 0, 0],
            [-1, 2/3, 1/3, 0, 0, 0, 0, 0, 0, 0],
            [ 0, 1/3, 2/3, -1, 0, 0, 0, 0, 0, 0],
        ], dtype='float')

        np.testing.assert_array_almost_equal(A_ub, expected_A_ub)


class TestMaxQAux(unittest.TestCase):
    def test_tiny(self):
        P_left = np.array([[1, 0], [1, 0]], dtype='float')
        P_right = np.array([[0, 1], [0, 1]], dtype='float')
        P_stop = np.array([[1, 0], [0, 1]], dtype='float')
        P_pi = np.array([[0, 1], [1, 0]], dtype='float')
        list_P_a = (P_left, P_right, P_stop)
        A_ub, b_ub = irl_optimal_policy_constraints(2, 0.5, P_pi, list_P_a)
        pi = (1, 0)

        A_ub, b_ub = max_Q_aux_variables(2, 0.5, pi, P_pi, list_P_a)
        constraint_sanity_check(self, 2, A_ub, b_ub)

        expected_A_ub = np.array([
            # Coeff(i, j) is at index i*N + j
            # {0: C(0,0), 1: C(0,1), 2: C(1,0), 3: C(1,1)]
            [ 1, 2/3, 1/3, 0, 0, 0, 0, 0,-1, 0],  # max_Q(0) >= Q_pi(0, left)
                                                  # [1, 2/3, 1/3, 0]R - Q_pi(0, 0) <= 0
            [ 1, 2/3, 1/3, 0, 0, 0, 0, 0,-1, 0],  # max_Q(0) >= Q_pi(0, stop)
            [ 0, 1/3, 2/3, 1, 0, 0, 0, 0, 0,-1],  # max_Q(1) >= Q_pi(1, right)
            [ 0, 1/3, 2/3, 1, 0, 0, 0, 0, 0,-1],  # max_Q(1) >= Q_pi(1, stop)
        ], dtype='float')
        np.testing.assert_array_almost_equal(A_ub, expected_A_ub)

class TestObjectiveCoefficients(unittest.TestCase):
    def test_tiny(self):
        N = 2
        lbda = 0.6
        gamma = 0.5
        P_pi = np.array([[0, 1], [1, 0]], dtype='float')
        C = objective_coefficients(N, lbda, P_pi, gamma)
        self.assertEqual(np.array(C).shape, (10,))

        expected = [0,-2,-2,0, 0.6,0.6,0.6,0.6, 1,1]
        np.testing.assert_array_almost_equal(list(C), expected)

class TestInverseReinforcementLearn(unittest.TestCase):
    def test_tiny(self):
        N = 2
        lbda = 0.6
        gamma = 0.5
        P_left = np.array([[1, 0], [1, 0]], dtype='float')
        P_right = np.array([[0, 1], [0, 1]], dtype='float')
        P_stop = np.array([[1, 0], [0, 1]], dtype='float')
        P_pi = np.array([[0, 1], [1, 0]], dtype='float')
        list_P_a = (P_left, P_right, P_stop)
        pi = (1, 0)

        result = inverse_reinforcement_learn(N, gamma, lbda, pi, P_pi, list_P_a)
        np.testing.assert_array_almost_equal(result.x, [-1, 1, 1, -1, 1, 1, 1, 1, 0, 0])
