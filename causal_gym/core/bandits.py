import numpy as np
from joblib import Parallel, delayed
from numpy.random.mtrand import beta
from scipy.optimize import brenth
from typing import Tuple
from contextlib import contextmanager

from .causal_graph import with_default

@contextmanager
def seeded(seed=None):
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)
        yield
        np.random.set_state(st0)
    else:
        yield

def pick_randomly(xs):
    return xs[np.random.randint(len(xs))]
def rand_argmax(xs):
    max_val = np.nanmax(xs)
    if max_val is np.nan:
        return pick_randomly(np.arange(len(xs)))

    max_indices = np.where(xs == max_val)[0]
    if not len(max_indices):
        print(xs, max_val)

    if len(max_indices) == 1:
        return max_indices[0]
    else:
        return pick_randomly(max_indices)

def KL(mu_x, mu_star, epsilon=1e-12):
    """ Kullback-Leibler Divergence with two parameters from two Bernoulli distributions """
    if mu_x == mu_star:
        return 0

    if mu_star <= 0 or mu_star >= 1:
        return np.inf

    if epsilon == 0:
        if mu_x == 0:
            return np.log(1 / (1 - mu_star))
        elif mu_x == 1:
            return np.log(1 / mu_star)

    return mu_x * np.log((mu_x + epsilon) / (mu_star + epsilon)) + (1 - mu_x) * np.log((1 - mu_x + epsilon) / (1 - mu_star + epsilon))


def sup_KL(mu_ref, divergence, lower=None):
    """ Find largest mu that satisfies KL(mu_ref, mu) <= divergence """
    if divergence <= 0:
        return mu_ref
    if KL(mu_ref, 1.0) <= divergence:
        return 1.0
    return brenth(lambda x: KL(mu_ref, x) - divergence, with_default(lower, mu_ref), 1)


class U_keeper:
    """  Keep look-ahead U values to save unnecessary computation (more effective if there is a large number of arms) """

    def __init__(self, K_: int, T: int):
        self.K_ = K_
        self.T = T
        self.lookahead_U = None
        self.lookahead_t = None
        self.step_sizes = None

    def update_U(self, t, f, mu_hat, N, U, arm_x):
        init_step_size = self.K_ * 2
        T = self.T
        K_ = self.K_
        if t <= 5 * K_:
            fval = f(t)
            for i in range(K_):
                U[i] = sup_KL(mu_hat[i], fval / N[i])
            if t == 5 * K_:
                ahead_t = min(t + init_step_size, T)
                ft2 = f(ahead_t)
                self.lookahead_U = np.array([sup_KL(mu_hat[i], ft2 / N[i]) for i in range(K_)])
                self.lookahead_t = np.ones((len(mu_hat),)) * ahead_t
        else:
            fval = f(t)
            threshold = U[arm_x] = sup_KL(mu_hat[arm_x], fval / N[arm_x])

            for arm in np.where(self.lookahead_U >= threshold)[0]:
                if arm != arm_x:
                    U[arm] = sup_KL(mu_hat[arm], fval / N[arm])

            for arm in set(np.where(self.lookahead_t == t)[0]) | {arm_x}:
                self.lookahead_t[arm] = ahead_t = t + init_step_size
                self.lookahead_U[arm] = sup_KL(mu_hat[arm], f(ahead_t) / N[arm])


def default_kl_UCB_func(t, value_at_small_t=1):
    if t < 3:
        return value_at_small_t
    else:
        return np.log(t) + 3 * np.log(np.log(t))


def kl_UCB(T: int, mu, f=None, seed=None, faster=True, prior_SF=None, **_kwargs):
    """Bernoulli kl-UCB"""
    if f is None:
        f = default_kl_UCB_func

    K_ = len(mu)
    faster = faster and K_ > 4
    N, mu_hat = np.zeros((K_,)), np.zeros((K_,))
    if prior_SF is not None:
        S, F = prior_SF
        for arm in range(K_):
            N[arm] = S[arm] + F[arm]
            mu_hat[arm] = S[arm] / (S[arm] + F[arm])

    ukeeper = U_keeper(K_, T)

    arms_selected = np.zeros((T,)).astype(int)
    rewards = np.zeros((T,))
    with seeded(seed):
        rands = np.random.rand(T)
        shuffled_arms = np.random.choice(K_, K_, replace=False)
        for t, arm_x in enumerate(shuffled_arms):
            reward_y = int(rands[t] <= mu[arm_x])
            N[arm_x] += 1
            mu_hat[arm_x] += (reward_y - mu_hat[arm_x]) / N[arm_x]

            arms_selected[t] = arm_x
            rewards[t] = reward_y

        U = np.array([sup_KL(mu_hat[i], f(K_) / N[i]) for i in range(K_)])

        # compute
        for t in range(K_, T):
            arm_x = rand_argmax(U)
            # select
            reward_y = int(rands[t] <= mu[arm_x])

            arms_selected[t] = arm_x
            rewards[t] = reward_y

            # update for next
            N[arm_x] += 1
            mu_hat[arm_x] += (reward_y - mu_hat[arm_x]) / N[arm_x]

            if not faster:
                fval = f(t + 1)
                U = np.array([sup_KL(mu_hat[i], fval / N[i]) for i in range(K_)])
            else:
                ukeeper.update_U(t + 1, f, mu_hat, N, U, arm_x)

    return arms_selected, rewards


def thompson_sampling(T: int, mu, seed=None, prior_SF=None, **_kwargs):
    """ Bernoulli Thompson Sampling with known mu"""
    K_ = len(mu)
    S, F, theta = np.zeros((K_,)), np.zeros((K_,)), np.zeros((K_,))
    if prior_SF is not None:
        S, F = prior_SF

    arms_selected = np.zeros((T,)).astype(int)
    rewards = np.zeros((T,))
    with seeded(seed):
        random_numbers = np.random.rand(T)

        for t in range(T):
            theta = [beta(S[i] + 1, F[i] + 1) for i in range(K_)]
            arm_x = rand_argmax(theta)
            reward_y = int(random_numbers[t] <= mu[arm_x])

            arms_selected[t] = arm_x
            rewards[t] = reward_y

            if reward_y == 1:
                S[arm_x] += 1
            else:
                F[arm_x] += 1

    return arms_selected, rewards


def play_bandits(T: int, mu, algo: str, repeat: int, n_jobs=1) -> Tuple[np.ndarray, np.ndarray]:
    if algo == 'TS':
        par_result = Parallel(n_jobs=n_jobs, verbose=100)(delayed(thompson_sampling)(T, mu, seed=trial) for trial in range(repeat))
    elif algo == 'UCB':
        par_result = Parallel(n_jobs=n_jobs, verbose=100)(delayed(kl_UCB)(T, mu, seed=trial) for trial in range(repeat))
    else:
        raise AssertionError(f'unknown algo: {algo}')

    return (np.vstack(tuple(arms_selected for arms_selected, _ in par_result)),
            np.vstack(tuple(rewards for _, rewards in par_result)))