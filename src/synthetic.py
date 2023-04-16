import jax
import numpy as np

from jax import jit, vmap
from jax.numpy.linalg import norm


def cnf(N, pmf, eps=1e-8):
    # splits N users with respect to probability mass function
    s = np.cumsum(pmf)+eps
    s = s*N
    s = s.astype(int)
    return s


def compute_p_k(k, U, T_pmi, T_pmf, beta):  # --> p_k = Prob{POIs[k] is chosen}

    p_k = 0
    u_x = jax.numpy.exp(beta*(U-jax.numpy.max(U, axis=1, keepdims=True)))  # for numerical stability

    for t, q in zip(T_pmi, T_pmf):
        # define the soft indicator function as ratio u_k / (u_k + sum_{j=t+1:J} u_(j))
        # divide
        softindicator_k = jax.numpy.divide(
            u_x[:,k],
            u_x[:,k] + jax.numpy.sum(u_x, axis=1) - jax.numpy.sum(jax.lax.top_k(u_x, t)[0], axis=1)
        )
        # update probability
        p_k += jax.numpy.mean(softindicator_k)*q

    return p_k


compute_p_k_vectorized = jit(
    fun=vmap(compute_p_k, (0, None, None, None, None), 0), static_argnames=('T_pmi', 'T_pmf'))  # --> p


def loss_fn(param_tree, f, bdata, beta, verbose=False):
    """
    z: [N, n_hid] vector representation of users in `param_tree` dict (trainable parameters)
    f: [J, n_hid] vector representation of items
    beta: inverse temperature
    """
    J = bdata['J']

    # true probabilities
    q = jax.numpy.array(bdata['J_pmf'])

    # eval probabilities
    p = compute_p_k_vectorized(
        jax.numpy.arange(J),
        jax.numpy.matmul(param_tree['z'], f.T),
        bdata['T_pmi'],
        bdata['T_pmf'],
        beta,
    )

    # compute normalised divergence between true distribution and computed probabilities
    pq = jax.numpy.sum(q*(p/q-1)**2)  # --> minimise

    if verbose:
        print(f'optim pq_loss={jax.lax.stop_gradient(pq):.6f}')

    return pq


def regularization_fn(param_tree, f, bdata, lam_wg, lam_bg):

    N = bdata['N']

    # penalty for deviation from normal distribution of preferences
    ll_wg = 0
    cents = []
    l = 0
    for r in cnf(N, bdata['G_pmf']):
        # select a group of users
        g = param_tree['z'][l:r]
        gcent = jax.numpy.mean(g, axis=0)
        cents.append(gcent)
        # update log prior
        ll_wg += jax.numpy.mean(jax.numpy.sum((g-gcent)*(g-gcent), axis=1))
        # -->
        l = r

    # penalty for nonzero cosine similarity between group centroids
    cents = jax.numpy.array(cents, dtype='f4')
    cents_norm = norm(cents, axis=1)
    cents_csim = jax.numpy.matmul(cents, cents.T) / jax.numpy.outer(cents_norm, cents_norm)
    ll_bg = jax.numpy.sum(jax.numpy.triu(cents_csim, k=1)**2)

    # aggregate
    ll = lam_wg*ll_wg + lam_bg*ll_bg
    print(
        f'll_wg={jax.lax.stop_gradient(lam_wg*ll_wg):>9.6f}  '
        f'll_bg={jax.lax.stop_gradient(lam_bg*ll_bg):>9.6f}  '
    )
    return ll


def check_model(param_tree, f, bdata):

    N = bdata['N']
    J = bdata['J']
    U = jax.lax.stop_gradient(jax.numpy.matmul(param_tree['z'], f.T))

    if U.shape != (N, J):
        raise ValueError('Shape mismatch')

    # iterate over all users and collect their made choices
    UI_mat = np.zeros((N, J), dtype='f8')

    for n in range(N):

        if (n+1) % 100 == 0:
            print(f'processed {n/N:.2f} users', end='\r')

        for t, q in zip(  # over feasible number of choices
                bdata['T_pmi'],
                bdata['T_pmf'],
        ):
            # gather top t choices
            _, top_choices = jax.lax.top_k(U[n], t)
            # update
            for choice in top_choices:
                UI_mat[n, choice] += q

    Q = np.array(bdata['J_pmf'], dtype='f8')
    P = np.mean(UI_mat, axis=0)

    chisq = jax.numpy.sum(Q*(P/Q-1)**2)
    kl_QP = jax.numpy.sum(Q*(jax.numpy.log(Q+1e-8) - jax.numpy.log(P+1e-8)))
    kl_PQ = jax.numpy.sum(P*(jax.numpy.log(P+1e-8) - jax.numpy.log(Q+1e-8)))

    print(f'estimated over a population of {N} users KL: QP_loss={kl_QP:.6f} PQ_loss={kl_PQ:.6f}\n'
          f'chi2_loss={chisq:.6f}')

    return P, UI_mat
