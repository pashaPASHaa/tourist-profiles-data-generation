import jax
import numpy as np

from jax import jit, vmap
from jax.numpy.linalg import norm


# def __compute_p(U, t_choices, t_max, beta):  # --> p[k] = Prob{POIs[k] is chosen}
#
#     # define the soft indicator functions
#     # assume the user `n` chooses exactly `t_choices[n]` items
#     softindicators = jax.numpy.zeros(U.shape, dtype='f4')
#     for t in range(1, t_max+1):
#         a = jax.numpy.sum(jax.lax.top_k(U, t-1)[0], axis=1, keepdims=True)
#         b = jax.numpy.sum(jax.lax.top_k(U, t+1)[0], axis=1, keepdims=True)
#         U_diff = U - (1/2)*(b-a)
#         # soft choices assignment (user `n` selects or not a particular item `k`)
#         softindicators += jax.numpy.where(
#             (t_choices == t)[:,None], jax.lax.logistic(beta*U_diff), 0
#         )
#     # define the sums
#     p = jax.numpy.mean(softindicators, axis=0)
#     s = jax.numpy.mean(jax.numpy.abs(t_choices - jax.numpy.sum(softindicators, axis=1)))
#     return p, s


def __compute_topksoftmax(u, m, beta):
    # u: [J] array
    # m: [k] array
    # beta: inverse temperature
    s = jax.numpy.sum(jax.nn.softmax(-beta*(u[None,:] - m[:,None])**2, axis=1), axis=0)  # [J] array
    return s


def __compute_p(U, t_choices, t_max, beta):  # --> p[k] = sum_{n=1:N}(n chooses POIs[k])

    # define the soft indicator functions
    # assume the user `n` chooses exactly `t_choices[n]` items
    softindicators = jax.numpy.zeros(U.shape, dtype='f4')
    for t in range(1, t_max+1):

        m, _ = jax.lax.top_k(U, t)
        s = vmap(__compute_topksoftmax, (0, 0, None), 0)(U, m, beta)

        softindicators += jax.numpy.where(
            (t_choices == t)[:,None],
            s,
            0,
        )
    # define the sums
    p = jax.numpy.sum(softindicators, axis=0)
    s = jax.numpy.sum(jax.numpy.sum(softindicators, axis=1)/t_choices - 1)
    return p, s


compute_p = jit(
    __compute_p,
    static_argnames='t_max',
)
# compute_p = __compute_p


def loss_fn(param_tree, f, bdata, t_choices, beta, verbose=False):
    """
    z: [N,n_hid] vector representation of users in `param_tree` dict (trainable parameters)
    f: [J,n_hid] vector representation of items
    bdata: behavioural data
    t_choices: [N] number of made choices
    beta: inverse temperature
    """

    # true probabilities (counter)
    q = jax.numpy.array(bdata['j_pmf'])

    # eval probabilities (counter)
    p, s = compute_p(
        jax.numpy.matmul(param_tree['z'], f.T),
        t_choices,
        t_choices.max().item(),
        beta,
    )

    # compute normalised divergence between true distribution and computed probabilities
    qsum = q.sum()
    psum = p.sum()
    q /= qsum
    p /= psum

    qlog = jax.numpy.log(q)
    plog = jax.numpy.log(p)
    a = jax.numpy.sum(q*(qlog-plog)) + jax.numpy.sum(p*(plog-qlog)) + jax.numpy.sum((p/q-1)**2)
    b = s

    l_pq = a + 0.01*b  # --> minimise

    if verbose:
        print(f'l_pq={jax.lax.stop_gradient(a):>9.6f}  '
              f'l_pq={jax.lax.stop_gradient(b):>9.6f}  '
              f'l_pq={jax.lax.stop_gradient(l_pq):>9.6f}')

    return l_pq


def regularization_fn(param_tree, g_lrs, lam_wg, lam_bg, verbose=False):

    # penalty for deviation from normal distribution of preferences
    cents, l_wg = [], 0
    for l, r in g_lrs:
        # select a group of users
        g = param_tree['z'][l:r]
        gcent = jax.numpy.mean(g, axis=0)
        cents.append(gcent)
        # update log prior
        l_wg += jax.numpy.mean(jax.numpy.sum((g-gcent)*(g-gcent), axis=1))
    cents = jax.numpy.array(cents)

    # penalty for nonzero cosine similarity between group centroids
    cents_norm = jax.lax.stop_gradient(
        norm(cents, axis=1)
    )
    cents_csim = jax.numpy.matmul(cents, cents.T) / (jax.numpy.outer(cents_norm, cents_norm) + 1e-8)
    l_bg = jax.numpy.sum(jax.numpy.triu(cents_csim, k=1)**2)

    # aggregate
    L = lam_wg*l_wg + lam_bg*l_bg

    if verbose:
        print(
            f'l_wg={jax.lax.stop_gradient(lam_wg*l_wg):>9.6f}  '
            f'l_bg={jax.lax.stop_gradient(lam_bg*l_bg):>9.6f}  '
        )
    return L


def check_model(param_tree, f, bdata, t_choices, eps=1e-8):

    N = bdata['N']
    J = bdata['J']
    U = jax.lax.stop_gradient(jax.numpy.matmul(param_tree['z'], f.T))

    if U.shape != (N, J):
        raise ValueError('Shape mismatch')

    # iterate over all users and collect their made choices
    UI_mat = np.zeros((N, J), dtype='i4')

    for n in range(N):

        if (n+1) % 100 == 0:
            print(f'processed {n/N:.2f} users', end='\r')

        # gather top t choices
        u = U[n]
        t = t_choices[n]
        UI_mat[n, np.argpartition(u, -t)[-t:]] += 1

    Q = np.array(bdata['j_pmf'], dtype='i4')
    P = np.sum(UI_mat, axis=0)

    Qsum = Q.sum()
    q = Q/Qsum
    p = P/Qsum

    chisq = np.sum(q*(p/q-1)**2)
    kl_QP = np.sum(q*(jax.numpy.log(q+eps) - jax.numpy.log(p+eps)))
    kl_PQ = np.sum(p*(jax.numpy.log(p+eps) - jax.numpy.log(q+eps)))

    print(f'estimated over a population of {N} users KL: QP_loss={kl_QP:.6f} PQ_loss={kl_PQ:.6f}\n'
          f'chi2_loss={chisq:.6f}')

    return P, UI_mat
