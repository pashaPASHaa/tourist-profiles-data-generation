import os
import h5py
import jax
import numpy as np
import picos


KEY = jax.random.PRNGKey(1)  # reproducibility


def zipf(r, s, p0):
    # rank and frequency are an inverse relation (r >= 0, s >= 1, 0 < p0 <= 1)
    # zipf probabilities assume no normalisation
    p = p0 * (r+1)**(-s)
    return p


def pack(x, jaxify=True):
    # for jax
    o = tuple(x) if jaxify else x
    return o


def _init_behavioural_data(N, J):
    """
    N: number of users
    J: number of items
    """
    print(f'Initializing behavioural data. There are {N} users and {J} items')

    # probability mass function of users activities
    # number of visited POIs
    T_pmi = np.array([1,2,3,4,5], dtype='i8')
    T_pmf = np.array([1,2,3,3,1], dtype='f8')
    T_pmf = T_pmf / np.sum(T_pmf)

    # probability mass function of items popularity
    # assume 50% of users visited top-ranked POIs[0]
    J_pmi = np.arange(J)
    J_pmf = zipf(r=J_pmi, s=1.0, p0=0.5)

    # probability mass function of different groups
    # assume 15 groups of users
    G_pmi = np.array([ 0, 1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14], dtype='i8')
    G_pmf = np.array([40,40,30,30,20,20,10,10,7,7, 5, 5, 2, 2, 1], dtype='f8')
    G_pmf = G_pmf / np.sum(G_pmf)

    bdata = {
        'N': N,
        'J': J,
        'T_pmi': pack(T_pmi),
        'T_pmf': pack(T_pmf),
        'J_pmi': pack(J_pmi),
        'J_pmf': pack(J_pmf),
        'G_pmi': pack(G_pmi),
        'G_pmf': pack(G_pmf),
    }
    return bdata


def _init_user_data(N, n_hid):
    a = 0.2
    b = 9.8
    z = jax.random.beta(KEY, a=a, b=b, shape=(N, n_hid), dtype='f4')
    return z


def _init_item_data(J, n_hid, sort_by_rank=True):

    # each centroid represents (average of) a group of semantically similar items
    # different centroids render dissimilar (unlike) groups
    a = 0.2
    b = 0.8
    ncent = 20
    cents = jax.random.beta(KEY, a=a, b=b, shape=(ncent, n_hid), dtype='f4')  # sparse representation a < 1 and b < 1

    # init group (aka cohort) representatives
    # each group consists of (nearly) the same number of items (each item belongs to only one group)
    cohorts = np.zeros(J, dtype='i4')
    for i, a in enumerate(np.array_split(np.arange(J, dtype='i4'), ncent)):
        cohorts[a] = i

    # init popularity ranks
    # this vector ties (associates) initialised items with the behavioral data items
    # at the same time it ensures organic behaviour when items belonging to the same
    # group (cohort) are scattered in the tail but not concentrated
    ranks = jax.random.permutation(KEY, J)

    # -------------------------------------------------------------------------

    # define optimisation problem
    P = picos.Problem('POIs similarity within groups')

    lam_min = 0.1
    lam_max = 0.5
    lam_obj = 0.7

    # define optimisation var
    # vector (when optimised) represents drifts in attributes from group origin
    x = picos.RealVariable('x', shape=(J, n_hid), lower=-1, upper=1)

    # define perturbation guideline var
    # design of this vector is to guide/push (when optimising) vector `x` in a direction collinear with `r`
    r = np.array(jax.random.uniform(KEY, shape=(J, n_hid), minval=-1, maxval=1), dtype='f8')
    r[r <= 0] = -1
    r[r >= 0] = +1

    z = 0

    # define optimisation constraints (linear)
    # define similarities shifts between items as a function of rank divergency
    for i in range(ncent):
        # item set that represents ith cohort
        A = {
            j for j in range(J) if cohorts[j] == i
        }
        print(A)
        # constant origin
        c = picos.Constant(f'c{i}', np.array(cents[i], dtype='f8'), shape=(1, n_hid))

        for j in A:

            # abs
            z += (r[j,:]|x[j,:])

            # box var constraints
            P += c+x[j,:] >= 0
            P += c+x[j,:] <= 1

            # sim var constraints
            for k in A:
                if k <= j:
                    continue
                else:
                    r_div = abs(ranks[j]-ranks[k]).item()/J
                    # sim
                    # 1. the lower the rank (item is more popular), higher the similarity to the group (cohort) origin
                    # 2. and similar (by popularity) items must be close in the vector space
                    if ranks[j] < ranks[k]:
                        P += (c|c+x[j,:]) - (c|c+x[k,:]) >= lam_min*r_div*(c|c)
                        P += (c|c+x[j,:]) - (c|c+x[k,:]) <= lam_max*r_div*(c|c)
                    else:
                        P += (c|c+x[k,:]) - (c|c+x[j,:]) >= lam_min*r_div*(c|c)
                        P += (c|c+x[k,:]) - (c|c+x[j,:]) <= lam_max*r_div*(c|c)

    # define optimisation obj
    P.set_objective('max', lam_obj*z - abs(x)**2)  # --> max
    P.solve(
        solver='cvxopt',
        abs_prim_fsb_tol=1e-4, abs_dual_fsb_tol=1e-4, abs_ipm_opt_tol=1e-4, abs_bnb_opt_tol=1e-4,
        rel_prim_fsb_tol=1e-4, rel_dual_fsb_tol=1e-4, rel_ipm_opt_tol=1e-4, rel_bnb_opt_tol=1e-4,
        verbosity=1,
    )

    if P.status != 'optimal':
        raise ValueError('Primal solution state claimed empty')

    # -------------------------------------------------------------------------

    # init items
    f = jax.numpy.clip(
        cents[cohorts] + x.np,
        a_min=0,
        a_max=1,
    )

    # sort items by popularity (first item assumed to be the most popular)
    if sort_by_rank:
        f = f[ranks]

    return f


def init_data(N, J):
    """
    N: number of users
    J: number of items
    seed: key to populate random number generator

    Utility = dot(z, f), where z -- user representation and f -- item representation
    """

    # init behavioural data
    bdata = _init_behavioural_data(N, J)
    print(f'Initializing user and item vector-representation data')

    n_hid = 64

    # -------------------------------------------------------------------------
    # init users [N, n_hid]
    # produced users will be learned later to satisfy apriori behaviour
    z = _init_user_data(N, n_hid)

    # -------------------------------------------------------------------------
    # init items [J, n_hid]
    # produced items will always stay fixed
    f = _init_item_data(J, n_hid)

    return bdata, z, f


def dump_condensed_artefacts(file, bdata, z, f):

    os.makedirs(os.path.dirname(file), mode=0o777, exist_ok=True)

    with h5py.File(file, 'w') as o:
        # attributes
        o.attrs['N'] = bdata['N']
        o.attrs['J'] = bdata['J']
        # behavioural data
        o.create_dataset('T_pmi', dtype='i', data=bdata['T_pmi'])
        o.create_dataset('T_pmf', dtype='f', data=bdata['T_pmf'])
        o.create_dataset('J_pmi', dtype='i', data=bdata['J_pmi'])
        o.create_dataset('J_pmf', dtype='f', data=bdata['J_pmf'])
        o.create_dataset('G_pmi', dtype='i', data=bdata['G_pmi'])
        o.create_dataset('G_pmf', dtype='f', data=bdata['G_pmf'])
        # users and items
        o.create_dataset('z', dtype='f', data=z)
        o.create_dataset('f', dtype='f', data=f)
        print(f'Dumped successfully condensed artefacts in file {os.path.basename(file)}')


def load_condensed_artefacts(file, jaxify=True):

    with h5py.File(file, 'r') as o:
        bdata = {
            'N': o.attrs['N'],
            'J': o.attrs['J'],
            'T_pmi': pack(o['T_pmi'][()], jaxify),
            'T_pmf': pack(o['T_pmf'][()], jaxify),
            'J_pmi': pack(o['J_pmi'][()], jaxify),
            'J_pmf': pack(o['J_pmf'][()], jaxify),
            'G_pmi': pack(o['G_pmi'][()], jaxify),
            'G_pmf': pack(o['G_pmf'][()], jaxify),
        }
        z = o['z'][()]
        f = o['f'][()]
        print(f'Loaded successfully condensed artefacts in file {os.path.basename(file)}')

    return bdata, z, f


if __name__ == '__main__':

    _init_item_data(J=20, n_hid=8, sort_by_rank=True)
