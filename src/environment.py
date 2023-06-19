import os
import h5py
import jax
import numpy as np
import picos


KEY = jax.random.PRNGKey(1)  # reproducibility


def cnf(N, pmf):
    assert np.isclose(sum(pmf), 1), 'Error in pmf (sum != 1)'
    # splits N users with respect to probability mass function `pmf`
    pmF = np.cumsum(pmf)
    pmN = np.fmin(N, (0.5+N*pmF).astype('i4'))
    return pmN


def zipf(J, s):
    # rank and frequency are an inverse relation
    r = np.arange(J, dtype='f4')
    # zipf probabilities
    p = (r+1)**(-s) / sum((r+1)**(-s))
    return p


def pack(x, jaxify=True):
    # for jax
    o = tuple(x) if jaxify else x
    return o


def _init_probabilistic_behavioural_data(N, J):
    """
    N: number of users
    J: number of items
    """
    print(f'Initializing behavioural data. There are {N} users and {J} items')

    # probability mass function of users activities
    # number of visited POIs
    t_pmi = np.array([1,2,3,4,5], dtype='i4')
    t_pmf = np.array([1,2,3,3,1], dtype='f4')
    t_pmf = t_pmf / np.sum(t_pmf)

    # probability mass function of different groups
    # assume 15 groups of users
    g_pmi = np.array([ 0, 1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14], dtype='i4')
    g_pmf = np.array([40,40,30,30,20,20,10,10,7,7, 5, 5, 2, 2, 1], dtype='f4')
    g_pmf = g_pmf / np.sum(g_pmf)

    # probability mass function of items popularity
    # assume that most of the visits are to the top-rated POIs[0]
    j_pmi = np.arange(J, dtype='i4')
    j_pmf = zipf(J=J, s=1.0)

    bdata = {
        'N': N,
        'J': J,
        't_pmi': pack(t_pmi),
        't_pmf': pack(t_pmf),
        'g_pmi': pack(g_pmi),
        'g_pmf': pack(g_pmf),
        'j_pmi': pack(j_pmi),
        'j_pmf': pack(j_pmf),
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
    a = 0.5
    b = 0.5
    ncent = 10
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

    lam_min = 0.1  # min corr coeff
    lam_max = 1.0  # max corr coeff
    lam_obj = 0.8  # strength of perturbation

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

    Utility = dot(z, f),
    where `z` is user representation and `f` is item representation
    """

    # init behavioural data
    bdata = _init_probabilistic_behavioural_data(N, J)
    n_hid = 64

    # -------------------------------------------------------------------------
    # init users [N,n_hid]
    # produced users will be learned later to satisfy apriori behaviour
    z = _init_user_data(N, n_hid)

    # -------------------------------------------------------------------------
    # init users groups
    g_lrs = []
    l = 0
    for r in cnf(N, bdata['g_pmf']):
        g_lrs.append((l, r))
        l = r
    g_lrs = np.array(g_lrs, dtype='i4')

    # -------------------------------------------------------------------------
    # init items [J,n_hid]
    # produced items will always stay fixed
    f = _init_item_data(J, n_hid)

    # -------------------------------------------------------------------------
    # init how many items each user is going to choose
    t = np.array(bdata['t_pmi'], dtype='i4')
    p = np.array(bdata['t_pmf'], dtype='f4')
    t_choices = jax.random.choice(KEY, t, shape=(N,), p=p)

    # -------------------------------------------------------------------------
    # from now I assume that `bdata` contains counters
    j_pmi_cnt = np.array(bdata['j_pmi'], dtype='i4')
    j_pmf_cnt = np.array(bdata['j_pmf'], dtype='i4')
    C = sum(t_choices)  # total number of choices made in the environment
    l = 0
    for i, r in enumerate(cnf(C, bdata['j_pmf'])):
        j_pmf_cnt[i] = r-l
        l = r

    bdata = {
        'N': N,
        'J': J,
        'j_pmi': pack(j_pmi_cnt),
        'j_pmf': pack(j_pmf_cnt),
    }

    return bdata, z, f, t_choices, g_lrs


def dump_condensed_artefacts(file, bdata, z, f, t_choices, g_lrs):

    os.makedirs(os.path.dirname(file), mode=0o777, exist_ok=True)

    with h5py.File(file, 'w') as o:
        # attributes
        o.attrs['N'] = bdata['N']
        o.attrs['J'] = bdata['J']
        # behavioural data
        o.create_dataset('j_pmi', dtype='i', data=bdata['j_pmi'])
        o.create_dataset('j_pmf', dtype='i', data=bdata['j_pmf'])
        # users and items
        o.create_dataset('z', dtype='f', data=z)
        o.create_dataset('f', dtype='f', data=f)
        o.create_dataset('t_choices', dtype='i', data=t_choices)
        o.create_dataset('g_lrs', dtype='i', data=g_lrs)
        print(f'Dumped successfully condensed artefacts in file {os.path.basename(file)}')


def load_condensed_artefacts(file, jaxify=True):

    with h5py.File(file, 'r') as o:
        bdata = {
            'N': o.attrs['N'],
            'J': o.attrs['J'],
            'j_pmi': pack(o['j_pmi'][()], jaxify),
            'j_pmf': pack(o['j_pmf'][()], jaxify),
        }
        z = o['z'][()]
        f = o['f'][()]
        t_choices = o['t_choices'][()]
        g_lrs = o['g_lrs'][()]
        print(f'Loaded successfully condensed artefacts in file {os.path.basename(file)}')

    return bdata, z, f, t_choices, g_lrs


if __name__ == '__main__':

    bdata, z, f, t_choices, g_lrs = init_data(N=100, J=20)
    print(t_choices.tolist())
    print(g_lrs)
