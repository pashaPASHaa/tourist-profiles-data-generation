import argparse
import os
import time
import jax
import numpy as np
import optax

from jax import grad
from synthetic import loss_fn, regularization_fn
from environment import dump_condensed_artefacts, load_condensed_artefacts, init_data


# higher beta is better, because soft indicators approach real indicators
# mostly very large values may cause convergence problems
BETA = 6


if __name__ == '__main__':

    # python3 run.py --help
    parser = argparse.ArgumentParser(description='Synthetic data generation protocol')
    parser.add_argument('--N', type=int, help='number of users', required=True)
    parser.add_argument('--J', type=int, help='number of items', required=True)
    parser.add_argument('--artefact_file', type=str, help='file with pretrained parameters', required=False)
    args = parser.parse_args()

    # init or load environment
    if not args.artefact_file:
        N = args.N
        J = args.J
        bdata, z, f = init_data(N=N, J=J)
        afile = f'out/experiment_condensed.hdf5'
    else:
        bdata, z, f = load_condensed_artefacts(args.artefact_file)
        N = bdata['N']
        J = bdata['J']
        afile = f'out/experiment_condensed_from_pretrained.hdf5'
    print(
        f'Configuration for users and items:  '
        f'z.shape={z.shape}  '
        f'f.shape={f.shape}  '
    )
    param_tree = {'z': z}

    # grad
    grad_loss_fn = grad(loss_fn, argnums=0)
    grad_regularization_fn = grad(regularization_fn, argnums=0)

    # step decay lr sheduler: lr = 1e-3 * (0.8)**np.floor((1+count)/1e4)
    # maps count to lr
    lr = optax.exponential_decay(
        1e-3,
        transition_steps=10000, decay_rate=0.8, staircase=True,
    )

    # optimizer
    optim = optax.adam(lr)
    state = optim.init(param_tree)

    print(f'\nStart learning script...')

    for j in range(100000):

        # ---------------------------------------------------------------------

        tic = time.perf_counter()

        grads = jax.tree_map(
            lambda g,r: g+r,
            grad_loss_fn(param_tree, f, bdata, beta=BETA),
            grad_regularization_fn(param_tree, f, bdata, lam_wg=0.001, lam_bg=0.001),
        )

        # get update
        param_tree_update, state = optim.update(
            grads,
            state,
        )

        # set update (is additive)
        # z -= lr*dz
        param_tree = jax.tree_map(lambda p,u: p+u, param_tree, param_tree_update)

        toc = time.perf_counter()

        # ---------------------------------------------------------------------

        if (j+1) % 10 == 0:
            loss = jax.lax.stop_gradient(loss_fn(param_tree, f, bdata, beta=BETA, verbose=True))
            print(f'-----------------------------------------------------------')
            print(f'epoch={j+1:>05d} | time per epoch={toc-tic:.2f} sec | total loss={loss:>.6f}')
            print(f'')
            if loss < 1e-6:
                break
            dump_condensed_artefacts(
                afile,
                bdata,
                z=param_tree['z'],
                f=f,
            ) if (j+1) % (10*100) == 0 else (None)

    # dump trained artefacts in file for later use
    dump_condensed_artefacts(
        afile,
        bdata,
        z=param_tree['z'],
        f=f,
    )
    # load trained artefacts (sanity check)
    bdata_loaded, z_loaded, f_loaded = load_condensed_artefacts(afile)
    print(
        'Sanity check:',
        np.allclose(z_loaded, param_tree['z']),
        np.allclose(f_loaded, f),
    )
