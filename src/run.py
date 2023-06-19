import argparse
import time
import jax
import numpy as np
import optax

from jax import value_and_grad
from environment import dump_condensed_artefacts, load_condensed_artefacts, init_data
from synthetic import loss_fn, regularization_fn


# higher beta is better, because soft indicators approach real indicators
# mostly very large values may cause convergence problems
BETA = 6
MAX_EPOCHS = 1000000


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
        bdata, z, f, t_choices, g_lrs = init_data(N=N, J=J)
        afile = f'out/experiment_condensed.hdf5'
    else:
        bdata, z, f, t_choices, g_lrs = load_condensed_artefacts(args.artefact_file)
        N = bdata['N']
        J = bdata['J']
        afile = f'out/experiment_condensed.hdf5'
    print(
        f'Configuration for users and items:  '
        f'z.shape={z.shape}  '
        f'f.shape={f.shape}  '
        f'users #t choices={t_choices}  '
        f'group lrs=\n{g_lrs}\n'
    )
    param_tree = {'z': z}

    # step decay lr sheduler: lr = warmup + 1e-2 * (0.8)**(count//2500)
    # maps count to lr
    lr = optax.warmup_exponential_decay_schedule(
        init_value=1e-4,
        peak_value=1e-1, warmup_steps=5000, transition_steps=5000, decay_rate=0.8, staircase=True, end_value=1e-4,
    )
    # optimizer
    optim = optax.chain(
        optax.clip(1.),
        optax.adam(lr),
    )
    state = optim.init(param_tree)

    for epoch in range(MAX_EPOCHS):

        verbose = ((epoch+1) % 100 ==  0)
        debuger = ((epoch+1) % 100 >= 90)

        tic = time.perf_counter()
        # get loss and regularizer
        loss, g_ = value_and_grad(loss_fn)(param_tree, f, bdata, t_choices, beta=BETA, verbose=verbose)
        regn, r_ = value_and_grad(regularization_fn)(param_tree, g_lrs, lam_wg=1e-6, lam_bg=1e-2, verbose=verbose)
        grad = jax.tree_map(
            lambda g,r: g+r,
            g_,
            r_,
        )
        # get update
        param_tree_update, state = optim.update(grad, state)
        # set update (is additive)
        # z -= lr*dz
        param_tree = jax.tree_map(lambda p,u: p+u, param_tree, param_tree_update)
        toc = time.perf_counter()

        if debuger:
            g_quant = np.quantile(g_['z'], (0.01, 0.99))
            r_quant = np.quantile(r_['z'], (0.01, 0.99))
            z_quant, u_quant = np.quantile(param_tree['z'], (0.01, 0.99)), np.quantile(param_tree_update['z'], (0.01, 0.99))

            print(f"L= {loss: .4f}  "
                  f"g= {g_quant[0]: .6f} {g_quant[1]: .6f}  |  "
                  f"R= {regn: .4f}  "
                  f"r= {r_quant[0]: .6f} {r_quant[1]: .6f}  |  "
                  f"ztree= {z_quant[0]:>9.4f} {z_quant[1]:>9.4f}  |  "
                  f"zgrad= {u_quant[0]:>9.4f} {u_quant[1]:>9.4f}")

        if verbose:
            print(f'-----------------------------------------------------------')
            print(f'epoch={epoch+1:>05d} | lr={lr(epoch):.4f} | time/epoch={toc-tic:.2f} s | loss={loss+regn:>.6f}')
            print(f'')
            if loss+regn < 1e-6:
                break
            dump_condensed_artefacts(
                afile,
                bdata,
                z=param_tree['z'],
                f=f,
                t_choices=t_choices,
                g_lrs=g_lrs,
            ) if (epoch+1) % (10*100) == 0 else (None)

    # dump trained artefacts in file for later use
    dump_condensed_artefacts(
        afile,
        bdata,
        z=param_tree['z'],
        f=f,
        t_choices=t_choices,
        g_lrs=g_lrs,
    )
    # load trained artefacts (sanity check)
    bdata_loaded, z_loaded, f_loaded, t_choices_loaded, g_lrs_loaded = load_condensed_artefacts(afile)
    print(
        'Sanity check:',
        np.allclose(z_loaded, param_tree['z']),
        np.allclose(f_loaded, f),
        np.allclose(t_choices_loaded, t_choices),
        np.allclose(g_lrs_loaded, g_lrs),
    )
