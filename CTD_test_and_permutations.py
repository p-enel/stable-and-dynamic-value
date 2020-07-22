from itertools import product
import pickle as pk
import numpy as np
from sklearn.model_selection import KFold
from dask.distributed import Client, LocalCluster

from decoding_functions import (vectorized_sub_split, combine_xval_folds,
                                get_ridge_param_CTD_vec)
from helper_func import (get_test_fname, get_ens_pp, get_test_pp,
                         get_best_ensembles, get_X_y)

from parameters import (dataseeds, NLOCALWORKERS, DISTCLUSTER, resfolder,
                        alpha_powers, nouterfolds)

################################################################
######################## Data parameters #######################
################################################################

monkeys = ['both', 'M', 'N']
regions = ['OFC', 'ACC']
taskvars = ['value', 'type']
subspaces = [True, False]
stables = [True, False]  # True for stable ensemble and no ensemble and False for dynamic ensemble
ensembles = [True, False]  # True for ensemble and False for full population
permutations = [None, 1000]  # a integer specifying the number of permutations or None for no permutation

################################################################
####################### PARALLELIZATION ########################
################################################################
# Permutation testing is computationally intensive, so it is highly recommended
# to use parallelization. Here we used dask (a parallelization tool) and a
# cluster comprising 128 cores to compute the permutations. Checkout dask and
# how to setup a distributed cluster if you want to use that option and set
# DISTCLUSTER accordingly.

# Whether to perform the calculations in parallel
parallel = True  # If set to False, the following three parameters don't matter

# Whether to use cores on a local machine or use a dask cluster
cluster = 'local'  # 'distributed' or 'local'

###############################################################################
###############################################################################

##### Cluster initialization

if parallel:
    if cluster == 'distributed':
        cluster_ = DISTCLUSTER
    else:
        nworkers = NLOCALWORKERS
        cluster_ = LocalCluster(n_workers=nworkers, threads_per_worker=1, memory_limit='5GB')
    client = Client(cluster_)
    client.upload_file('structure.py')
    client.upload_file('decoding_functions.py')
else:
    client = None

###############################################################################
(dataseed, taskvar, monkey, stable, region, subspace, ensemble, permutes) = next(product(dataseeds, taskvars, monkeys, stables, regions, subspaces, ensembles, permutations))
for (dataseed, taskvar, monkey, stable,
     region, subspace, ensemble, permutes) in product(dataseeds, taskvars, monkeys, stables,
                                                      regions, subspaces, ensembles, permutations):
    print(f"monkey:{monkey} | region:{region} | var:{taskvar} | stable:{stable} | subspace:{subspace} | "
          f"permutes:{permutes} | ensemble:{ensemble} | seed:{dataseed}")

    params_preproc_ens = get_ens_pp(stable)
    params_preproc_test = get_test_pp()

    CTDparams = {'monkey': monkey,
                 'region': region,
                 'taskvar': taskvar,
                 'subspace': subspace,
                 'ensemble': ensemble,
                 'stable': stable}

    filename = get_test_fname(CTDparams, params_preproc_ens,
                              params_preproc_test, dataseed, perm=permutes)
    fullpath = resfolder/filename

    # Skipping already done computations and irrelevant cases
    if fullpath.exists():
        print('Already done, skipping\n')
        continue

    if not stable and subspace:
        print("Skipping subspace for dynamic ensemble\n")
        continue

    if not ensemble and not stable:
        print("Full population and dynamic ensemble are not compatible, skipping\n")
        continue

    ##### Loading up the ensemble results if needed #####
    if ensemble:
        try:
            ens_res, bestens = get_best_ensembles(CTDparams, params_preproc_ens, dataseed)
        except FileNotFoundError:
            print("Ensemble not found, skipping\n")
            continue

    ##### Preparing data #####
    X, y, delaymask = get_X_y(dataseed, params_preproc_test, monkey, region,
                              taskvar)

    Xalpha = X[:, delaymask][:, ::5]

    Xfut, Xalphafut, yfut = client.scatter((X, Xalpha, y), broadcast=True)

    if not ensemble:
        nneurons = X.shape[2]
        bestens = [np.arange(nneurons) for i in range(5)]

    if permutes:
        np.random.seed(dataseed)
        permseeds = np.random.randint(0, 999999, permutes)
    else:
        permseeds = [None]

    outerxval = KFold(n_splits=nouterfolds)
    subxval = KFold(n_splits=2)

    acc_test_futs = []
    for iseed, permseed in enumerate(permseeds):
        ##### Initializing variables #####
        acc_test_xval = []

        all_alphas = []
        for ifold, (restind, testind) in enumerate(outerxval.split(X)):
            ntrials = len(restind)

            pop = bestens[ifold]

            dataalpha = Xalphafut, yfut, None, pop
            alpha = get_ridge_param_CTD_vec(dataalpha, alpha_powers, subspace,
                                            perm_seed=permseed,
                                            trialinds=restind, client=client)

            if parallel:
                acc_test_xval.append(client.submit(vectorized_sub_split, Xfut,
                                                   yfut, restind, testind,
                                                   population=pop,
                                                   permseed=permseed,
                                                   subspace=subspace,
                                                   alpha=alpha,
                                                   mask=delaymask))
            else:
                acc_test_xval.append(vectorized_sub_split(X, y, restind,
                                                          testind,
                                                          population=pop,
                                                          permseed=permseed,
                                                          subspace=subspace,
                                                          alpha=alpha,
                                                          mask=delaymask))

        acc_test_futs.append(client.submit(combine_xval_folds, acc_test_xval))
        if iseed % 250 == 0:
            acc_test = client.gather(acc_test_futs)

    acc_test = client.gather(acc_test_futs)

    if ensemble:
        tosave = ens_res
        tosave['acc_ens'] = ens_res['accuracies']
        tosave['preproc_ens'] = tosave['preproc']
        del tosave['accuracies'], tosave['preproc']
    else:
        tosave = {'monkey': monkey,
                  'region': region,
                  'taskvar': taskvar,
                  'subspace': subspace,
                  'dataseed': dataseed}

    tosave['acc_test'] = acc_test
    tosave['preproc_test'] = params_preproc_test

    with open(fullpath, 'wb') as f:
        pk.dump(tosave, f)
