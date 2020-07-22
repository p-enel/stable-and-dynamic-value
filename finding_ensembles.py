from itertools import product
import pickle as pk
import numpy as np
from sklearn.model_selection import KFold
from dask.distributed import Client, LocalCluster

from decoding_functions import get_ridge_param_CTD_vec, train_test_CTD
from helper_func import (get_ens_filename, get_ens_pp, get_X_y, combine_ens_folds)

from parameters import (ensfolder, dataseeds, DISTCLUSTER, NLOCALWORKERS,
                        alpha_powers, nouterfolds, dynparams)

################################################################
######################## DATA PARAMETERS #######################
################################################################

monkeys = ['both', 'M', 'N']
regions = ['OFC', 'ACC']
taskvars = ['value', 'type']
subspaces = [True, False]
stables = [True, False]  # True for stable ensemble and False for dynamic ensemble

################################################################
####################### PARALLELIZATION ########################
################################################################
# Ensemble searching is computationally intensive, it is likely that the
# computational complexity grows at least quadratically with the number of
# units, so it is highly recommended to use parallelization. Here we used dask
# (a parallelization tool) and a cluster comprising 128 cores to optimize the
# OFC ensembles with a starting population of ~800 neurons in a reasonable
# amount of time. Smaller populations (400 neurons) are much faster to process.
# Checkout dask and how to setup a distributed cluster if you want to use that
# option and set DISTCLUSTER accordingly.

# Whether to perform the calculations in parallel
parallel = True  # If set to False, the following three parameters don't matter

# Whether to use cores on a local machine or use a dask cluster
cluster = 'local'  # 'distributed' or 'local'

###############################################################################
###############################################################################

#### Setting up parallelization
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

#### Looping across all the possible combinations of parameters specified above
#### or in the parameters.py file
for (taskvar, monkey, stable,
     region, subspace, dataseed) in product(taskvars, monkeys, stables,
                                            regions, subspaces, dataseeds):

    CTDparams = {'monkey': monkey,
                 'region': region,
                 'taskvar': taskvar,
                 'subspace': subspace,
                 'stable': stable}

    params_preproc = get_ens_pp(stable)

    # Checking that the exploration has not already been done
    filename = get_ens_filename(CTDparams, params_preproc, dataseed)
    if (ensfolder/filename).exists():
        print("Ensemble optimization for these variables already done")
        continue

    # Skipping non-compatible combinations of parameters
    if not stable and subspace:
        print("Skipping subspace for dynamic ensemble")
        continue
    if taskvar == 'type' and monkey != 'both':
        print('Variable type only with both monkeys, skipping')
        continue

    # Load data with current parameters
    X, y, delaymask = get_X_y(dataseed, params_preproc, monkey, region, taskvar)
    X = X[:, delaymask]  # for ensemble searching, only the delay activity is used

    _, nbins, nneurons = X.shape

    # Outer cross-validation
    outerxval = KFold(n_splits=nouterfolds)
    for ifold, (restind, testind) in enumerate(outerxval.split(X)):
        ntrials = len(restind)

        # Getting the file name to save the current fold's results
        foldname = get_ens_filename(CTDparams, params_preproc, dataseed, ifold=ifold)

        # Skip this fold if it has already been done
        if (ensfolder/foldname).exists():
            print(f'Cross-validation fold #{ifold} already done, starting the next one!!\n')
            continue

        print(f"monkey:{monkey} | region:{region} | variable:{taskvar} | subspace:{subspace} | "
              f"stable:{stable} | #neurons:{nneurons} | seed:{dataseed} | fold{ifold}")

        # Leaving out the test trials, ensemble searching is done on a fraction of the trials only
        if parallel:
            Xfold, yfold = client.scatter((X[restind], y[restind]), broadcast=True)
        else:
            Xfold, yfold = X[restind], y[restind]
        data = Xfold, yfold

        # Initializing results and temporary variables
        remaining = list(range(nneurons))
        ensbl_tmp = []
        ensperfs_tmp = []
        pop = remaining

        #### Getting performance for the full population
        # Getting the ridge parameter
        dataalpha = Xfold, yfold, None, pop
        alpha = get_ridge_param_CTD_vec(dataalpha, alpha_powers, subspace, client=client)

        # Testing on full population
        params = pop, alpha, subspace
        perf = train_test_CTD(data, params, stable, dynparams=dynparams, client=client)

        # Storing result
        if parallel:
            ensperfs_tmp.append(client.gather(perf))
        else:
            ensperfs_tmp.append(perf)

        #### Iterative exploration of ensembles of decreasing size

        for size_ens in np.arange(nneurons-1, 0, -1):
            # Getting the ridge parameter from the n+1 population for all n ensemble
            # otherwise it's too many computations
            dataarg = Xfold, yfold, None, remaining
            alpha = get_ridge_param_CTD_vec(dataarg, alpha_powers, subspace, client=client)

            print(f"monkey:{monkey} | region:{region} | variable:{taskvar} | subspace:{subspace} | "
                  f"stable:{stable}  | #neurons:{size_ens} | seed:{dataseed} | fold{ifold}")

            # Generate all the ensembles of size n that will be tested
            remaintmp = [remaining.copy() for neuronid in remaining]
            [pop.remove(neuronid) for neuronid, pop in zip(remaining, remaintmp)]

            # Testing each of these ensembles
            perfs = []
            for pop in remaintmp:
                params = pop, alpha, subspace
                perf = train_test_CTD(data, params, stable, dynparams=dynparams, client=client)
                perfs.append(perf)
            if parallel:
                perfs = client.gather(perfs)

            # Finding the ensembles with the best performance
            bestid = remaining.pop(np.argmax(perfs))

            # Storing results
            ensbl_tmp.insert(0, bestid)
            ensperfs_tmp.insert(0, np.nanmax(perfs))

        # Inserting the last neuron
        ensbl_tmp.insert(0, remaining[0])

        with open(ensfolder/foldname, 'wb') as f:
            pk.dump((ensperfs_tmp, ensbl_tmp, testind), f)

    combine_ens_folds(CTDparams, stable, dataseed)
