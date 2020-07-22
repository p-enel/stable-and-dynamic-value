import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import KFold


class NestedXval():
    '''A generator for nested cross-validation that ensures that there is the
    same number of trials for each class in training.

    It is necessary to have the same number of trials in each category to
    vectorize the training of the decoder so that the training of all 6
    decoders (in one vs one scheme) is done simultaneously.
    '''
    def __init__(self, n_outer_splits=None):
        '''Nested crossvalidation to get the same number of trials in each class for training'''
        self.nouter = n_outer_splits
        self.ninner = 2
        self.outerxval = KFold(n_splits=n_outer_splits)

    def split(self, targets):
        '''Returns a generator that splits data in test, train and subspace
        with the same number of trials in each category
        '''
        labels, counts = np.unique(targets, return_counts=True)
        nclasses = len(labels)
        if not np.all(counts[0] == counts) and max(counts) - min(counts) > 1:
            raise ValueError("The number of trials in each class in not consistant")

        interleaved_outer = np.concatenate(list(zip(*[np.where(targets == label)[0] for label in labels])))

        leftovers = []
        for iclass in np.where(np.min(counts) < counts)[0]:
            leftovers.append(np.where(targets == labels[iclass])[0][-1])
        interleaved_outer = np.concatenate((interleaved_outer, np.array(leftovers))).astype(int)

        targets_ = targets[interleaved_outer]

        outersplit = self.outerxval.split(targets)
        for ioutsplit in range(self.nouter):
            restinds, testinds = next(outersplit)
            ntrain_per_class = np.ceil(len(restinds) / 2 / nclasses).astype(int)
            inner_inds_by_class = [np.where(targets_[restinds] == label)[0] for label in labels]

            traininds = np.concatenate(list(zip(*[restinds[classinds[:ntrain_per_class]] for classinds in inner_inds_by_class])))
            subinds = np.concatenate([restinds[classinds[ntrain_per_class:]] for classinds in inner_inds_by_class])

            testinds = interleaved_outer[testinds]
            traininds = interleaved_outer[traininds]
            subinds = interleaved_outer[subinds]
            yield np.sort(testinds), np.sort(traininds), np.sort(subinds)

            traininds = np.concatenate(list(zip(*[restinds[classinds[:-ntrain_per_class:-1]] for classinds in inner_inds_by_class])))
            subinds = np.concatenate([restinds[classinds[-ntrain_per_class::-1]] for classinds in inner_inds_by_class])
            testinds = interleaved_outer[testinds]
            traininds = interleaved_outer[traininds]
            subinds = interleaved_outer[subinds]
            yield np.sort(testinds), np.sort(traininds), np.sort(subinds)


def sub_split(targets, trainind):
    '''Cross-validation generator for the decoder and subspace trials

    Function to split training trials in training and subspace trials, ensuring
    that there is the same number of trials in each class for training.

    Parameters
    ----------
    targets : np.array - The targets (or y values)
    trainind : np.array - The indices of the training trials

    Returns
    -------
    Generator for each fold. Yields a tuple of np.array, one array for the
      training trials and one array for the subspace
    '''
    targets = targets[trainind]
    labels = np.unique(targets)
    nclasses = len(labels)
    ntrain_per_class = np.ceil(len(targets) / 2 / nclasses).astype(int)
    inner_inds_by_class = [np.where(targets == label)[0] for label in labels]

    ridgeind = np.concatenate(list(zip(*[classinds[:ntrain_per_class] for classinds in inner_inds_by_class])))
    subind = np.concatenate([classinds[ntrain_per_class:] for classinds in inner_inds_by_class])

    yield np.sort(trainind[ridgeind]), np.sort(trainind[subind])

    ridgeind = np.concatenate(list(zip(*[classinds[:-ntrain_per_class:-1] for classinds in inner_inds_by_class])))
    subind = np.concatenate([classinds[-ntrain_per_class::-1] for classinds in inner_inds_by_class])

    yield np.sort(trainind[ridgeind]), np.sort(trainind[subind])


def combine_xval_folds(acc_fold):
    '''Combine CTD cross-validation accuracies by averaging them

    Parameters
    ----------
    acc_fold : list of np.array<bins * bins> - The CTD accuracy matrices of all
      the cross-validation folds

    Returns
    -------
    np.array<bins * bins> - The averaged CTD accuracy
    '''
    return np.stack(acc_fold).mean(0)


def get_acc_mean(acc):
    '''Averages all accuracy points in a CTD accuracy matrix

    Parameters
    ----------
    acc : np.array<bins * bins> - The CTD accuracy matrix

    Returns
    -------
    float - The stability score
    '''
    return acc.mean()


def gaussian_func(x, mu, sigma, a):
    '''A gaussian function

    Parameters
    ----------
    x : np.array - the x values to feed the function
    mu : float - the mean of the gaussian
    sigma : float - the standard deviation of the gaussian
    a : float - a scaling coefficient

    Returns
    -------
    The transformed values in a np.array for each value in x.
    '''
    b = .25
    return a * np.exp(-(x-mu)**2/(2*sigma**2)) + b


def get_CT_score(acc, bounds, dstraining=None):
    '''Get the "locality" score from a CTD accuracy

    Fits a gaussian on each vector formed by training at a time bin and testing
    at all time bins. Calculates the ratio between the maximum of the gaussian
    divided by its standard deviation. Then averages all the ratios to get a
    locality score.

    Parameters
    ----------
    acc : np.array<bins * bins> - The accuracy matrix of a CTD
    bounds : a 2-element tuple of 3-element np.array - the bounds for
      gaussian fitting for the locality score, e.g.
      #                   mu      sigma    a
                np.array([0,      2,       0]),  # lower bounds
                np.array([0, np.inf,       1]))  # upper bounds
    dstraining : int - if the CTD was trained on a subset of the time bins.
      Every 'dstraining' bins have been selected for a down sampled training.

    Returns
    -------
    Locality score
    '''
    if dstraining is None:
        dstraining = 1
    opted = []
    nbinstrain, nbinstest = acc.shape
    x = np.arange(nbinstest)

    scores = np.empty(nbinstrain)
    for ibintrain in range(nbinstrain):
        data = acc[ibintrain, :]
        data[data < .25] = .25
        ibintest = dstraining - 1 + ibintrain * dstraining
        params0 = [ibintest, 10, .5]
        bounds[0][0] = np.max((ibintest - 5, 0))
        bounds[1][0] = np.min((ibintest + 5, nbinstest))
        try:
            optparams = curve_fit(gaussian_func, x, data, params0, bounds=bounds)[0]
        except RuntimeError:
            optparams = [0, 0, 0]
            scores[ibintrain] = 0
        else:
            max_val = np.max(gaussian_func(x, *optparams))
            scores[ibintrain] = (max_val - .25) / optparams[1] * 1000
        opted.append(optparams)
    return np.mean(scores)


###############################################################################
################################## VECTORIZED #################################
###############################################################################

def vectorized_xval_CTD(X, y, population=None, permseed=None, subspace=True,
                        alpha=1, mask=None, dstraining=None):
    '''Cross-validation of vectorized cross-temporal decoding

    Cross-validation using a custom generator to ensure that the number of
    trials in each class is identical.

    Parameters
    ----------
    X : np.array<trials * bins * neurons> - data
    y : np.array<ntrials> - targets
    population : np.array of int - the indices of the neurons included
    permseed : int - a seed for permutation testing
    subspace : bool - whether to use a subspace or not
    alpha : float - the ridge parameter
    mask : np.array<nbins> of bool - which bins to take to build the subspace
    dstraining : int - Every 'dstraining' bins will be selected for a down
      sampled training

    Returns
    -------
    accuracy : np.array<bins * bins> - the CTD accuracy averaged across folds
      of the cross-validation
    '''
    acc_fold = []
    if subspace:
        nestedxval = NestedXval(n_outer_splits=5)
        for testind, trainind, subind in nestedxval.split(y):
            acc_fold.append(vectorized_CTD_job(X, y, trainind, testind, subind=subind, population=population,
                                               permseed=permseed, alpha=alpha, mask=mask, dstraining=dstraining)[0])
    else:
        kfold = KFold(n_splits=5)
        for trainind, testind in kfold.split(y):
            acc_fold.append(vectorized_CTD_job(X, y, trainind, testind, population=population, permseed=permseed,
                                               alpha=alpha, mask=mask, dstraining=dstraining)[0])
    accuracy = combine_xval_folds(acc_fold)
    return accuracy


def vectorized_sub_split(X, y, restind, testind,
                         population=None, permseed=None, subspace=True, alpha=1, mask=None):
    '''Cross-validation of vectorized cross-temporal decoding (for testing)

    Cross-validation of CTD with pre-defined testing trials. Used for testing,
    when the testing trials have already been set aside and only the remaining
    trials must be split into training and subspace trials

    Parameters
    ----------
    X : np.array<trials * bins * neurons> - data
    y : np.array<ntrials> - targets
    restind : np.array - The indices of all the trials except the testing trials
    testind : np.array - The indices of the testing trials
    population : np.array of int - the indices of the neurons included
    permseed : int - a seed for permutation testing
    subspace : bool - whether to use a subspace or not
    alpha : float - the ridge parameter
    mask : np.array<nbins> of bool - which bins to take to build the subspace

    Returns
    -------
    accuracy : np.array<bins * bins> - the CTD accuracy averaged across folds
      of the cross-validation
    '''
    if subspace:
        acc_split = []
        for ridgeind, subind in sub_split(y, restind):
            acc_split.append(vectorized_CTD_job(X, y, ridgeind, testind, subind=subind,
                                                population=population, permseed=permseed, alpha=alpha, mask=mask)[0])
        accuracy = combine_xval_folds(acc_split)
    else:
        accuracy = vectorized_CTD_job(X, y, restind, testind,
                                      population=population, permseed=permseed, alpha=alpha, mask=mask)[0]
    return accuracy


def vectorized_CTD_job(X, y, trainind, testind,
                       population=None, **kwargs):
    '''Calling vectorized cross-temporal decoding with a given ensemble

    Parameters
    ----------
    X : np.array<trials * bins * neurons> - data
    y : np.array<ntrials> - targets
    trainind : np.array - The indices of the training trials
    testind : np.array - The indices of the testing trials
    population : np.array of int - the indices of the neurons included
    **kwargs : keyword arguments for function 'vectorized_CTD'

    Returns
    -------
    accuracy : np.array<bins * bins> - The CTD matrix accuracy
    testout : np.array<bins * bins * test trials> - The output of the
      classifier for each pair of train and test bins, for each trial
    '''
    if population is None:
        population = np.arange(X.shape[-1])
    newX = X[..., population]
    return vectorized_CTD(newX, y, trainind, testind, **kwargs)


def vectorized_CTD(X, y, trainind, testind,
                   alpha=1, subind=None, mask=None, permseed=None, dstraining=None):
    '''Vectorized cross-temporal decoding

    This is a vectorized version of the cross-temporal decoding algorithm. The
    six decoders (in a one vs one scheme) are trained simultaneously thanks to
    vectorization which considerably speeds up computations. Unfortunately it
    makes the code less readable. The decoding algorithm was inspired by scikit
    learn's implementation of ridge regression. Note that to be able to
    vectorize training and testing, each class must have the same number of
    training and testing trials.

    Parameters
    ----------
    X : np.array<trials * bins * neurons> - data
    y : np.array<ntrials> - targets
    trainind : np.array
      The indices of the training trials
    testind : np.array
      The indices of the testing trials
    alpha : float - the ridge parameter
    subind : np.array
      The indices of trials used to define the subspace. If not None, a
      subspace will be defined
    mask : np.array<nbins> of bool - which bins to take to build the subspace
    permseed : int - a seed for permutation testing, only the training trials
      are shuffled
    dstraining : int
      Every 'dstraining' bins will be selected for a down sampled training

    Returns
    -------
    accuracy : np.array<bins * bins> - The CTD matrix accuracy
    testout : np.array<bins * bins * test trials> - The output of the
      classifier for each pair of train and test bins, for each trial
    '''
    subspace = bool(subind is not None)

    if dstraining is None:
        dstraining = 1

    ntrials, nbins, _ = X.shape
    if mask is None:
        mask = range(nbins)
    labels = np.unique(y)
    nclasses = len(labels)

    Xtrain, Xtest = X[trainind], X[testind]
    ytrain, ytest = y[trainind], y[testind]
    if permseed is not None:
        np.random.seed(permseed)
        np.random.shuffle(ytrain)

    if dstraining is not None:
        Xtrain = Xtrain[:, dstraining-1::dstraining]
        nbinstrain = Xtrain.shape[1]
    else:
        nbinstrain = nbins

    Xtrain = Xtrain.transpose((1, 0, 2))

    if subspace:
        ysub = y[subind]
        Xsub = X[:, mask][subind].mean(1)  # Averaging over time bins
        Xsub = np.stack([Xsub[ysub == label].mean(0) for label in labels])
        subspace = PCA()
        subspace.fit(Xsub)
        Xtrain = (Xtrain - subspace.mean_[None, None, :]) @ subspace.components_.T[None, ...]

    # We need to have the exact same number of trials for each class
    _, traincounts = np.unique(ytrain, return_counts=True)
    mintrials = np.min(traincounts)
    if not np.all(traincounts[0] == traincounts):
        mintrials = np.min(traincounts)
        keptind = []
        for iclass, count in enumerate(traincounts):
            if count > mintrials:
                inds = np.where(ytrain == labels[iclass])[0][:-(count-mintrials)]
            else:
                inds = np.where(ytrain == labels[iclass])[0]
            keptind.append(inds)
        keptind = np.concatenate(keptind)
    else:
        keptind = np.arange(len(ytrain))
    ytrain_cut = ytrain[keptind]
    Xtrain_cut = Xtrain[:, keptind]

    nestimators = (nclasses * (nclasses - 1)) // 2
    nsamples = mintrials * 2
    nfeatures = Xtrain_cut.shape[-1]
    ytrain_ = np.empty((nestimators, nsamples))
    Xtrain_ = np.empty((nbinstrain, nestimators, nsamples, nfeatures))

    k = 0
    for c1 in range(nclasses):
        for c2 in range(c1+1, nclasses):
            cond = np.logical_or(ytrain_cut == c1, ytrain_cut == c2)
            ytrain_[k, ytrain_cut[cond] == c1] = -1
            ytrain_[k, ytrain_cut[cond] == c2] = 1
            Xtrain_[:, k] = Xtrain_cut[:, cond]
            k += 1

    X_offset = Xtrain_.mean(2, keepdims=True)
    Xtrain_ -= X_offset

    if nfeatures > nsamples:
        XXT = Xtrain_ @ Xtrain_.transpose((0, 1, 3, 2))
        XXT = XXT + np.eye(XXT.shape[-1])[None, None, ...] * alpha
        dual_coef = np.linalg.solve(XXT, ytrain_.reshape(1, nestimators, -1))
        coefs = Xtrain_.transpose((0, 1, 3, 2)) @ dual_coef.reshape(dual_coef.shape[0], nestimators, -1, 1)
    else:
        XTX = Xtrain_.transpose((0, 1, 3, 2)) @ Xtrain_
        Xy = Xtrain_.transpose((0, 1, 3, 2)) @ ytrain_.reshape((1, ytrain_.shape[0], -1, 1))
        XTX = XTX + np.eye(XTX.shape[-1])[None, None, ...] * alpha
        coefs = np.linalg.solve(XTX, Xy)

    intercepts = - X_offset @ coefs
    Xtest_ = Xtest.reshape(Xtest.shape[0] * Xtest.shape[1], Xtest.shape[2])
    if subspace:
        Xtest_ = subspace.transform(Xtest_)
    scores = (Xtest_ @ coefs) + intercepts
    scores = scores.reshape(scores.shape[:-1])
    predictions = (scores > 0).astype(np.int)
    nsamples = predictions.shape[-1]
    predsT = predictions.transpose((0, 2, 1))
    scoresT = scores.transpose((0, 2, 1))
    votes = np.zeros((nbinstrain, nsamples, nclasses))
    sum_of_confidences = np.zeros((nbinstrain, nsamples, nclasses))
    k = 0
    for i in range(nclasses):
        for j in range(i + 1, nclasses):
            sum_of_confidences[:, :, i] -= scoresT[:, :, k]
            sum_of_confidences[:, :, j] += scoresT[:, :, k]
            votes[predsT[:, :, k] == 0, i] += 1
            votes[predsT[:, :, k] == 1, j] += 1
            k += 1
    transformed_confidences = (sum_of_confidences /
                               (3 * (np.abs(sum_of_confidences) + 1)))
    preds = np.argmax(votes + transformed_confidences, 2)
    preds = preds.reshape(nbinstrain, len(testind), nbins)
    accuracy = (preds == ytest[None, :, None]).mean(1)
    return accuracy, preds


def ensemble_mean_acc_vec(X, y, **kwargs):
    '''Get stable score for a given ensemble

    Parameters
    ----------
    X : np.array<trials * bins * neurons> - data
    y : np.array<trials> - targets
    population : np.array of int - the indices of the neurons included
    alpha : float - the ridge parameter
    subspace : bool - whether to use a subspace or not

    Returns
    -------
    float - Stable score
    '''
    accuracy = vectorized_xval_CTD(X, y, **kwargs)
    perf = get_acc_mean(accuracy)
    return perf


def ensemble_dynamic_acc_vec(X, y, bounds, dstraining=None, **kwargs):
    '''Get "locality" or dynamic score for a given ensemble

    Parameters
    ----------
    X : np.array<trials * bins * neurons> - data
    y : np.array<trials> - targets
    bounds : a 2-element tuple of 3-element np.array - the bounds
    #                   mu      sigma    a
              np.array([0,      2,       0]),  # lower bounds
              np.array([0, np.inf,       1]))  # upper bounds
    population : np.array of int - the indices of the neurons included
    alpha : float - the ridge parameter
    subspace : bool - whether to use a subspace or not

    Returns
    -------
    float - Stable score
    '''
    if 'subspace' in kwargs and kwargs['subspace']:
        raise ValueError("There is no subspace with dynamic ensembles")
    accuracy = vectorized_xval_CTD(X, y, subspace=False, dstraining=dstraining, **kwargs)
    score = get_CT_score(accuracy, bounds, dstraining=dstraining)
    return score


def vectorized_xval_subset(X, y, trialinds=None, **kwargs):
    '''Calling vectorized cross-validation of CTD with a subset of trials

    Used to get the ridge parameters from the training trials only.

    Parameters
    ----------
    X : np.array<trials * bins * neurons> - data
    y : np.array<trials> - targets
    trialinds : np.array of int - The indices of the included trials
    **kwargs : see arguments for vectorized_xval_CTD

    Returns
    -------
    np.array<bins * bins> - the CTD accuracy averaged across folds
      of the cross-validation
    '''
    if trialinds is not None:
        X = X[trialinds]
        y = y[trialinds]
    return vectorized_xval_CTD(X, y, **kwargs)


def get_ridge_param_CTD_vec(data, alpha_powers, subspace, perm_seed=None,
                            trialinds=None, client=None):
    '''Get the ridge parameter for CTD

    Parameters
    ----------
    data : a tuple of arguments - It contains the following in that order
      X : np.array<trials * bins * neurons> - data
      y : np.array<trials> - targets
      mask : np.array<bins> of bool - which bins to take to build the subspace
      population : np.array of int - the indices of the neurons included
    alpha_powers : np.array - the powers of ten of alpha values that will be
      tested
    subspace : bool - Whether to use a subspace for CTD
    perm_seed : int - A permutation seed to shuffle labels
    trialinds : np.array of int - The indices of the included trials
    client : dask client - To perform the parameter exploration in parallel

    Returns
    -------
    float or dask future (if parallel) - The ridge parameter that yields the
      best decoding accuracy
    '''
    X, y, delaymask, pop = data

    acc_alpha = []
    for ap in alpha_powers:
        alpha = 10.**ap
        if client:
            acc_alpha.append(client.submit(vectorized_xval_subset, X, y,
                                           population=pop, permseed=perm_seed, subspace=subspace,
                                           alpha=alpha, mask=delaymask, trialinds=trialinds))
        else:
            acc_alpha.append(vectorized_xval_subset(X, y, population=pop, permseed=perm_seed,
                                                    subspace=subspace, alpha=alpha, mask=delaymask,
                                                    trialinds=trialinds))

    def get_mean(acc):
        return np.mean(acc[delaymask][:, delaymask])

    acc_alpha = client.map(get_mean, acc_alpha)

    def best_alpha(acc_alpha):
        return 10. ** alpha_powers[np.argmax(acc_alpha)]
    if client:
        alpha = client.submit(best_alpha, acc_alpha)
    else:
        alpha = best_alpha(acc_alpha)
    return alpha


def train_test_CTD(data, params, stable, dynparams=None, client=None):
    '''Generic function to test stable or dynamic ensembles

    Parameters
    ----------
    data : 2-element tuple containing (or their equivalent dask future if parallelized)
      X : np.array<trials * bins * neurons*> - The firing rate activity for all
        bins and trials
      y : np.array<trials> - The identity of each trial
    params : 3-element tuple containing
      pop : np.array of int - the indices of the neurons included
      alpha : float - the ridge parameter
      subspace : bool - whether to use a subspace
    stable : bool - whether to test for a stable (True) or dynamic (False)
      ensemble
    dynparams : 2-element tuple containing (ignored if stable is True)
      bounds : a 2-element tuple of 3-element np.array - the bounds for
        gaussian fitting for the locality score, e.g.
        #                   mu      sigma    a
                  np.array([0,      2,       0]),  # lower bounds
                  np.array([0, np.inf,       1]))  # upper bounds
      dstraining : int - Every 'dstraining' bins will be selected for a down
        sampled training
    client : dask client - A dask client to perform the computations in parallel
    '''
    if client is not None:
        parallel = True
    Xfold, yfold = data
    pop, alpha, subspace = params
    if not stable:
        bounds, dstraining = dynparams
    if stable:
        if parallel:
            perf = client.submit(ensemble_mean_acc_vec, Xfold, yfold,
                                 population=pop, alpha=alpha, subspace=subspace)
        else:
            perf = ensemble_mean_acc_vec(Xfold, yfold,
                                         population=pop, alpha=alpha, subspace=subspace)
    else:
        if parallel:
            perf = client.submit(ensemble_dynamic_acc_vec, Xfold, yfold, bounds,
                                 population=pop, alpha=alpha, dstraining=dstraining)
        else:
            perf = ensemble_dynamic_acc_vec(Xfold, yfold, bounds,
                                            population=pop, alpha=alpha,
                                            dstraining=dstraining)
    return perf


###############################################################################
################################ NON VECTORIZED ###############################
###############################################################################

# Non vectorized versions of the above functions

def decoding_subspace_xval_ridge(X, y, trainind, testind, alpha=1,
                                 subspace=True, mask=None):
    '''Decoding with cross-validation in subspace

    Parameters
    ----------
    X : list of np.arrays [ntrials] np.array<nbins*nneurons>
    y : list of np.arrays [ntrials] np.array<nbins>
    trainind : np.array
      The indices of the training trials
    testind : np.array
      The indices of the testing trials

    Returns
    -------
    correct : np.array<nbins*nbins> of Boolean's
      True if the output is correct, False otherwise
    testout : np.array<nbins*nbins*ntesttrials>
      The output of the classifier for each pair of train and test bins
    '''
    sub_train_ratio = .5
    nbins = X.shape[1]
    labels = np.unique(y)
    testout = np.empty((len(testind), nbins))
    correct = np.empty(nbins)

    if mask is None:
        mask = range(nbins)

    ### Split subspace and training
    if subspace:
        nsubtrials = int(len(trainind)*sub_train_ratio)
        subind, trainind = trainind[:nsubtrials], trainind[nsubtrials:]
        ysub = y[subind]
        Xsub = X[:, mask][subind].mean(1)  # Averaging over time bins
        Xsub = np.stack([Xsub[ysub == label].mean(0) for label in labels])
        subspace = PCA()
        subspace.fit(Xsub)

    Xtrain, Xtest = X[trainind], X[testind]
    ytrain, ytest = y[trainind], y[testind]

    for ibin in range(nbins):
        if subspace:
            Xbintrain = subspace.transform(Xtrain[:, ibin])
        else:
            Xbintrain = Xtrain[:, ibin]
        model = OneVsOneClassifier(RidgeClassifier(alpha=alpha, solver='cholesky'))
        model.fit(Xbintrain, ytrain)

        if subspace:
            Xbintest = subspace.transform(Xtest[:, ibin])
        else:
            Xbintest = Xtest[:, ibin]
        out = model.predict(Xbintest)
        testout[:, ibin] = out
        correct[ibin] = np.mean(out == ytest)

    return correct, testout


def crosstemp_decoding_subspace_xval_ridge(X, y, indtrain, indtest,
                                           alpha=1, indsub=None, mask=None):
    '''Cross-temporal decoding with cross-validation in subspace

    Parameters
    ----------
    X : np.array<trials * bins * neurons> - data
    y : np.array<trials> - targets
    trainind : np.array - the indices of the training trials
    testind : np.array - the indices of the testing trials
    alpha : float - the L2 "ridge" regularization parameter
    subind : np.array - the indices of the trials used to define the subspace.
      They must be different from the training and testing indices.
    mask : np.array<nbins> of bool - a mask to select which bins are used to
      define the subspace. E.g.: np.array([False, False, True, True, False])
      here only bins 2 and 3 are used to define the subspace.

    Returns
    -------
    correct : np.array<bins * bins> of Boolean's
      True if the output is correct, False otherwise
    testout : np.array<bins * bins * test trials>
      The output of the classifier for each pair of train and test bins
    '''
    assert len(set(indtrain) & set(indtest)) == 0
    if indsub is not None:
        subspace = True
        assert len(set(indtrain) & set(indsub)) == 0
        assert len(set(indtest) & set(indsub)) == 0
    else:
        subspace = False

    nbins = X.shape[1]
    labels = np.unique(y)
    testout = np.empty((len(indtest), nbins, nbins))
    correct = np.empty((nbins, nbins))

    Xtrain, Xtest = X[indtrain], X[indtest]
    ytrain, ytest = y[indtrain], y[indtest]

    ### Split subspace and training if subspace indices are provided
    if subspace:
        if mask is None:
            mask = range(nbins)
        ysub = y[indsub]
        Xsub = X[:, mask][indsub].mean(1)  # Averaging over time bins
        Xsub = np.stack([Xsub[ysub == label].mean(0) for label in labels])
        subspace = PCA()
        subspace.fit(Xsub)

    ### A decoder is trained on each bin, and each decoder is tested on every bins
    for itrain in range(nbins):
        if subspace:
            Xbintrain = subspace.transform(Xtrain[:, itrain])
        else:
            Xbintrain = Xtrain[:, itrain]
        model = OneVsOneClassifier(RidgeClassifier(alpha=alpha, solver='cholesky'))
        model.fit(Xbintrain, ytrain)

        # The test data is reshaped to test all the bins in a single shot (much faster)
        Xtest_ = Xtest.reshape(Xtest.shape[0] * Xtest.shape[1], Xtest.shape[2])
        if subspace:
            Xtest_ = subspace.transform(Xtest_)
        preds = model.predict(Xtest_)
        # The output is reshaped to the original shape of the test data
        preds = preds.reshape(len(indtest), nbins)
        accs = (preds == ytest[:, None]).mean(0)
        testout[:, itrain, :] = preds
        correct[itrain, :] = accs

    return correct, testout


def job_CT(X, y, trainind, testind, population,
           perm_seed=None, **kwargs):
    poparray = np.array(population)
    newX = X[:, :, poparray]
    if perm_seed:
        np.random.seed(perm_seed)
        np.random.shuffle(newX)
    result = crosstemp_decoding_subspace_xval_ridge(newX, y, trainind, testind, **kwargs)[0]
    return result


def xval_job(data, client, pop,subspace=True, **kwargs):
    X, y, ntrials = data
    kfold = KFold(n_splits=5)
    twofold = KFold(n_splits=2)
    acc_fold = []
    for trainind, testind in kfold.split(range(ntrials)):
        if subspace:
            acc_sub_split = []
            for ridgeindind, subindind in twofold.split(trainind):
                ridgeind, subind = trainind[ridgeindind], trainind[subindind]
                acc_fold.append(job_CT(X, y, ridgeind, testind, pop,
                                       subind=subind, **kwargs))
            acc_fold.append(combine_xval_folds(acc_sub_split))
        else:
            acc_fold.append(job_CT(X, y, trainind, testind, pop, **kwargs))
    accuracy = combine_xval_folds(acc_fold)
    return accuracy


def submit_xval_jobs(data, client, pop,
                     subspace=True, alpha=1, **kwargs):
    Xfut, yfut, ntrials = data
    kfold = KFold(n_splits=5)
    twofold = KFold(n_splits=2)
    acc_fold = []
    for trainind, testind in kfold.split(range(ntrials)):
        if subspace:
            acc_sub_split = []
            for ridgeindind, subindind in twofold.split(trainind):
                ridgeind, subind = trainind[ridgeindind], trainind[subindind]
                # acc_fold.append(job_CT(Xfut.result(), yfut.result(), ridgeind, testind, pop,
                #                        subind=subind, alpha=alpha.result(), **kwargs))
                acc_sub_split.append(client.submit(job_CT, Xfut, yfut, ridgeind, testind, pop,
                                                   subind=subind, alpha=alpha, **kwargs))
            acc_fold.append(client.submit(combine_xval_folds, acc_sub_split))
        else:
            acc_fold.append(client.submit(job_CT, Xfut, yfut, trainind, testind, pop, **kwargs))
    accuracy = client.submit(combine_xval_folds, acc_fold)
    return accuracy


def ensemble_mean_acc(data, client, *args, **kwargs):
    '''Get stable score for a given ensemble

    Parameters
    ----------
    data : tuple (X, y, ntrials) - The data for decoding and the number of
      trials in a tuple
    client : the dask client
    **kwargs : keyword arguments, contains
      alpha : float - the L2 "ridge" regularization parameter
      subspace : bool - whether to use a subspace
      pop : np.array of int - the indices of the neurons included

    Returns
    -------
    Stable score
    '''
    accuracy = submit_xval_jobs(data, client, *args, **kwargs)
    return client.submit(get_acc_mean, accuracy)


def ensemble_dynamic_acc(data, client, *args, **kwargs):
    '''Get "locality" or dynamic score for a given ensemble

    Parameters
    ----------
    data : tuple (X, y, ntrials) - The data for decoding and the number of
      trials in a tuple
    client : the dask client
    **kwargs : keyword arguments, contains
      alpha : float - the L2 "ridge" regularization parameter
      subspace : bool - whether to use a subspace
      pop : np.array of int - the indices of the neurons included

    Returns
    -------
    Stable score
    '''
    accuracy = submit_xval_jobs(data, client, *args, **kwargs)
    return client.submit(get_CT_score, accuracy)


def CT_permutations(data, client, subspace, alpha, nperms):
    Xfut, yfut, ntrials, delaymask, pop = data
    kfold = KFold(n_splits=10)
    permseeds = np.random.randint(0, 10**9, nperms)
    perm_res = []
    for permseed in permseeds:
        acc_fold = []
        for trainind, testind in kfold.split(range(ntrials)):
            # acc_fold.append(job_CT(Xfut, yfut, trainind, testind, pop, alpha, subspace))
            acc_fold.append(client.submit(job_CT, Xfut, yfut, trainind, testind, pop, alpha=alpha, subspace=subspace,
                                          mask=delaymask, perm_seed=permseed))

        perm_res.append(client.submit(combine_xval_folds, acc_fold))
    return np.concatenate([perm[..., None] for perm in client.gather(perm_res)], 2)
