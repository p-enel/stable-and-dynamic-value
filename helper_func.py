import pickle as pk
import numpy as np
from parameters import STABLE_ENS_PARAMS, BASIC_PARAMS, EVT_WINS
from generate_population_dataset import generate_dataset
from parameters import nouterfolds, ensfolder, unitfolder, resfolder
from matplotlib import pyplot as plt


def open_result_file(CTDparams, pp_ens, pp_test, dataseed):
    orig_fname = get_test_fname(CTDparams, pp_ens, pp_test, dataseed, perm=False)
    print(orig_fname)
    with open(resfolder/orig_fname, 'rb') as f:
        res_orig = pk.load(f)
    return res_orig


def get_acc_and_perms_one_seed(CTDparams, pp_ens, pp_test, dataseed):
    res_orig = open_result_file(CTDparams, pp_ens, pp_test, dataseed)

    acc_orig = res_orig['acc_test'][0]

    perm_fname = get_test_fname(CTDparams, pp_ens, pp_test, dataseed, perm=True)
    perm_fullpath = resfolder/perm_fname
    if perm_fullpath.exists():
        with open(perm_fullpath, 'rb') as f:
            res_perm = pk.load(f)
        acc_perm = res_perm['acc_test']
    else:
        print(f"File {perm_fullpath} not found")
        acc_perm = None
    return acc_orig, acc_perm


def get_acc_and_perms(CTDparams, pp_ens, pp_test, dataseeds):
    accs, perms = [], []
    for dataseed in dataseeds:
        acc, perm = get_acc_and_perms_one_seed(CTDparams, pp_ens, pp_test, dataseed)
        accs.append(acc)
        perms.append(perm)
    return accs, perms


def get_ens_filename(CTDparams, pp_ens, dataseed, ifold=None):
    '''Get the name of an ensemble results file

    Arguments:
    CTDparams - dict
      'monkey':   'both' | 'M' | 'N'
      'region':   'OFC' | 'ACC'
      'taskvar':  'value' | 'type'
      'subspace': True | False
      'neurons':  'all' | 'linear
      'normalization': True | False
      'stable':   True | False
    '''
    p = CTDparams
    filename = (f"removing_units_NOTtimenormed_CTdecoding_{p['region']}_ensembles_smooth{pp_ens['smooth']}"
                f"_winsize{pp_ens['smoothsize']}_binsize{pp_ens['binsize']}_align{pp_ens['align']}"
                f"_step{pp_ens['step']}_multi{pp_ens['multi']}_zscore{pp_ens['zscore']}_{p['monkey']}"
                f"_{p['region']}_{p['taskvar']}_subspace{p['subspace']}_seed{dataseed}")
    if ifold is not None:
        filename += f'_fold{ifold}'
    if not p['stable']:
        filename = "DYNAMIC_" + filename
    return filename + ".pk"


def get_test_fname(CTDparams, pp_ens, pp_test, dataseed, wayback=False, perm=False):
    '''Get the name of a CTD perfs and permutations

    Arguments:
    CTDparams - dict
      'monkey':   'both' | 'M' | 'N'
      'region':   'OFC' | 'ACC'
      'taskvar':  'value' | 'type'
      'subspace': True | False
      'ensemble': True | False
      'neurons':  'all' | 'linear
      'normalization': True | False
      'stable':   True | False
    '''
    p = CTDparams
    filename = (f"CTD_{p['monkey']}_{p['region']}_{p['taskvar']}_sub{p['subspace']}_ensbl{p['ensemble']}"
                f"_smens{pp_ens['smooth']}_winens{pp_ens['smoothsize']}_binens{pp_ens['binsize']}"
                f"_alingens{pp_ens['align']}_stepens{pp_ens['step']}_smtest{pp_test['smooth']}"
                f"_wintest{pp_test['smoothsize']}_bintest{pp_test['binsize']}_alingtest{pp_test['align']}"
                f"_steptest{pp_test['step']}_seed{dataseed}.pk")

    if perm:
        filename = "permutations_" + filename
    else:
        filename = "accuracy_" + filename

    if not p['stable']:
        filename = "DYNAMIC_" + filename

    if wayback:
        filename = "wayback_" + filename

    return filename


def get_ens_pp(stable):
    '''Get ensemble preprocessing parameters (pp)'''
    if stable:
        pp = dict(STABLE_ENS_PARAMS)
    else:
        pp = dict(BASIC_PARAMS)

    pp['evt_wins'] = EVT_WINS
    pp['events'] = list(EVT_WINS.keys())

    return pp


def get_test_pp():
    '''Get test preprocessing parameters (pp)'''
    pp = dict(BASIC_PARAMS)
    pp['evt_wins'] = EVT_WINS

    pp['events'] = list(EVT_WINS.keys())

    return pp


def get_X_y(dataseed, params_preproc, monkey, region, taskvar):
    unitfile = get_unit_path(params_preproc)
    X, y, delaymask, _ = generate_dataset(dataseed, unitfolder, unitfile, unitfolder)
    X, y = X[monkey, region, taskvar], y[monkey, region, taskvar]
    #### Let's keep only the delay activity for best ensemble searching ####
    first, last = np.where(np.diff(delaymask))[0]
    delaymask[last] = False
    return X, y, delaymask


def combine_ens_folds(CTDparams, stable, dataseed, erasefolds=True):
    '''Loading all intermediate fold results and saving them in a single file'''

    # Loading individual fold results
    params_preproc = get_ens_pp(stable)
    filename = get_ens_filename(CTDparams, params_preproc, dataseed)
    acc_test_xval = []
    ensbls_test_xval = []
    testind_xval = []
    for ifold in range(nouterfolds):
        foldname = get_ens_filename(CTDparams, params_preproc, dataseed, ifold=ifold)
        with open(ensfolder/foldname, 'rb') as f:
            ensperfs_tmp, ensbl_tmp, test_ind = pk.load(f)
        acc_test_xval.append(ensperfs_tmp)
        ensbls_test_xval.append(ensbl_tmp)
        testind_xval.append(test_ind)

    # Saving all folds in one file
    tosave = {'ensembles': ensbls_test_xval,
              'accuracies': acc_test_xval,
              'testinds': testind_xval,
              'preproc': params_preproc,
              'monkey': CTDparams['monkey'],
              'region': CTDparams['region'],
              'taskvar': CTDparams['taskvar'],
              'subspace': CTDparams['subspace'],
              'testinds': testind_xval,
              'dataseed': dataseed}
    with open(ensfolder/filename, 'wb') as f:
        pk.dump(tosave, f)

    if erasefolds:
        for ifold in range(nouterfolds):
            foldname = get_ens_filename(CTDparams, params_preproc, dataseed, ifold=ifold)
            fullpath = ensfolder / foldname
            fullpath.unlink()


def get_best_ensembles(CTDparams, params_preproc_ens, dataseed):
    filename = get_ens_filename(CTDparams, params_preproc_ens, dataseed)

    with open(ensfolder/filename, 'rb') as f:
        ens_res = pk.load(f)
    '''
    Content of the results file:
    {'ensembles', 'accuracies', 'labels', 'preproc', 'monkey',
    'region', 'taskvar', 'subspace', 'testinds'}
    '''
    # accuracies, ensembles, testinds = ens_res
    # bestens = [set(ensembles[:np.argmax(accuracies)])]

    ensembles = ens_res['ensembles']
    accuracies = ens_res['accuracies']
    # testinds = ens_res['testinds']

    bestens = [np.array(ens[:np.argmax(acc)]) for ens, acc in zip(ensembles, accuracies)]

    return ens_res, bestens


def get_unit_path(params_preproc):
    '''Returns the full path of a unit file'''
    unitfile = 'unit_dataset_align.{align}_binsize.{binsize}_smooth.{smooth}'\
        '_smoothsize.{smoothsize}_step.{step}.pk'
    unitfile = unitfile.format(**params_preproc)
    return unitfile


def ticks_labels(evt_wins, step, ticks_every):
    '''Returns ticks position and label for fragmented x axis

    Parameters
    ----------
    evt_wins : OrderedDict {'evt': (start, end)}
      The window boundary for each event represented on the x axis
    step : int
      The number of ms between every bin center
    ticks_every : int
      The distance in ms between each tick

    Returns
    -------
    ticks : np.array<nticks>
      The position of each ticks (in bins)
    labels : np.array<nticks>
      The label of each tick
    '''
    start, end = evt_wins
    base = np.arange(-start - ticks_every * (-start//ticks_every),
                     -start + end, ticks_every)
    ticks = base // step
    labels = base + start
    return ticks, labels


def format_fragmented_CT(evt_wins, evtxs, step, ticks_every, evt_names,
                         sep_color='white', evt_color='white', evttxt=True,
                         textsize=13, ax=None):
    '''Format a figure with a fragmented x axis

    Parameters
    ----------
    evt_wins : OrderedDict {'evt': (start, end)}
      The window boundary for each event represented on the x axis
    evtxs : dict {'evt': np.array<nbins>}
      The timing around the event of each bin
    step : int
      The number of ms between every bin center
    ticks_every : int
      The distance in ms between each tick
    evt_names : dict {'evt': 'evt_name'}
      The plain English name of each event (corresponding to each code name)
    sep_color : Matplotlib color
      The color of the separator between events
    ax : AxesSubplot
      The subplot in which the formatting must be done
    '''
    if ax:
        plt.sca(ax)

    evt_list = list(evt_wins.keys())
    xtickslab = np.concatenate([evtxs[evt] for evt in evt_list])
    eventpos = np.where(xtickslab == 0)[0]
    borderpos = np.cumsum(np.array([len(evtxs[evt]) for evt in evt_list]))

    [plt.axvline(x, color=sep_color, linewidth=2) for x in borderpos[:-1]]
    [plt.axhline(x, color=sep_color, linewidth=2) for x in borderpos[:-1]]

    ylims = plt.ylim()
    # xlims = plt.xlim()
    ytext = (ylims[1] - ylims[0]) * .01
    xtext = 0  # (xlims[1] - xlims[0]) * .01

    offset = 0
    xticks_ = []
    xlabels_ = []
    for ievt, (evt, evtpos) in enumerate(zip(evt_names.keys(), eventpos)):
        plt.axvline(evtpos, color=evt_color, linewidth=.5, linestyle='--')
        plt.axhline(evtpos, color=evt_color, linewidth=.5, linestyle='--')
        if evttxt:
            plt.text(evtpos+1, ytext, evt_names[evt], horizontalalignment='left', color=evt_color, size=textsize)
            plt.text(xtext, evtpos+1, evt_names[evt], verticalalignment='bottom', rotation=90, color=evt_color, size=textsize)
        ticks, labels = ticks_labels(evt_wins[evt], step, ticks_every)
        xticks_.append(ticks + offset)
        xlabels_.append(labels)
        offset += sum((evtxs[evt]>=evt_wins[evt][0]) & (evtxs[evt]<=evt_wins[evt][1]))
    xticks_ = np.concatenate(xticks_)
    xlabels_ = np.concatenate(xlabels_)

    plt.xticks(xticks_, xlabels_)
    plt.yticks(xticks_, xlabels_)
