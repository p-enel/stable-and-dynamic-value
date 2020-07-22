import pickle as pk
import pylab as pl
import numpy as np

from helper_func import (get_test_fname, get_acc_and_perms, get_ens_pp,
                         get_test_pp, format_fragmented_CT, get_unit_path)
from generate_population_dataset import generate_dataset
from parameters import resfolder, unitfolder, EVT_WINS, EVT_NAMES

###############################################################################
### FUNCTIONS
###############################################################################


def get_acc_and_sig(monkey, region, taskvar, subspace, ensemble, stable,
                    pp_ens, pp_test, dataseed):
    orig_fname = get_test_fname(monkey, region, taskvar, subspace, ensemble,
                                stable, pp_ens, pp_test, dataseed, perm=False)
    with open(resfolder/orig_fname, 'rb') as f:
        res_orig = pk.load(f)

    acc_orig = res_orig['acc_test'][0]

    perm_fname = get_test_fname(monkey, region, taskvar, subspace, ensemble,
                                stable, pp_ens, pp_test, dataseed, perm=True)
    if (resfolder/perm_fname).exists():
        with open(resfolder/perm_fname, 'rb') as f:
            res_perm = pk.load(f)
        acc_perm = np.stack(res_perm['acc_test'])
        pvals = (acc_perm > acc_orig).mean(0)
        sig = pvals < 0.01
    else:
        sig = None

    return acc_orig, sig


def plot_CTD(acc, CTDparams=None, sig=None, vmin=.15, vmax=.93, cbar=True,
             sigmask=True):
    pl.imshow(acc.T, label=CTDparams['region'], origin='lower', vmin=vmin,
              vmax=vmax, cmap='inferno')
    if cbar:
        pl.colorbar()
    if CTDparams is not None:
        pl.title('{monkey}-{region}-{taskvar}-stable:{stable}\n'.format(**CTDparams)+\
                 'sub:{subspace} - ensemble:{ensemble}'.format(**CTDparams))
    pl.xlabel('time from event (training)')
    pl.ylabel('time from event (testing)')
    if sig is not None:
        if sigmask:
            palette = pl.get_cmap('gray')
            palette.set_under('w', 0.0)
            palette.set_over('gray', 1)
            pl.imshow(sig.T < .01, origin='lower', cmap=palette, vmin=.4, vmax=.5)
        else:
            pl.contour(sig.T, linewidths=.5, levels=0, colors='white')


def plot_one_CTD(CTDparams, pp_ens, pp_test, dataseeds, vmin=.15, vmax=.93):
    accs, perms = get_acc_and_perms(CTDparams, pp_ens, pp_test, dataseeds)
    acc = np.stack(accs).mean(0)
    if perms[0] is not None:
        perms = [np.stack(acc_one_perm).mean(0) for acc_one_perm in zip(*perms)]
        sig = np.mean(perms > acc, 0) < .01
    else:
        sig = None

    plot_CTD(acc, CTDparams, sig=sig, vmin=vmin, vmax=vmax)


def get_one_ds_pval(acc, perms):
    if isinstance(perms, list):
        perms = np.stack(perms)
    assert len(perms.shape) == 3
    assert acc.shape == perms[0].shape
    return (np.sum(acc < perms, 0) + 1) / (len(perms) + 1)


def plot_one_cond(CTDparams, dataseeds):
    '''Plot the CTD average accuracy across data sets of one condition

    Parameters
    ----------
    CTDparams : dict - the parameters of the cross-temporal decoding, it
      includes the following keys:
      - 'monkey': 'both' | 'M' | 'N'
      - 'region': 'OFC' | 'ACC'
      - 'taskvar': 'value' | 'type'
      - 'subspace': True | False
      - 'ensemble': True | False
      - 'stable': True | False
    dataseeds : list of int - the seeds of the data sets
    '''
    pp_ens = get_ens_pp(CTDparams['stable'])
    pp_test = get_test_pp()

    accs, allperms = get_acc_and_perms(CTDparams, pp_ens, pp_test, dataseeds)

    acc = np.stack(accs).mean(0)
    if allperms[0] is not None:
        allpvals = [get_one_ds_pval(acc, perms) for acc, perms in zip(accs, allperms)]
        pvals = 4 * np.stack(allpvals).mean(0)
        sig = pvals < .01
    else:
        sig = None

    if CTDparams['taskvar'] == 'value':
        vmin, vmax = .25, .93
    if CTDparams['taskvar'] == 'type':
        vmin, vmax = .5, .9

    pl.figure()
    plot_CTD(acc, CTDparams, sig=sig, vmin=vmin, vmax=vmax)

    step = 50
    ticks_every = 500
    unitfile = get_unit_path(pp_test)
    X, y, delaymask, bins = generate_dataset(dataseeds[0], unitfolder, unitfile)
    bins['cues ON'] = bins['cues ON'][bins['cues ON'] >= -500]
    evts = dict(EVT_NAMES)
    evts['response cue'] = 'resp.\ncue'
    evts['rwd'] = 'rwd'
    format_fragmented_CT(EVT_WINS, bins, step, ticks_every, evts,
                         sep_color='white', evt_color='white', evttxt=True,
                         textsize=13, ax=None)
    pl.show()


###############################################################################
##### Plot one condition CTD results (all seeds)
dataseeds = [634564236, 9453241, 70010207, 43661999, 60410205]
CTDparams = {'monkey': 'both', # 'both' | 'M' | 'N'
             'region': 'OFC', # 'OFC' | 'ACC'
             'taskvar': 'value', # 'value' | 'type'
             'subspace': False, # True | False
             'ensemble': True, # True | False
             'stable': False} # True | False

plot_one_cond(CTDparams, dataseeds)
