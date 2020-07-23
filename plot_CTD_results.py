import pylab as pl
import numpy as np

from helper_func import (get_acc_and_perms, get_ens_pp,
                         get_test_pp, format_fragmented_CT, get_unit_path)
from generate_population_dataset import generate_dataset
from parameters import unitfolder, EVT_WINS, EVT_NAMES, dataseeds


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
CTDparams = {'monkey': 'both', # 'both' | 'M' | 'N'
             'region': 'OFC', # 'OFC' | 'ACC'
             'taskvar': 'value', # 'value' | 'type'
             'subspace': True, # True | False
             'ensemble': True, # True | False
             'stable': True} # True | False

plot_one_cond(CTDparams, dataseeds)
# The above function load the results for the given condition for each data set
# (data seed) and average the accuracy, computes p-values for individual data
# sets and then calculate the significance by aggregating the p-values of the
# different data sets with an averaging method appropriate to our case where
# the data sets are not independent.
