from pathlib import Path
import numpy as np
import pickle as pk
from itertools import chain, product
from collections import OrderedDict

from structure import Struct

MONKEYS = ['M', 'N']
REGIONS = ['OFC', 'ACC']
TASKVARS = ['value', 'type']
SUBSPACES = [True, False]
EVT_WINS = OrderedDict((('cues ON', (-500, 1500)),
                        ('response cue', (-500, 500)),
                        ('rwd', (-400, 400))))


def pp_from_filename(filename):
    '''Get the preprocessing parameters from a unit data set filename

    Arguments:

    filename - str or Path: name or full path of unit data set file
    '''
    fnamestr = filename if isinstance(filename, str) else filename.name
    params = [paramstr.split('.') for paramstr in fnamestr.split('_')[2:]]
    preproc_params = {'align': params[0][1],
                      'binsize': int(params[1][1]),
                      'smooth': params[2][1],
                      'smoothsize': int(params[3][1]),
                      'step': int(params[4][1])}
    return preproc_params


def get_dataset_fname(dataseed, pp):
    '''Generate the file name of a population data set given data seed and preprocessing parameters

    Arguments:

    dataseed - int: the seed of the data set that will be included in the file name
    pp - dict: the pre-processing parameters of the data set'''
    fname = "population_dataset_align.{align}_binsize.{binsize}_smooth.{smooth}"
    fname += "_smoothsize.{smoothsize}_step.{step}_seed.%d.pk" % dataseed
    fname = fname.format(**pp)
    return fname


def generate_dataset(dataseed, unit_folder, unit_file, save_folder=None):
    '''Generate a pseudo-population by combining data from monkeys and sessions

    Arguments:

    dataseed - int: the seed for pseudo-random selection of the trials to be
      part of the data set
    unit_file - str: the path to the file containing the unit data set
    save_folder - str or Path: optional, a folder to save the generated data
      set. After being saved once, if the same folder is specified, it will be
      loaded instead of being generated.

    Returns:

    X - Structure: A structure that contains the pseudo-population firing rate
      data. The structure contains 3 levels:
        - monkey: which can take values 'M' or 'N' for individual monkey data,
          or 'both' for the data of both monkeys combined
        - region: which can take value 'OFC' or 'ACC'
        - task variable: which can take value 'value' or 'type' for data sets
          targeted to decoding these variables
      The elements of the structure are numpy arrays of the shape:
        trials x bins x neurons
      Example:
        X['N', 'ACC', 'value'] contains a matrix of the pseudo-population
        firing rate of monkey N for region ACC meant to decode value
    y - Structure: A structure of numpy vectors with the same map as 'X' that
      contains the ground truth of the related variable for each trial.
      Example:
        y['N', 'ACC', 'value'] contains the value of each trials of monkey N
        for ACC population.
    delaymask - numpy vector of booleans: A boolean mask for the time bin
      dimension to select time bins that are part of the delay activity
    bins - numpy vector of ints: The time of each bin of the firing rate data
      in the structure X, with events ordered this way:
      'cues ON' -> 'response cue' -> 'rwd'
    '''
    events = list(EVT_WINS.keys())

    pp = pp_from_filename(unit_file)

    if save_folder is not None:
        dataset_fname = get_dataset_fname(dataseed, pp)
        dataset_fullpath = Path(save_folder)/dataset_fname

        if dataset_fullpath.exists():
            print("Data set already generated, loading...")
            with open(dataset_fullpath, 'rb') as f:
                X, y, delaymask, bins = pk.load(f)
            return X, y, delaymask, bins

    with open(Path(unit_folder)/unit_file, 'rb') as f:
        data = pk.load(f)

    evtxs = data['M']['OFC'][0]['bins']

    #### Format the data for decoding
    #################################
    keymap = [MONKEYS, REGIONS, TASKVARS]
    act = Struct.new_empty(keymap)
    minntrials = Struct.new_empty(keymap)

    for monkey, region in product(MONKEYS, REGIONS):
        act[monkey, region, 'value'] = [[] for _ in range(4)]
        act[monkey, region, 'type'] = [[], []]
        minntrials[monkey, region, 'value'] = [[] for _ in range(4)]
        minntrials[monkey, region, 'type'] = [[], []]

        datamr = data[monkey][region]

        ## Select bins that are within the window of interest for each event
        ## then concatenate the activity of the different events in a single tensor
        catepochs = []
        for sessdata in datamr:
            if sessdata['fr'] is not None:
                cattmp = []
                for evt in events:
                    included_bins = (evtxs[evt] >= EVT_WINS[evt][0]) & (evtxs[evt] <= EVT_WINS[evt][1])
                    cattmp.append(sessdata['fr'][evt][included_bins])
                catepochs.append(np.concatenate(cattmp))
            else:
                catepochs.append(None)

        ## Separate trials by value and type
        for sessfr, sessdata in zip(catepochs, datamr):
            if sessfr is not None:
                if sessdata['fr'] is not None:
                    sessvars = sessdata['vars']
                    for val in range(1, 5):
                        trialbool = (sessvars.value == val)
                        act[monkey, region, 'value'][val-1].append(sessfr[:, :, trialbool])
                    for itype, type_ in enumerate(['juice', 'bar']):
                        trialbool = (sessvars.type == type_)
                        act[monkey, region, 'type'][itype].append(sessfr[:, :, trialbool])

        ## Get the minimum number of trials across all sessions for each value/type
        minntrials[monkey, region, 'value'] = [np.nanmin([sessfr.shape[2] for sessfr in valdata])
                                               for valdata in act[monkey, region, 'value']]
        minntrials[monkey, region, 'type'] = [np.nanmin([sessfr.shape[2] for sessfr in typedata])
                                              for typedata in act[monkey, region, 'type']]

    ## Get the minimum number of trials for pooled data across monkeys
    minntrials.move_level_(0, 2)
    mintogether = minntrials.apply(lambda x: [min(valmin) for valmin in zip(*x.values())], depth=2)
    mintogether = Struct.from_nested_dict({'both': mintogether.ndict}, n_layers=3)
    minntrials.move_level_(2, 0)
    minntrials = minntrials.combine(mintogether)

    # extra trials are discarded after trials are shuffled
    np.random.seed(dataseed)

    catactboth = Struct.empty_like(act, values=list)
    # taskvar, monkey, region = next(product(TASKVARS, MONKEYS, REGIONS))
    for taskvar, monkey, region in product(TASKVARS, MONKEYS, REGIONS):
        keymap = [monkey, region, taskvar]
        minns = minntrials['both', region, taskvar]
        # minn, acttmp = next(zip(minns, act[keymap]))
        for minn, acttmp in zip(minns, act[keymap]):
            tocat = []
            for sessdata in acttmp:
                ntrials = sessdata.shape[2]
                trialind = np.arange(ntrials)
                np.random.shuffle(trialind)
                tmp = sessdata[:, :, trialind]
                tocat.append(tmp[:, :, :minn])
            catactboth[keymap].append(np.concatenate(tocat, 1))

    catact = Struct.empty_like(act, values=list)
    for taskvar, monkey, region in product(TASKVARS, MONKEYS, REGIONS):
        keymap = [monkey, region, taskvar]
        minns = minntrials[keymap]
        for minn, acttmp in zip(minns, act[keymap]):
            tocat = []
            for sessdata in acttmp:
                ntrials = sessdata.shape[2]
                trialind = np.arange(ntrials)
                np.random.shuffle(trialind)
                tmp = sessdata[:, :, trialind]
                tocat.append(tmp[:, :, :minn])
            catact[keymap].append(np.concatenate(tocat, 1))

    catactboth.move_level_(0, 2)

    def cat_monkeys(x):
        '''x: {monkey}[4 (values)] np.array<nbins*nneurons*ntrials>'''
        return [np.concatenate([x['M'][ival], x['N'][ival]], axis=1) for ival in range(len(x['M']))]

    catactboth.apply_agg_(cat_monkeys, depth=2)
    catactboth = Struct.from_nested_dict({'both': catactboth.ndict}, n_layers=3)
    catact = catact.combine(catactboth)

    #### Moving data from arrays to a list ####
    def get_actvallist(x):
        tmp = [[(trial, ival) for trial in np.moveaxis(x[ival], 2, 0)] for ival in range(len(x))]
        return list(zip(*chain(*zip(*tmp))))

    actvallist = catact.apply(get_actvallist)
    X, y = actvallist.apply(lambda x: x[0]), actvallist.apply(lambda x: x[1])
    X.apply_(np.stack)
    y.apply_(np.array)

    del(catact, act)

    #### Defining a boolean mask to get only the bins between cue ON and rwd
    ########################################################################
    cuesON_bins_mask = (evtxs['cues ON'] >= EVT_WINS['cues ON'][0]) & (evtxs['cues ON'] <= EVT_WINS['cues ON'][1])
    cuesON_bins = evtxs['cues ON'][cuesON_bins_mask]
    resp_bins_mask = (evtxs['response cue'] >= EVT_WINS['response cue'][0]) &\
        (evtxs['response cue'] <= EVT_WINS['response cue'][1])
    resp_bins = evtxs['response cue'][resp_bins_mask]
    rwd_bins_mask = (evtxs['rwd'] >= EVT_WINS['rwd'][0]) & (evtxs['rwd'] <= EVT_WINS['rwd'][1])
    rwd_bins = evtxs['rwd'][rwd_bins_mask]
    delaymask = np.concatenate((cuesON_bins >= 0, np.ones(resp_bins.shape, dtype=bool), rwd_bins <= 0))

    bins = {}
    for evt, (start, end) in EVT_WINS.items():
        xs = evtxs[evt]
        bins[evt] = xs[(xs >= start) & (xs <= end)]

    if save_folder is not None:
        with open(dataset_fullpath, 'wb') as f:
            pk.dump((X, y, delaymask, bins), f)
        print(f'data set created and saved in {unit_folder}')

    return X, y, delaymask, bins


# The following is an example. Replace the right hand side of the first three
# statements to get a specific data set
if __name__ == '__main__':
    # Data seeds used to generate the pseudo population data for decoding are
    # listed below:
    # dataseeds = [634564236, 9453241, 70010207, 43661999, 60410205]
    dataseed = 634564236
    # The following folder path must contain the unit data set file specified
    # below
    unit_folder = Path("/home/john/datasets")
    # The following statement specifies which unit data set (with which
    # preprocessing parameters) is to be used to generate the population data
    # set
    unit_file = "unit_dataset_align.center_binsize.100_smooth.gaussian_smoothsize.100_step.25.pk"
    # The last argument of the function allows you to save the data set in a
    # specified folder, or to load an already generated population data set if
    # it already exists in this folder. In this example the population data set
    # is saved in the same folder as the unit data set.
    X, y, delaymask, bins = generate_dataset(dataseed, unit_folder, unit_file,
                                             save_folder=unit_folder)

