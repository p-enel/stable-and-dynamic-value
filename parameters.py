from pathlib import Path
import numpy as np
from collections import OrderedDict

###############################################################################
# FOLDERS

# Where the pre-processed firing rate data can be found
unitfolder = Path('/home/pierre/datasets')
# Where the ensemble results will be saved
ensfolder = Path('/home/pierre/Documents/Analyzes/VBDM other-analyzes/Decoding/Results/ensembles')
# Where the decoding results and permutations will be saved
resfolder = Path('/home/pierre/Documents/Analyzes/VBDM other-analyzes/Decoding/Results/accuracies/CTD')

###############################################################################
# ANALYZES PARAMETERS

multi = True
zscore = True

STABLE_ENS_PARAMS = {'smooth': 'square',
                     'smoothsize': 1,
                     'binsize': 200,
                     'step': 200,
                     'align': 'end'}

BASIC_PARAMS = {'smooth': 'gaussian',
                'smoothsize': 100,
                'binsize': 100,
                'step': 50,
                'align': 'center'}

STABLE_ENS_PARAMS['multi'] = multi
STABLE_ENS_PARAMS['zscore'] = zscore

BASIC_PARAMS['multi'] = multi
BASIC_PARAMS['zscore'] = zscore

EVT_WINS = OrderedDict((('cues ON', (-500, 1500)),
                        ('response cue', (-500, 500)),
                        ('rwd', (-400, 400))))

EVT_NAMES = OrderedDict((['cues ON', 'cue'],
                         ['response cue', 'response cue'],
                         ['rwd', 'reward']))

###############################################################################
# ANALYZES PARAMETERS

# Powers of ten of alpha explored
alpha_powers = np.arange(-5, 5.1, .5)

# Data seeds to generate pseudo-population data sets
dataseeds = [634564236, 9453241, 70010207, 43661999, 60410205]

# The number of cross-validation folds for each ensemble search
nouterfolds = 5

# Dynamic ensemble gaussian fit: bounds for function gaussian_func
# These bounds ensure that the gaussian fit doesn't take an unexpected shape (e.g. upside down)
#                   mu      sigma    a
bounds = (np.array([0,      2,       0]),  # lower bounds
          np.array([0, np.inf,       1]))  # upper bounds

# Down sampling of training bins for dynamic ensembles to speed things up:
# the CTD is performed on one bin every 'dstraining' bins. The value 5 is a
# good compromise between precision and speed
dstraining = 5

dynparams = bounds, dstraining

###############################################################################
# PARALLELIZATION

# If the above parameter is set to 'distributed', DISTCLUSTER must be the IP
# address or network name of a dask cluster and its TCP port.
DISTCLUSTER = '192.168.0.10:8786'

# If 'cluster' is set to 'local', here you can specify how many cores will be
# used to explore ensembles on the machine that runs the script
NLOCALWORKERS = 4
