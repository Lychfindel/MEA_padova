############
## Import ##
############

import spikeinterface.full as si
from spikeinterface.core import crea
import numpy as np
from pathlib import Path
import yaml
import os

#####################
## Load parameters ##
#####################

params_file = "MEA_padova/params.yml"
with open(params_file, 'r') as f:
    params = yaml.load(f, yaml.Loader)

###############
## Load data ##
###############

# Define the path
base_folder = Path(params['data_folder'])
biocam_folder = base_folder / params['biocam_folder']
biocam_file = biocam_folder / params['biocam_file']

# Load biocam data
raw = si.read_biocam(biocam_file)

# Sampling frequency
fs = raw.get_sampling_frequency()
num_samples = raw.get_num_samples()

###################
## Preprocessing ##
###################

# Remove chan 0: not real signal
raw_good = raw.remove_channels([0])

# Filter data
raw_filtered = si.bandpass_filter(raw_good, freq_min=params['filter_lf'], freq_max=params['filter_hf'])

# Reduce sample size
start_frame = params['start_s'] * fs
end_frame = params['end_s'] * fs
# if start frame negative the start is the beginning
if start_frame < 0:
    start_frame = 0
elif start_frame >= num_samples:
    raise Exception(f"Start ({params['start_s']}) is after the end of the recording ({raw.get_duration()})")

if end_frame > 0 and end_frame < start_frame:
    raise Exception({f"End ({params['end_s']}) is before start ({params['start_s']})"})

if end_frame < 0 or end_frame > num_samples:
    end_frame = num_samples

raw_sub = raw_filtered.frame_slice(start_frame=start_frame, end_frame=end_frame)

#############
## Sorting ##
#############

sorter_name = params['sorter']

# Check if sorter exists
if not sorter_name in si.available_sorters():
    raise Exception(f"Sorter {sorter_name} is not available! The list of available sorters is: \n\t{'\n\t'.join(si.available_sorters())}")

output_folder = biocam_folder / "output"
sorter_folder = output_folder / f"sorter_{sorter_name}"

# if requested update custom params
sorter_params = si.get_default_sorter_params(sorter_name_or_class=sorter_name)
if 'sorter_params' in params.keys():
    custom_params = params['sorter_params']
    # ignore custom param keys not present in the default params 
    for k, v in custom_params.items():
        if k in sorter_params.keys():
            sorter_params[k] = v

sorting = si.run_sorter(sorter_name, raw_sub,
                        output_folder=sorter_folder,
                        remove_existing_folder=True,
                        verbose=True,
                        **sorter_params)

###############
## Waveforms ##
###############

waveforms_folder = output_folder / "waveforms_dense"
we = si.extract_waveforms(raw_sub, sorting,
                            folder=waveforms_folder,
                            spase=True,
                            ms_before=1.5, 
                            ms_after=2.5)

# Compute noise level
si.compute_noise_levels(we)

# Compute Principal Component
si.compute_principal_components(we, n_components=3, mode="by_channel_local", whiten=True)

# Compute Spike Amplitude
si.compute_spike_amplitudes(we)