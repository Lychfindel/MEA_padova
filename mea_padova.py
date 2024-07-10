############
## Import ##
############

import spikeinterface.full as si
import numpy as np
from pathlib import Path
import yaml
import numcodecs

#####################
## Load parameters ##
#####################

params_file = "params.yml"
with open(params_file, 'r') as f:
    params = yaml.load(f, yaml.Loader)

###############
## Load data ##
###############

# Define the path
base_folder = Path(params['data_folder'])
biocam_folder = base_folder / params['biocam_folder']
biocam_file = biocam_folder / params['biocam_file']

if not biocam_file.exists():
    raise Exception(f"File {biocam_file} does not exist!")

## Output folder
output_main_folder = Path(params['output_folder'])
output_main_folder.mkdir(exist_ok=True)
output_folder = output_main_folder / params['biocam_folder']
output_folder.mkdir(exist_ok=True)

# Load biocam data
raw = si.read_biocam(biocam_file)

# Sampling frequency
fs = raw.get_sampling_frequency()
num_samples = raw.get_num_samples()

###################
## Preprocessing ##
###################

# Global job arguments
if 'job_kwargs' in params.keys():
    si.set_global_job_kwargs(**params['job_kwargs'])

# Output folder
preprocessing_folder = output_folder / "preprocessing.zarr"

# Parameters
params_preproc = params['preproc']

if params_preproc['load_existing_data'] and preprocessing_folder.exists():
    print(f"Loading saved preprocessed data from {preprocessing_folder}")
    raw_preproc = si.read_zarr(preprocessing_folder)
else:
    # Remove chan 0: not real signal
    if params_preproc['test_few_channels']:
        removed_channel_list = list(range(3500))
    else:
        removed_channel_list = [0]

    raw_good = raw.remove_channels(raw.channel_ids[removed_channel_list])

    # Filter data
    raw_filtered = si.bandpass_filter(raw_good, freq_min=params_preproc['filter_lf'], freq_max=params_preproc['filter_hf'])

    # Common Median Reference
    raw_cmr = si.common_reference(raw_filtered, reference='global', operator='median')
    
    # Reduce sample size
    start_frame = params_preproc['start_s'] * fs
    end_frame = params_preproc['end_s'] * fs
    # if start frame negative the start is the beginning
    if start_frame < 0:
        start_frame = 0
    elif start_frame >= num_samples:
        raise Exception(f"Start ({params_preproc['start_s']}) is after the end of the recording ({raw.get_duration()})")

    if end_frame > 0 and end_frame < start_frame:
        raise Exception({f"End ({params_preproc['end_s']}) is before start ({params_preproc['start_s']})"})

    if end_frame < 0 or end_frame > num_samples:
        end_frame = num_samples

    raw_sub = raw_cmr.frame_slice(start_frame=start_frame, end_frame=end_frame)

    raw_preproc = raw_sub

    # Save data after preprocessing
    compressor = numcodecs.Blosc(cname="zstd", clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)
    raw_preproc.save(format="zarr", folder=preprocessing_folder, overwrite=True, compressor=compressor)

#############
## Sorting ##
#############

params_sort = params['sorting']

sorter_name = params_sort['sorter']

# Check if sorter exists
if not sorter_name in si.available_sorters():
    raise Exception(f"Sorter {sorter_name} is not available! The list of available sorters is: {si.available_sorters()}")

# output_folder = biocam_folder / "output"
sorter_folder = output_folder / f"sorter_{sorter_name}"

if params_sort['load_existing_data'] and sorter_folder.exists():
    print(f"Loading saved sorted data from {sorter_folder}")
    sorting = si.load_extractor(preprocessing_folder)
else:
    # if requested update custom params
    sorter_params = si.get_default_sorter_params(sorter_name_or_class=sorter_name)
    if 'sorter_params' in params_sort.keys():
        custom_params = params_sort['sorter_params']
        # ignore custom param keys not present in the default params 
        for k, v in custom_params.items():
            if k in sorter_params.keys():
                sorter_params[k] = v

    sorting = si.run_sorter(sorter_name, raw_preproc,
                            output_folder=sorter_folder,
                            remove_existing_folder=True,
                            verbose=True,
                            **sorter_params)

###############
## Waveforms ##
###############

params_wave = params['waveforms']

waveforms_folder = output_folder / "waveforms_dense"
waveforms_curated_folder = output_folder / "waveforms_curated"
we = si.extract_waveforms(raw_preproc, sorting,
                            folder=waveforms_folder,
                            sparse=True,
                            ms_before=1.5, 
                            ms_after=2.5)

# Compute noise level
si.compute_noise_levels(we)

# Compute Principal Component
si.compute_principal_components(we, n_components=3, mode="by_channel_local", whiten=True)

# Compute Spike Amplitude
si.compute_spike_amplitudes(we)

# Compute Unit and Spike locations
si.compute_unit_locations(we, method="monopolar_triangulation", load_if_exists=True)
si.compute_spike_locations(we, method="center_of_mass", load_if_exists=True)

# Template metrics
si.compute_template_metrics(we)

# Quality metrics
qm_params = si.get_default_qm_params()
qm_params['isi_violation']['isi_threshold_ms'] = 1.1
metric_names = si.get_quality_metric_list()
qm = si.compute_quality_metrics(we, metric_names=metric_names, qm_params=qm_params)

# Automatic curation based on quality metrics
isi_violation_thresh = 0.5
amp_cutoff_thresh = 0.1

curation_query = f"amplitude_cutoff < {amp_cutoff_thresh} & isi_violation_ratio < {isi_violation_thresh}"

keep_units = qm.query(curation_query)
keep_units_id = keep_units.index.values

sorting_auto = sorting.select_units(keep_units_id)
print(f"Number of units before curation: {len(sorting.get_unit_ids)}")
print(f"Number of units after curation: {len(sorting_auto.get_unit_ids)}")

we_curated = we.select_units(keep_units_id, new_folder=waveforms_curated_folder)