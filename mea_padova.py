############
## Import ##
############

import spikeinterface.full as si
import numpy as np
from pathlib import Path
import yaml
import numcodecs

################
## Parameters ##
################
print(">>> PARAMETERS <<<")

params_file = "params.yml"
print(f"Parameters file: {params_file}")

with open(params_file, 'r') as f:
    params = yaml.load(f, yaml.Loader)

###############
## Load data ##
###############
print(">>> LOAD DATA <<<")

# Define the path
base_folder = Path(params['data_folder'])
biocam_folder = base_folder / params['biocam_folder']
biocam_file = biocam_folder / params['biocam_file']

if not biocam_file.exists():
    raise Exception(f"File {biocam_file} does not exist!")

print(f"Raw file: {biocam_file}")

## Output folder
output_main_folder = Path(params['output_folder'])
output_main_folder.mkdir(exist_ok=True)
output_folder = output_main_folder / params['biocam_folder']
output_folder.mkdir(exist_ok=True)

print(f"Output folder: {output_folder}")

# Load biocam data
raw = si.read_biocam(biocam_file)

# Sampling frequency
fs = raw.get_sampling_frequency()
num_samples = raw.get_num_samples()

print(raw)

###################
## Preprocessing ##
###################
print(">>> PREPROCESSING <<<")

# Global job arguments
if 'job_kwargs' in params.keys():
    si.set_global_job_kwargs(**params['job_kwargs'])

# Parameters
params_preproc = params['preproc']

# Output folder
if params_preproc['save_compressed']:
    folder_extension = ".zarr"
else:
    folder_extension = ""

preprocessing_folder = output_folder / f"preprocessing{folder_extension}"


if params_preproc['load_existing_data'] and preprocessing_folder.exists():
    print(f"Loading saved preprocessed data from {preprocessing_folder}")
    if params_preproc['save_compressed']:
        raw_preproc = si.read_zarr(preprocessing_folder)
    else:
        raw_preproc = si.load_extractor(preprocessing_folder)
else:
    # Remove chan 0: not real signal
    if params_preproc['test_few_channels']:
        removed_channel_list = list(range(3500))
    else:
        removed_channel_list = [0]

    raw_good = raw.remove_channels(raw.channel_ids[removed_channel_list])

    print(f"Removed channels: {removed_channel_list}")

    # Filter data
    print(f"Filter data between {params_preproc['filter_lf']}Hz and {params_preproc['filter_hf']}Hz")
    raw_filtered = si.bandpass_filter(raw_good, freq_min=params_preproc['filter_lf'], freq_max=params_preproc['filter_hf'])

    # Common Median Reference
    print(f"Common Median Refence")
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

    print(f"Use samples from {start_frame} to {end_frame}")
    raw_sub = raw_cmr.frame_slice(start_frame=start_frame, end_frame=end_frame)

    raw_preproc = raw_sub

    # Save data after preprocessing
    if params_preproc['save_compressed']:
        print(f"Save compressed data")
        compressor = numcodecs.Blosc(cname="zstd", clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)
        raw_preproc.save(format="zarr", folder=preprocessing_folder, overwrite=True, compressor=compressor)
    else:
        print(f"Save uncompressed data")
        raw_preproc.save(folder=preprocessing_folder, overwrite=True)

#############
## Sorting ##
#############
print(">>> SORTING <<<")

params_sort = params['sorting']

sorter_name = params_sort['sorter']

print(f"Sorter: {sorter_name}")

# Check if sorter exists
if not sorter_name in si.available_sorters():
    raise Exception(f"Sorter {sorter_name} is not available! The list of available sorters is: {si.available_sorters()}")

# output_folder = biocam_folder / "output"
sorter_folder = output_folder / f"sorter_{sorter_name}"

if params_sort['load_existing_data'] and sorter_folder.exists():
    print(f"Loading saved sorted data from {sorter_folder}")
    sorting = si.read_sorter_folder(sorter_folder)
else:
    # if requested update custom params
    sorter_params = si.get_default_sorter_params(sorter_name_or_class=sorter_name)
    if 'sorter_params' in params_sort.keys():
        custom_params = params_sort['sorter_params']
        # ignore custom param keys not present in the default params 
        for k, v in custom_params.items():
            if k in sorter_params.keys():
                print(f"New param value: {k}: {v}")
                sorter_params[k] = v
    
    print("Run sorter")
    sorting = si.run_sorter(sorter_name, raw_preproc,
                            output_folder=sorter_folder,
                            remove_existing_folder=True,
                            verbose=True,
                            **sorter_params)

# ###############
# ## Waveforms ##
# ###############
# print(">>> WAVEFORMS <<<")

# params_wave = params['waveforms']

# waveforms_folder = output_folder / "waveforms_dense"
# waveforms_curated_folder = output_folder / "waveforms_curated"
# print("Extract waveforms")
# we = si.extract_waveforms(raw_preproc, sorting,
#                             folder=waveforms_folder,
#                             sparse=True,
#                             ms_before=1.5, 
#                             ms_after=2.5)

# # Compute noise level
# print("Compute noise level")
# si.compute_noise_levels(we)

# # Compute Principal Component
# print("Compute Principal Component")
# si.compute_principal_components(we, n_components=3, mode="by_channel_local", whiten=True)

# # Compute Spike Amplitude
# print("Compute Spike Amplitude")
# si.compute_spike_amplitudes(we)

# # Compute Unit and Spike locations
# print("Compute Unit locations")
# si.compute_unit_locations(we, method="monopolar_triangulation", load_if_exists=True)
# print("Compute Spike locations")
# si.compute_spike_locations(we, method="center_of_mass", load_if_exists=True)

# # Template metrics
# print("Compute template metrics")
# si.compute_template_metrics(we)

# # Quality metrics
# print("Quality metrics")
# qm_params = si.get_default_qm_params()
# qm_params['isi_violation']['isi_threshold_ms'] = 1.1
# print("'isi_threshold_ms': 1.1")
# metric_names = si.get_quality_metric_list()
# print(f"Quality metric list: {metric_names}")
# print("Compute quality metrics")
# qm = si.compute_quality_metrics(we, metric_names=metric_names, qm_params=qm_params)

# # Automatic curation based on quality metrics
# isi_violation_thresh = 0.5
# amp_cutoff_thresh = 0.1

# curation_query = f"amplitude_cutoff < {amp_cutoff_thresh} & isi_violation_ratio < {isi_violation_thresh}"

# print(f"Curation query: {curation_query}")
# keep_units = qm.query(curation_query)
# keep_units_id = keep_units.index.values

# sorting_auto = sorting.select_units(keep_units_id)
# print(f"Number of units before curation: {len(sorting.get_unit_ids)}")
# print(f"Number of units after curation: {len(sorting_auto.get_unit_ids)}")

# we_curated = we.select_units(keep_units_id, new_folder=waveforms_curated_folder)

######################
## SORTING ANALYZER ##
######################
print(">>> SORTING ANALYZER <<<")

analyzer_folder = output_folder / "analyzer.zarr"

analyzer = si.create_sorting_analyzer(sorting=sorting, recording=raw_preproc, format="memory")
print(analyzer)

print("Compute random spikes")
analyzer.compute(
    "random_spikes",
    method="uniform",
    max_spikes_per_unit=500
)

print("Compute waveforms")
analyzer.compute(
    "waveforms",
    ms_before=1.0,
    ms_after=2.0
)

print("Compute templates")
analyzer.compute(
    "templates",
    operators=["average", "median", "std"]
)
print("Noise level")
analyzer.compute(
    "noise_levels",
    recording=raw_preproc
)

print("Principal component")
analyzer.compute(
    "principal_components", 
    n_components=3, 
    mode="by_channel_local"
    )

# Compute Spike Amplitude
print("Compute Spike Amplitude")
analyzer.compute(
    "spike_amplitudes",
    peak_sign="neg",
    outputs="concatenated"
)

# Compute Unit and Spike locations
print("Compute Unit locations")
analyzer.compute(
    "unit_locations",
    method="monopolar_triangulation",
)

print("Compute Spike locations")
analyzer.compute(
    "spike_locations",
    method="center_of_mass"
)

# Template metrics
print("Compute template metrics")
analyzer.compute("template_metrics")

print("Saving analyzer")
analyzer.save_as(folder=analyzer_folder, format="zarr")

############
## EXPORT ##
############
print(">>> EXPORT <<<")


phy_folder = output_folder / f"phy_{sorter_name}"
report_folder = output_folder / f"report_{sorter_name}"

print("Export to phy")
si.export_to_phy(we, phy_folder)
print("Export report")
si.export_report(we_curated, output_folder=report_folder)


