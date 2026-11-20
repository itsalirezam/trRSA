import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json, os, sys, time, re
from functions import *
import yaml
import argparse, importlib

print(f'\n\nStarted on {time.ctime()} Local Time')

# args

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
parser.add_argument('subject', type=int)
args = parser.parse_args()

# subject and config

subject = args.subject; print(f'subject: {subject}')
config_path = args.config_path



with open(config_path, "r") as f:
    config = yaml.safe_load(f)


# neural data

frequency_band = config['frequency_band']
region = config['region']
neural_data_folder_path = config['neural_data_folder_path']
simulation = config['simulation'] # boolean

if simulation:
    selected_neural_data = np.load(config['simulation_neural_data_path'])
else:
    selected_neural_data = np.load(f'{neural_data_folder_path}{region}-{frequency_band}/sub{subject}-{frequency_band}-{region}.npy')

neural_rdm_metric = config['neural_rdm_metric']

selected_neural_data.shape

# models
# shape (n_events, features)

selected_models = []
selected_models_labels = []
model_rdm_metrics = []

for model_config in config['models']:

    model = np.load(model_config['path'])

    selected_models.append(model)
    selected_models_labels.append(model_config['name'])
    model_rdm_metrics.append(model_config['rdm_metric'])

# other parameters

n_subsamples = config['n_subsamples']
subsampling_iterations = config['subsampling_iterations']
subsample_duration_sec = config['subsample_duration_sec']
resolution = config['resolution']

double_precision = config['double_precision']

# derived parameters

tps = selected_neural_data.shape[1]
subsample_tps = int(subsample_duration_sec * resolution) # subsample size in tps - also number of RDMs calculated for each subsample


# candidate wods
candidate_events_path = Path(config['candidate_events_path'])
candidate_events = np.load(candidate_events_path) # word_ids/indices

# get corresponding start times on neural time series

# select event candidates corresponding tps in the neural time series based on candidate_events
transcript_path = config['transcript_path']
df = df = pd.read_csv(transcript_path)

df_with_tps = df.copy()
df_with_tps['start'] = np.floor(df['start'] * resolution).astype(int)
df_with_tps['end'] = np.floor(df['end'] * resolution).astype(int)
df_with_tps

word_start_tps = df_with_tps['start'].to_numpy()


# run

time_series_subsample_idx, event_idx = subsampling_trRSA(tps, subsample_tps, n_subsamples, subsampling_iterations, selected_event_indices=candidate_events, event_start_tps=word_start_tps, mask=None, random_state=None)
print("\u2713 subsample indices")
subsampled_neural_data = apply_subsampling_indices(selected_neural_data, time_series_subsample_idx) # shape (iterations, n_subsamples, features, subsample_tps)
neural_rdm_series = compute_rdm_series(subsampled_neural_data, neural_rdm_metric) # shape (iterations, subsample_tps, rdm_length)
model_rdms_list = [compute_model_rdm(m, event_idx, metric=model_rdm_metrics[i]) for i, m in enumerate(selected_models)]
print("\u2713 rdms")
rdm_correlations_mean_list = [np.mean(batched_corr(neural_rdm_series, model_rdm), axis=0) for model_rdm in model_rdms_list]
print("\u2713 corr rdms")

# save lags
condition_name = candidate_events_path.stem
save_path = Path('_results') / f'{condition_name}'
file_name = (
    f'sub{subject} - {condition_name}'
)

np.save(get_unique_path(save_path, file_name + '.npy'), np.asarray(rdm_correlations_mean_list))

# save lables
with open(save_path / 'selected_model_labels.json', 'w') as f:
    json.dump(selected_models_labels, f)

# plot

half = subsample_tps // 2
lags = np.arange(subsample_tps) - half     # index i -> offset i - half
lags_in_seconds = lags / resolution
n_models = len(selected_models_labels)

fig, axs = plt.subplots(n_models, 1, figsize=(8, 3 * n_models), squeeze=False)

for i in range(n_models):
    model_name = selected_models_labels[i]
    curve = np.asarray(rdm_correlations_mean_list[i])

    ax_lag = axs[i, 0]
    ax_lag.plot(lags_in_seconds, curve)

    ax_lag.axvline(0, color='gray', linestyle='--')
    ax_lag.axhline(0, color='gray', linestyle='--')
    ax_lag.set_xlabel('Time [s]', fontsize=7)
    ax_lag.set_ylabel('Correlation', fontsize=7)
    ax_lag.set_title(str(model_name), fontsize=8)

    # Peak of this model's curve
    peak_idx = int(np.argmax(curve))
    peak_lag = lags_in_seconds[peak_idx]
    peak_val = curve[peak_idx]

    ax_lag.plot(peak_lag, peak_val, 'ro', markersize=3)
    ax_lag.annotate(
        f'Lag: {peak_lag:.2f}s',
        xy=(peak_lag, peak_val),
        xytext=(peak_lag + 0.2, peak_val),
        arrowprops=dict(arrowstyle='->', color='red'),
        fontsize=7,
    )

fig.tight_layout()
fig.savefig(get_unique_path(save_path, file_name + '.png'), dpi=300, bbox_inches='tight')
# plt.show()

