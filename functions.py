import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from pathlib import Path

def get_unique_path(directory: Path, filename: str) -> Path:
    """Return a path that won't overwrite an existing file. Creates directory if needed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    full_path = directory / filename
    base, ext = full_path.stem, full_path.suffix
    i = 1
    while full_path.exists():
        full_path = directory / f"{base}_{i}{ext}"
        i += 1
    return full_path

def subsampling_trRSA(n_timepoints, subsample_tps, n_subsamples, n_iterations,
                      selected_event_indices, event_start_tps,
                      mask=None, min_event_gap=5, random_state=None):
    # ---- checks ----
    if subsample_tps <= 0:
        raise ValueError("subsample_tps must be positive.")
    if n_timepoints < subsample_tps:
        raise ValueError("Subsample size is larger than number of time points.")
    if min_event_gap < 1:
        raise ValueError("min_event_gap must be >= 1 (events cannot repeat).")

    event_start_tps = np.asarray(event_start_tps)
    # unique + sorted
    selected_event_indices = np.unique(np.asarray(list(selected_event_indices), dtype=int))
    if selected_event_indices.size and (selected_event_indices.min() < 0
                                        or selected_event_indices.max() >= event_start_tps.size):
        raise ValueError("selected_event_indices out of range for event_start_tps.")

    rng = (random_state if isinstance(random_state, np.random.Generator)
           else np.random.default_rng(random_state))

    if mask is None:
        mask = np.ones(n_timepoints, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != n_timepoints:
            raise ValueError("mask length must equal n_timepoints.")

    # ---- event -> window start (event in the middle of the window) ----
    event_tps = np.rint(event_start_tps[selected_event_indices]).astype(int)
    cand_starts = event_tps - subsample_tps // 2

    # window must fit in the recording
    in_range = (cand_starts >= 0) & (cand_starts <= n_timepoints - subsample_tps)

    # window must lie entirely inside True regions of the mask
    run_sum = np.convolve(mask.astype(np.int32),
                          np.ones(subsample_tps, dtype=np.int32), mode='valid')
    fully_valid = np.zeros_like(in_range)
    fully_valid[in_range] = run_sum[cand_starts[in_range]] == subsample_tps
    # fully valid is the same shape as the input indices, we want to make it a mask for filtering

    # keep start <-> event pairing (don't use intersect1d, it loses the mapping)
    cand_starts = cand_starts[fully_valid]
    cand_events = selected_event_indices[fully_valid]

    if cand_starts.size == 0:
        raise ValueError("No valid event-centered windows fit within the mask.")
    if cand_starts.size < n_subsamples:
        raise ValueError(
            f"Only {cand_starts.size} valid events available, "
            f"but {n_subsamples} are needed per iteration."
        )

    # ---- draw non-overlapping, non-repeating events per iteration ----
    starts = np.empty((n_iterations, n_subsamples), dtype=int)
    event_idx = np.empty((n_iterations, n_subsamples), dtype=int)

    for it in range(n_iterations):
        order = rng.permutation(cand_starts.size) # taking a random pick of indexes from the candidate starts for the current iteration
        s, e = cand_starts[order], cand_events[order]
        alive = np.ones(s.size, dtype=bool)
        chosen = []
        for i in range(s.size):
            if not alive[i]:
                continue
            chosen.append(i)
            if len(chosen) == n_subsamples:
                break
            # drop windows that overlap in time, or events too close in index
            alive &= (np.abs(s - s[i]) >= subsample_tps) & \
                     (np.abs(e - e[i]) >= min_event_gap)
        if len(chosen) < n_subsamples:
            raise ValueError(
                f"Not enough non-overlapping events for iteration {it}. "
                f"Needed {n_subsamples}, got {len(chosen)}."
            )
        starts[it] = s[chosen]
        event_idx[it] = e[chosen]

    timeseries_subsampling_idx = starts[:, :, None] + np.arange(subsample_tps, dtype=int)
    return timeseries_subsampling_idx, event_idx


# outputs of the function: 1. subsample indices to be applied to a time series, with the event at the center 2. the event index

def apply_subsampling_indices(data, subsample_indices):
    """
    Efficiently applies precomputed subsampling indices using NumPy advanced indexing.

    Parameters:
    - data: (features, timepoints) array
    - subsample_indices: (iterations, n_subsamples, subsample_tps) array

    Returns:
    - subsampled_data: (iterations, n_subsamples, features, subsample_tps) array
    """
    # Reshape data for broadcasting: (1, 1, features, timepoints)
    data_expanded = data[np.newaxis, np.newaxis, :, :]
    
    # Reshape indices for advanced indexing: (iterations, n_subsamples, 1, subsample_tps)
    indices_expanded = subsample_indices[:, :, np.newaxis, :]
    
    # Use advanced indexing to gather all subsamples at once
    subsampled_data = np.take_along_axis(
        data_expanded,
        indices_expanded,
        axis=-1  # Take along the timepoints dimension
    )
    
    return subsampled_data  # (iterations, n_subsamples, features, subsample_tps)

def compute_rdm_series(subsampled_data, metric):
    """
    Computes RDM series for all iterations.

    Parameters:
    - subsampled_data: shape (iterations, n_subsamples, n_features, subsample_tps)
    - metric: distance metric string (e.g., 'correlation', 'cosine', 'euclidean')

    Returns:
    - rdm_series: array of shape (iterations, subsample_tps, rdm_length), e.g. if n_subsamples = 20, 20*19/2 = 190 is the rdm_length
    """

    iterations, n_subsamples, _, subsample_tps = subsampled_data.shape
    rdm_length = n_subsamples * (n_subsamples - 1) // 2
    rdm_series = np.empty((iterations, subsample_tps, rdm_length))

    for it in range(iterations):
        for t in range(subsample_tps):
            # shape: (n_subsamples, n_features)
            data_slice = subsampled_data[it, :, :, t]
            rdm_series[it, t] = pdist(data_slice, metric=metric)

    return rdm_series

def compute_model_rdm(model, event_idx, metric):
    '''
    model shape: (n_events, features)
    '''

    n_iterations, n_events = event_idx.shape
    rdm_length = n_events * (n_events - 1) // 2
    selected_events_rdm = np.empty((n_iterations, rdm_length))

    for it in range(n_iterations):
        selected_events_rdm[it] = pdist(model[event_idx[it], :], metric=metric)

    return selected_events_rdm

# for each iteration correlate the single model RDM vectro with all the rdms for different tps within a subsample
def batched_corr(A, b):
    """
    A: (n_iterations, n_subsample_tps, rdm_len)
    b: (n_iterations, rdm_len)
    returns: (iterations, n_subsample_tps) Pearson correlations
    """
    # center values along the last axis
    A_c = A - A.mean(axis=-1, keepdims=True)
    b_c = b - b.mean(axis=-1, keepdims=True)

    # numerator: per-iteration dot of each of the rows with b (matrix multiplication)
    num = np.einsum('ijk,ik->ij', A_c, b_c)

    # denominator: product of norms
    den = np.linalg.norm(A_c, axis=-1) * np.linalg.norm(b_c, axis=-1, keepdims=True)

    # achieve correlation = dot_product / vector_len

    return num / den

# np.einsum note ijk is dim0, dim1, and dim2. i and j remain after arrow because we want to keep the iterations and the subsample tps dimension, but multiply over dimension k
