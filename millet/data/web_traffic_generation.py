"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

Generation functions for the WebTraffic dataset

Signature injection code inspired by (under Apache 2.0 License):
https://github.com/mononitogoswami/tsad-model-selection/blob/master/src/tsadams/model_selection/inject_anomalies.py
"""

import json
import os
import os.path
import re
# See https://stackoverflow.com/a/15012814/355230
from _ctypes import PyObj_FromPtr  # type: ignore
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import bernoulli, norm

WEBTRAFFIC_CLZ_NAMES = [
    "none",
    "spikes",
    "flip",
    "skew",
    "noise",
    "cutoff",
    "average",
    "wander",
    "peak",
    "trough",
]


def create_random_week(interval: int = 10) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Create a random week-long time series.

    :param interval: Interval between samples (minutes)
    :return: Time series rate, time series sample, random params
    """
    # Sample random values
    amplitude = np.random.uniform(2, 4)
    intercept = np.random.uniform(2.5, 5)
    phase = np.random.uniform(-0.05, 0.05)
    skew = np.random.uniform(1.0, 3.0)
    sample_std = np.random.uniform(0.1, 0.3)
    week_amplitude = np.random.uniform(0.8, 1.2)
    # Create param dict
    params = {
        "amplitude": amplitude,
        "intercept": intercept,
        "phase": phase,
        "skew": skew,
        "sample_std": sample_std,
        "week_amplitude": week_amplitude,
    }
    # Return time series rate, sample, and params
    rate, sample = _create_week_with_seasonality(
        amplitude, intercept, phase, skew, sample_std, week_amplitude, interval=interval
    )
    return rate, sample, params


def inject_signature(ts: np.ndarray, signature_name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Inject a signature into an existing time series.

    :param ts: Existing (underlying) time series.
    :param signature_name: Signature type to inject.
    :return: Time series with injected signature.
    """
    # Check for valid signature name
    if signature_name not in WEBTRAFFIC_CLZ_NAMES:
        raise ValueError("Invalid signature name: {:s}".format(signature_name))
    # Inject signature
    signature_data = _inject_signatures_switcher(ts, signature_name)
    ts_w_signature, signature_locations, signature_details = signature_data
    # Ensure time series doesn't go below zero
    ts_w_signature = ts_w_signature.clip(min=0)
    # Calculate signature deviations
    signature_details["mae"] = float(np.mean(np.abs(ts - ts_w_signature)))
    signature_details["mse"] = float(np.mean((ts - ts_w_signature) ** 2))
    return ts_w_signature, signature_locations, signature_details


def generate_and_save_dataset(
    dataset_name: str, split: str, signature_clzs: List[str], interval: int = 10, n_ts_per_clz: int = 50
) -> None:
    """
    Generate a complete WebTraffic dataset and write to file.

    :param dataset_name: Name of dataset.
    :param split: Split e.g. train or test.
    :param signature_clzs: List of signature types. Defines number of classes and order.
    :param interval: Sampling interval in minutes.
    :param n_ts_per_clz: Number of time series per class.
    :return: None
    """
    n_clz = len(signature_clzs)
    ts_len = 60 * 24 * 7 // interval
    ts_data = np.zeros((n_clz * n_ts_per_clz, ts_len + 1))
    metadata = []
    idx = 0
    for clz, signature_type in enumerate(signature_clzs):
        for _ in range(n_ts_per_clz):
            # Generate random sample
            _, ts, ts_params = create_random_week(interval=interval)
            # Inject signature
            signature_data = inject_signature(ts, signature_type)
            ts_w_signature, signature_locations, signature_params = signature_data
            # Add to dataframe
            ts_data[idx, 0] = clz
            ts_data[idx, 1:] = ts_w_signature
            # Add to metadata
            metadata.append(
                {
                    "idx": idx,
                    "clz": clz,
                    "signature": signature_type,
                    "time_series_params": ts_params,
                    "signature_params": signature_params,
                    "signature_locations": NoIndent(signature_locations.tolist()),
                }
            )
            idx += 1
    df = pd.DataFrame(
        columns=["Class"] + ["t_{:d}".format(f) for f in range(ts_len)],
        data=ts_data,
    )
    df["Class"] = df["Class"].astype("int")
    print("Generated dataframe containing {:d} time series".format(len(df)))

    save_dir = "data/WebTraffic"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data_path = "{:s}/{:s}_{:s}.csv".format(save_dir, dataset_name, split)
    metadata_path = "{:s}/{:s}_{:s}_metadata.json".format(save_dir, dataset_name, split)
    df.to_csv(data_path, index=False)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, cls=JSONNoIndentEncoder)
    print("Saved data to {:s}".format(data_path))
    print("Saved metadata to {:s}".format(metadata_path))


def _warped_sin(
    n_cycles: int,
    n_samples_per_cycle: int,
    amplitude: float,
    intercept: float,
    phase: float,
    skew: float,
    sample_std: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function for plotting warped sine waves

    :param n_cycles: How many cycles to produce (one cycle = one day)
    :param n_samples_per_cycle: How many timesteps per cycle (per day)
    :param amplitude: Time series amplitude scalar (daily)
    :param intercept: Time series intercept scalar
    :param phase: Time series phase shift scalar
    :param skew: Time series skew scalar (straighter or more angled)
    :param sample_std: Standard deviation to sample from rate
    :return: Numpy array of rate, numpy array of sample (noise added to rate)
    """
    # Generate xs and shift
    xs = np.linspace(0, n_cycles, n_cycles * n_samples_per_cycle)
    xs -= phase
    xs *= 2.0 * np.pi
    # Calculate rate
    rate = amplitude / 2.0 * np.sin(xs - np.sin(xs) / skew) + intercept
    # Sample from rate
    sample = np.random.normal(rate, sample_std)
    return rate, sample


def _create_week_with_seasonality(
    amplitude: float,
    intercept: float,
    phase: float,
    skew: float,
    sample_std: float,
    week_amplitude: float,
    interval: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a week long time series with daily and weekly seasonality

    :param amplitude: Time series amplitude scalar (daily)
    :param intercept: Time series intercept scalar
    :param phase: Time series phase shift scalar
    :param skew: Time series skew scalar (straighter or more angled). Closer to 1 = more skew. >> 1 normal sine.
    :param sample_std: Standard deviation to sample from rate
    :param week_amplitude: Time series amplitude scalar (weekly). 0 = no weekly seasonality.
    :param interval: Interval between samples (minutes)
    :return: Numpy array of rate, numpy array of sample (noise added to rate)
    """
    # Calculate samples per day based on minute interval
    n_samples_per_day = 24 * 60 // interval
    # Create week seasonality
    week_seasonal_rate, _ = _warped_sin(1, 60 * 24 * 7 // interval, week_amplitude, 1, 0.6, 2, 0.0)
    # Create week time series (without week seasonality)
    week_rate, week_sample = _warped_sin(7, n_samples_per_day, amplitude, intercept, phase + 0.55, skew, sample_std)
    # Apply week seasonality to week time series
    week_rate *= week_seasonal_rate
    week_sample *= week_seasonal_rate
    return week_rate, week_sample


def _inject_signatures_switcher(ts: np.ndarray, signature_name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Switcher to call correct injection method for signature type.

    :param ts: Time series to inject signature in to.
    :param signature_name: Signature type.
    :return: Time series with signature injected.
    """
    # Inject signature type and return
    if signature_name == "none":
        return ts, np.asarray([]), {"len": 0}
    if signature_name == "spikes":
        return _inject_spikes(ts)
    if signature_name == "flip":
        return _inject_flip(ts)
    if signature_name == "skew":
        return _inject_skew(ts)
    if signature_name == "noise":
        return _inject_noise(ts)
    if signature_name == "cutoff":
        return _inject_cutoff(ts)
    if signature_name == "average":
        return _inject_average(ts)
    if signature_name == "wander":
        return _inject_wander(ts)
    if signature_name == "peak":
        return _inject_peak_or_trough(ts, False)
    if signature_name == "trough":
        return _inject_peak_or_trough(ts, True)
    raise NotImplementedError


def _get_random_start_end(ts: np.ndarray, len_min: int, len_max: int) -> Tuple[int, int]:
    """
    Get random window in time series.

    :param ts: Time series to find random window in.
    :param len_min: Minimum length of window.
    :param len_max: Maximum length of window.
    :return: Start position, end position.
    """
    signature_len = np.random.randint(len_min, len_max + 1)
    signature_start = np.random.randint(len(ts) - signature_len + 1)
    signature_end = signature_start + signature_len
    return signature_start, signature_end


def _inject_spikes(
    ts: np.ndarray,
    spike_proba: float = 0.01,
    spike_mean: float = 3.0,
    spike_scale: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    ts_len = len(ts)
    # Generate random spike locations
    mask = bernoulli.rvs(p=spike_proba, size=ts_len)
    signature_locations = np.asarray([[s, s] for s in np.nonzero(mask)[0]])
    # Generate random spike values
    spikes = np.random.normal(loc=spike_mean, scale=spike_scale, size=ts_len)
    for i in range(len(spikes)):
        spikes[i] *= np.random.choice([-1, 1])
    # Mask spikes and add to original time series
    signatures = mask * spikes
    ts_w_signature = ts + signatures
    # Gather signature details
    signature_details = {"len": int(len(np.nonzero(mask)[0]))}
    return ts_w_signature, signature_locations, signature_details


def _inject_flip(
    ts: np.ndarray,
    len_min: int = 36,  # 0.25 day
    len_max: int = 288,  # 2 days
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    # Get random signature location
    signature_start, signature_end = _get_random_start_end(ts, len_min, len_max)
    signature_locations = np.asarray([[signature_start, signature_end]])
    # Inject the flip signature
    ts_w_signature = ts.copy()
    rev_window = ts_w_signature[signature_start:signature_end][::-1]
    ts_w_signature[signature_start:signature_end] = rev_window
    signature_details = {
        "len": int(signature_end - signature_start),
    }
    return ts_w_signature, signature_locations, signature_details


def _inject_skew(
    ts: np.ndarray,
    len_min: int = 36,  # 0.25 day
    len_max: int = 288,  # 2 days
    skew_min: float = 0.25,
    skew_max: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    # Get random signature location
    signature_start, signature_end = _get_random_start_end(ts, len_min, len_max)
    signature_locations = np.asarray([[signature_start, signature_end]])
    # Fit an interpolation function on the signature window
    signature_len = signature_end - signature_start
    xs = np.arange(signature_end - signature_start)
    interp_func = interpolate.interp1d(xs, ts[signature_start:signature_end])
    # Get the point in the window we want to skew the midpoint to
    #   [0.5 - skew_max, 0.5 - skew_min] U [0.5 + skew_min, 0.5 + skew_max]
    skew_amount = np.random.uniform(skew_min, skew_max)
    skew_point_f = 0.5 + np.random.choice([-1, 1]) * skew_amount
    skew_point = int((signature_end - signature_start) * skew_point_f)
    # Generate interpolated time steps - squishing or stretching the x-axis
    mid_point = signature_len // 2
    left_x_interp = np.linspace(0, mid_point, skew_point)
    right_x_interp = np.linspace(mid_point, signature_len - 1, signature_len - skew_point)
    # Get new interpolated values for the interpolated time steps
    ts_left_interp = interp_func(left_x_interp)
    ts_right_interp = interp_func(right_x_interp)
    # Inject the skewed values
    ts_w_signature = ts.copy()
    mid = signature_start + len(ts_left_interp)
    ts_w_signature[signature_start:mid] = ts_left_interp
    ts_w_signature[mid:signature_end] = ts_right_interp
    signature_details = {
        "len": int(signature_end - signature_start),
        "skew": float(skew_point_f),
    }
    return ts_w_signature, signature_locations, signature_details


def _inject_noise(
    ts: np.ndarray,
    len_min: int = 36,  # 0.25 day
    len_max: int = 288,  # 2 days
    noise_std_min: float = 0.5,
    noise_std_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    # Get random signature location
    signature_start, signature_end = _get_random_start_end(ts, len_min, len_max)
    signature_locations = np.asarray([[signature_start, signature_end]])
    # Generate noise
    noise_std = np.random.uniform(noise_std_min, noise_std_max)
    noise = np.random.normal(scale=noise_std, size=(signature_end - signature_start))
    # Inject the noise
    ts_w_signature = ts.copy()
    ts_w_signature[signature_start:signature_end] += noise
    signature_details = {
        "len": signature_end - signature_start,
        "noise_std": float(noise_std),
    }
    return ts_w_signature, signature_locations, signature_details


def _inject_cutoff(
    ts: np.ndarray,
    len_min: int = 36,  # 0.25 day
    len_max: int = 288,  # 2 days
    cutoff_min: float = 0.0,
    cutoff_max: float = 0.2,
    noise_std: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    # Get random signature location
    signature_start, signature_end = _get_random_start_end(ts, len_min, len_max)
    signature_locations = np.asarray([[signature_start, signature_end]])
    # Generate cutoff
    cutoff_val = np.random.uniform(cutoff_min, cutoff_max)
    cutoff = np.random.normal(loc=cutoff_val, scale=noise_std, size=(signature_end - signature_start))
    # Inject the cutoff
    ts_w_signature = ts.copy()
    ts_w_signature[signature_start:signature_end] = cutoff
    signature_details = {
        "len": int(signature_end - signature_start),
        "cutoff": float(cutoff_val),
    }
    return ts_w_signature, signature_locations, signature_details


def _inject_average(
    ts: np.ndarray,
    len_min: int = 36,  # 0.25 day
    len_max: int = 288,  # 2 days
    window_size_min: int = 5,
    window_size_max: int = 10,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    # Get random signature location
    signature_start, signature_end = _get_random_start_end(ts, len_min, len_max)
    signature_locations = np.asarray([[signature_start, signature_end]])
    # Generate moving average
    window_size = np.random.randint(window_size_min, window_size_max)
    moving_average = np.convolve(ts, np.ones(window_size), "same") / window_size
    # Inject the average smoothing
    ts_w_signature = ts.copy()
    ts_w_signature[signature_start:signature_end] = moving_average[signature_start:signature_end]
    signature_details = {
        "len": int(signature_end - signature_start),
        "window_size": int(window_size),
    }
    return ts_w_signature, signature_locations, signature_details


def _inject_wander(
    ts: np.ndarray,
    len_min: int = 36,  # 0.25 day
    len_max: int = 288,  # 2 days
    wander_min: float = 2.0,
    wander_max: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    # Get random signature location
    signature_start, signature_end = _get_random_start_end(ts, len_min, len_max)
    signature_locations = np.asarray([[signature_start, signature_end]])
    # Get the amount we want to wander by
    #   [-wander_max, -wander_min] U [wander_min, wander_max]
    wander_amount = np.random.uniform(wander_min, wander_max)
    wander_amount *= np.random.choice([-1, 1])
    wander = np.linspace(0, wander_amount, signature_end - signature_start)
    # Inject the wandering
    ts_w_signature = ts.copy()
    ts_w_signature[signature_start:signature_end] += wander
    signature_details = {
        "len": int(signature_end - signature_start),
        "w:": float(wander_amount),
    }
    return ts_w_signature, signature_locations, signature_details


def _inject_peak_or_trough(
    ts: np.ndarray,
    trough: bool,
    len_min: int = 36,  # 0.25 day
    len_max: int = 288,  # 2 days
    scale_min: float = 1.5,
    scale_max: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    # Get random signature location
    signature_start, signature_end = _get_random_start_end(ts, len_min, len_max)
    signature_locations = np.asarray([[signature_start, signature_end]])
    # Generate normal distribution
    scale_amount = np.random.uniform(scale_min, scale_max)
    # Invert if adding a trough
    if trough:
        scale_amount *= -1
    pdf = norm.pdf(np.linspace(-5, 5, signature_end - signature_start), loc=0, scale=1)
    scale = pdf * scale_amount + 1
    # Inject the peak or trough
    ts_w_signature = ts.copy()
    ts_w_signature[signature_start:signature_end] *= scale
    signature_details = {
        "len": int(signature_end - signature_start),
        "scale": float(scale_amount),
    }
    return ts_w_signature, signature_locations, signature_details


# Fix for JSON formatting.
# Adapted (CC BY-SA 3.0) from: https://stackoverflow.com/questions/42710879/write-two-dimensional-list-to-json-file
class NoIndent(object):
    """No indent JSON value wrapper."""

    def __init__(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError("Only lists and tuples can be wrapped")
        self.value = value


class JSONNoIndentEncoder(json.JSONEncoder):
    """No indent JSON encoder - produces cleaner JSON outputs."""

    FORMAT_SPEC = "@@{}@@"  # Unique string pattern of NoIndent object ids.
    regex = re.compile(FORMAT_SPEC.format(r"(\d+)"))  # compile(r'@@(\d+)@@')

    def __init__(self, **kwargs):
        # Keyword arguments to ignore when encoding NoIndent wrapped values.
        ignore = {"cls", "indent"}
        # Save copy of any keyword argument values needed for use here.
        self._kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
        super(JSONNoIndentEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (
            self.FORMAT_SPEC.format(id(obj))
            if isinstance(obj, NoIndent)
            else super(JSONNoIndentEncoder, self).default(obj)
        )

    def iterencode(self, obj, **kwargs):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        # Replace any marked-up NoIndent wrapped values in the JSON repr
        # with the json.dumps() of the corresponding wrapped Python object.
        for encoded in super(JSONNoIndentEncoder, self).iterencode(obj, **kwargs):
            match = self.regex.search(encoded)
            if match:
                _id = int(match.group(1))
                no_indent = PyObj_FromPtr(_id)
                json_repr = json.dumps(no_indent.value, **self._kwargs)
                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                encoded = encoded.replace('"{}"'.format(format_spec.format(_id)), json_repr)
            yield encoded
