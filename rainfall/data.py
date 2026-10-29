"""Data acquisition and date-aware partitions, independent of model fitting."""
from pathlib import Path
import hashlib
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

SOURCE = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
SHA256 = "573fd715cd69fcacc4df32024d823b450ae3edaae7e8ff2eeb623adbed424014"
STATIONS = ("Melbourne", "MelbourneAirport", "Watsonia")


def download(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCE, timeout=90) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != SHA256:
            raise ValueError("Source checksum changed; inspect the new dataset before using it.")
        path.write_bytes(content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Dataset checksum does not match the documented source snapshot.")
    return path


def load(path):
    frame = pd.read_csv(path, parse_dates=["Date"])
    frame = frame.loc[frame.Location.isin(STATIONS) & frame.RainTomorrow.isin(["Yes", "No"])].copy()
    if frame.duplicated(["Date", "Location"]).any():
        raise ValueError("Duplicate station/date observations.")
    return frame.sort_values(["Date", "Location"]).reset_index(drop=True)


def features(frame):
    """Keep RainToday as an available predictor; predict the original RainTomorrow."""
    X = frame.drop(columns=["Date", "RainTomorrow"]).copy()
    phase = 2 * np.pi * (frame.Date.dt.dayofyear - 1) / 365.25
    X["SeasonSin"] = np.sin(phase)
    X["SeasonCos"] = np.cos(phase)
    return X


def partitions(dates):
    """70/15/15 percent of unique dates; never split a date across sets."""
    dates = pd.Series(dates).reset_index(drop=True)
    days = np.sort(dates.unique())
    if len(days) < 20:
        raise ValueError("At least 20 distinct dates are required.")
    a, b = int(.70 * len(days)), int(.85 * len(days))
    return tuple(np.flatnonzero(dates.isin(part)) for part in (days[:a], days[a:b], days[b:]))


def date_folds(dates, n_splits=3):
    """Expanding training windows with a one-observed-date gap."""
    dates = pd.Series(dates).reset_index(drop=True)
    days = np.sort(dates.unique())
    for tr, va in TimeSeriesSplit(n_splits=n_splits, gap=1).split(days):
        yield np.flatnonzero(dates.isin(days[tr])), np.flatnonzero(dates.isin(days[va]))
