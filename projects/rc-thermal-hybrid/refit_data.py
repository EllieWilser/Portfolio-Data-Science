"""Load and clean the REFIT Smart Home dataset for the hybrid thermal model.

About the data
--------------
REFIT Smart Home dataset, Loughborough University (Firth et al., 2017).
DOI: https://doi.org/10.17028/rd.lboro.2070091   License: CC BY 4.0 (credit the authors).

20 homes near Loughborough, UK, monitored Oct 2013 to Apr 2015. What we use:

* Room air temperature in each room (Hobo loggers, every 15 or 30 min, deg C)
* Whole-house gas use (gas meter, cubic metres per 30 min)
* Outdoor air temperature and solar irradiance from the Loughborough University
  campus weather station (every 15 min). The homes are all near the campus, so
  we do not need a separate weather download.

The download is two files that matter:

* REFIT_BUILDING_SURVEY.xml (2 MB): describes every home, room and sensor.
  Each sensor reading series has an id like "TimeSeriesVariable41".
* REFIT_TIME_SERIES_VALUES.csv (1.3 GB unzipped, 25 million rows): the readings,
  as three columns: series id, time (UTC), value.

The CSV is too big to load every time, so `extract_subset()` reads it once in
chunks, keeps only the series we need, and saves a small file (`refit_subset.pkl`).
After that, everything runs from the small file in a second or two.

Typical use (from the repo root, in Positron's Python console or a terminal):

    from refit_data import download_refit, extract_subset, home_coverage, load_home

    download_refit()          # once: about 100 MB
    extract_subset()          # once: about 1 minute
    home_coverage()           # which homes have the best data?
    df = load_home("Building01", "2013-11-04", "2014-03-31")

Times are in UTC. The UK is on UTC in winter, so winter clock times are local times.
"""

from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Data files live next to this script, in data/refit/. That folder is git-ignored:
# the files are large, and the license lets anyone download them from the source.
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "refit"

# Direct download links from the Loughborough figshare record
# (https://repository.lboro.ac.uk/articles/dataset/REFIT_Smart_Home_dataset/2070091)
DOWNLOADS = {
    "ReadMe.txt": "https://ndownloader.figshare.com/files/8706646",
    "REFIT_BUILDING_SURVEY.xml": "https://ndownloader.figshare.com/files/8669629",
    "REFIT_TIME_SERIES_VALUES.zip": "https://ndownloader.figshare.com/files/7302803",
}

# The XML uses a "namespace", so every tag name has this prefix.
NS = "{http://www.refitsmarthomes.org}"

# The winter before the smart-home kit (smart radiator valves etc.) went in.
# Heating was run the normal way, which is simpler to model. Weather data starts 4 Nov 2013.
WINTER_2013 = ("2013-11-04", "2014-03-31")

# Turning gas volume into heat delivered to the house.
# 1 m3 of UK natural gas holds about 11.2 kWh (39.5 MJ/m3 x 1.02264 volume correction / 3.6).
# A condensing boiler turns roughly 85% of that into heat in the radiators.
KWH_PER_M3_GAS = 11.2
BOILER_EFFICIENCY = 0.85


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
def download_refit(data_dir: Path = DATA_DIR) -> None:
    """Download the REFIT files (about 100 MB) and unzip the readings CSV.

    Files that are already there are skipped, so it is safe to run again.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in DOWNLOADS.items():
        path = data_dir / name
        if path.exists():
            print(f"already have {name}")
            continue
        print(f"downloading {name} ...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for block in r.iter_content(chunk_size=1 << 20):
                    f.write(block)

    csv_path = data_dir / "REFIT_TIME_SERIES_VALUES.csv"
    if not csv_path.exists():
        print("unzipping the readings (1.3 GB) ...")
        with zipfile.ZipFile(data_dir / "REFIT_TIME_SERIES_VALUES.zip") as z:
            z.extractall(data_dir)
    print("done:", data_dir)


# ---------------------------------------------------------------------------
# 2. Catalogue of sensors (from the XML)
# ---------------------------------------------------------------------------
def sensor_catalog(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per reading series, saying which home/room it belongs to and what it measures.

    Columns: var_id, building, parent ("Climate", "Space" = a room, "Meter", ...),
    room_type, condition ("Heated"/"Unheated"), volume_m3, variable, units,
    interval_min, start, end.
    """
    root = ET.parse(data_dir / "REFIT_BUILDING_SURVEY.xml").getroot()
    rows = []

    def add(var, building, parent, room=None):
        rows.append(
            {
                "var_id": var.get("id"),
                "building": building,
                "parent": parent.tag.replace(NS, ""),
                "room_type": room.get("roomType") if room is not None else None,
                "condition": room.get("conditionType") if room is not None else None,
                "volume_m3": float(room.get("volume")) if room is not None and room.get("volume") else np.nan,
                "variable": var.get("variableType"),
                "units": var.get("units"),
                "interval_min": var.get("intervalLength"),
                "start": var.get("startDateTime"),
                "end": var.get("endDateTime"),
            }
        )

    climate = root.find(f".//{NS}Climate")
    for var in climate.iter(f"{NS}TimeSeriesVariable"):
        add(var, "climate", climate)

    for building in root.iter(f"{NS}Building"):
        bid = building.get("id")
        for child in building:  # Space (room), Meter, Radiator, Plug, ...
            room = child if child.tag == f"{NS}Space" else None
            for var in child.iter(f"{NS}TimeSeriesVariable"):
                add(var, bid, child, room)

    cat = pd.DataFrame(rows)
    cat["start"] = pd.to_datetime(cat["start"])
    cat["end"] = pd.to_datetime(cat["end"])
    return cat


def _wanted_series(cat: pd.DataFrame) -> pd.DataFrame:
    """The series the thermal model needs: room air temps, gas meters, weather."""
    room_temp = (cat["parent"] == "Space") & (cat["variable"] == "Air temperature")
    gas = (cat["parent"] == "Meter") & (cat["variable"] == "Gas volume")
    weather = (cat["parent"] == "Climate") & cat["variable"].isin(
        ["Air temperature", "Average horizontal solar irradiance", "Wind speed"]
    )
    return cat[room_temp | gas | weather]


# ---------------------------------------------------------------------------
# 3. Pull the series we need out of the 1.3 GB CSV (run once)
# ---------------------------------------------------------------------------
def extract_subset(data_dir: Path = DATA_DIR) -> Path:
    """Read the big CSV in chunks and keep only the series we need. Takes about a minute."""
    out = data_dir / "refit_subset.pkl"
    wanted = set(_wanted_series(sensor_catalog(data_dir))["var_id"])
    parts = []
    reader = pd.read_csv(
        data_dir / "REFIT_TIME_SERIES_VALUES.csv",
        names=["var_id", "time", "value"],
        header=0,
        chunksize=2_000_000,  # 2 million rows at a time keeps memory use small
    )
    for i, chunk in enumerate(reader):
        parts.append(chunk[chunk["var_id"].isin(wanted)])
        print(f"  read {(i + 1) * 2} million rows", end="\r")
    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df.to_pickle(out)
    print(f"\nsaved {len(df):,} readings to {out}")
    return out


def _load_subset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / "refit_subset.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run download_refit() then extract_subset() first")
    return pd.read_pickle(path)


def _series(subset: pd.DataFrame, var_ids, start, end, freq: str, how: str) -> pd.Series:
    """Join one or more series (a sensor can be split over several ids), then resample."""
    s = subset[subset["var_id"].isin(list(var_ids))].set_index("time")["value"].sort_index()
    s = s[~s.index.duplicated()]
    s = s[start:end]
    return s.resample(freq).sum(min_count=1) if how == "sum" else s.resample(freq).mean()


# ---------------------------------------------------------------------------
# 4. One tidy table per home
# ---------------------------------------------------------------------------
def load_home(
    building: str,
    start: str = WINTER_2013[0],
    end: str = WINTER_2013[1],
    freq: str = "30min",
    data_dir: Path = DATA_DIR,
    subset: pd.DataFrame | None = None,
    catalog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Tidy half-hourly table for one home, ready for the RC model.

    Columns
      t_in       whole-house indoor temperature [degC]: average of the heated rooms,
                 weighted by room volume (bigger rooms count more)
      n_rooms    how many rooms had a reading in that half hour
      t_out      outdoor air temperature [degC]
      i_sol      horizontal solar irradiance [kW/m2]
      wind       wind speed [m/s]
      gas_m3     gas used in the half hour [m3]
      q_heat     heat delivered by the boiler [kW] (gas -> kWh -> average power)

    Note: gas also heats hot water and (in some homes) cooks, so q_heat over-counts
    space heating a little. That is one thing the neural-network residual can soak up.
    Gaps are left as NaN; use `clean_for_model()` to deal with them.
    """
    subset = _load_subset(data_dir) if subset is None else subset
    cat = sensor_catalog(data_dir) if catalog is None else catalog
    want = _wanted_series(cat)
    mine = want[(want["building"] == building) | (want["parent"] == "Climate")]["var_id"]
    subset = subset[subset["var_id"].isin(mine)]  # small table for this home only: much faster
    start = pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)  # include the whole end day
    index = pd.date_range(start, end, freq=freq)

    # Indoor: one column per heated room, then a volume-weighted average.
    rooms = want[(want["building"] == building) & (want["parent"] == "Space") & (want["condition"] == "Heated")]
    room_cols = {}
    for key, grp in rooms.groupby(["room_type", "volume_m3"], dropna=False):
        room_cols[key] = _series(subset, grp["var_id"], start, end, freq, "mean")
    room_t = pd.DataFrame(room_cols).reindex(index)
    room_t = room_t.where((room_t > 0) & (room_t < 40))  # drop impossible readings
    vols = np.array([v if np.isfinite(v) else 0 for (_, v) in room_t.columns]) + 1e-9
    w = room_t.notna() * vols
    t_in = (room_t.fillna(0) * vols).sum(axis=1) / w.sum(axis=1)

    gas_ids = want[(want["building"] == building) & (want["parent"] == "Meter")]["var_id"]
    gas = _series(subset, gas_ids, start, end, freq, "sum").reindex(index)

    def weather(name, how="mean"):
        ids = want[(want["parent"] == "Climate") & (want["variable"] == name)]["var_id"]
        return _series(subset, ids, start, end, freq, how).reindex(index)

    hours_per_step = pd.Timedelta(freq) / pd.Timedelta(hours=1)
    df = pd.DataFrame(
        {
            "t_in": t_in.where(w.sum(axis=1) > 0),
            "n_rooms": room_t.notna().sum(axis=1),
            "t_out": weather("Air temperature"),
            "i_sol": weather("Average horizontal solar irradiance"),
            "wind": weather("Wind speed").where(lambda s: s < 60),  # a few bad spikes
            "gas_m3": gas,
            "q_heat": gas * KWH_PER_M3_GAS * BOILER_EFFICIENCY / hours_per_step,
        },
        index=index,
    )
    df.index.name = "time_utc"
    return df


def home_coverage(start: str = WINTER_2013[0], end: str = WINTER_2013[1], data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Rank homes by how complete their data is over the period. Higher is better.

    pct_complete = share of half hours where indoor temp, gas and weather are all present.
    """
    subset = _load_subset(data_dir)
    cat = sensor_catalog(data_dir)
    rows = []
    for b in sorted(cat.loc[cat["building"] != "climate", "building"].unique()):
        df = load_home(b, start, end, subset=subset, catalog=cat)
        ok = df[["t_in", "t_out", "gas_m3"]].notna().all(axis=1)
        rows.append(
            {
                "building": b,
                "pct_complete": round(100 * ok.mean(), 1),
                "pct_t_in": round(100 * df["t_in"].notna().mean(), 1),
                "pct_gas": round(100 * df["gas_m3"].notna().mean(), 1),
                "heated_rooms": int(df["n_rooms"].max()),
                "mean_t_in": round(df["t_in"].mean(), 1),
                "gas_m3_total": round(df["gas_m3"].sum(), 0),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_complete", ascending=False).reset_index(drop=True)


def clean_for_model(df: pd.DataFrame, max_gap_steps: int = 4) -> pd.DataFrame:
    """Fill short gaps (up to 2 hours at 30 min) by straight-line interpolation,
    then keep the longest unbroken stretch, because the RC simulation needs
    one continuous run of data.
    """
    out = df.copy()
    cols = ["t_in", "t_out", "i_sol", "wind", "gas_m3", "q_heat"]
    out[cols] = out[cols].interpolate(limit=max_gap_steps, limit_area="inside")
    ok = out[["t_in", "t_out", "i_sol", "q_heat"]].notna().all(axis=1)
    run_id = (ok != ok.shift()).cumsum()  # number each run of True/False
    longest = ok.groupby(run_id).sum().idxmax()
    return out[run_id == longest]


if __name__ == "__main__":
    # Quick check: python projects/rc-thermal-hybrid/refit_data.py
    print(home_coverage().to_string())
