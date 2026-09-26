# -*- coding: utf-8 -*-
"""ACS state-year cell loading through folktables, with our own resumable
acquisition (the folktables downloader is not reliable on every network).

Layout expected by folktables: {root}/{year}/1-Year/psam_p{fips}.csv,
extracted from csv_p{st}.zip. fetch_acs_cell() ensures both, resuming
partial downloads with HTTP Range requests.
"""
from __future__ import annotations
import io, os, time, zipfile
import urllib.request
import numpy as np
import pandas as pd

STATE_FIPS = {
 "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10",
 "FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20",
 "KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28",
 "MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36",
 "NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45",
 "SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54",
 "WI":"55","WY":"56"}

URL = "https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year/csv_p{st}.zip"


def _download_resumable(url, target, tries=6, timeout=120):
    for attempt in range(1, tries + 1):
        have = os.path.getsize(target) if os.path.exists(target) else 0
        req = urllib.request.Request(url, headers={"User-Agent": "study2/1.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                with open(target, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            return
        except Exception:
            if attempt == tries:
                raise
            time.sleep(min(30, 2 ** attempt))


def fetch_acs_cell(root, state, year):
    """Ensure the extracted person CSV for one state-year exists; return its path."""
    st = state.lower(); fips = STATE_FIPS[state.upper()]
    d = os.path.join(root, str(year), "1-Year")
    os.makedirs(d, exist_ok=True)
    csv_path = os.path.join(d, f"psam_p{fips}.csv")
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return csv_path
    zip_path = os.path.join(d, f"csv_p{st}.zip")
    if not (os.path.exists(zip_path) and zipfile.is_zipfile(zip_path)):
        _download_resumable(URL.format(year=year, st=st), zip_path)
        if not zipfile.is_zipfile(zip_path):        # torn download: refetch once
            os.remove(zip_path)
            _download_resumable(URL.format(year=year, st=st), zip_path)
    with zipfile.ZipFile(zip_path) as z:
        member = [m for m in z.namelist() if m.lower().endswith(".csv")][0]
        with z.open(member) as src, open(csv_path, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
    return csv_path


def occp_major_group(code):
    """SOC major group label for a four-digit OCCP code, or None."""
    from .specs import OCCP_MAJOR_GROUPS
    try:
        c = int(code)
    except (TypeError, ValueError):
        return None
    for lo, hi, name in OCCP_MAJOR_GROUPS:
        if lo <= c <= hi:
            return name
    return None


def weighted_median(values, weights):
    """Weighted median: the smallest value at which the cumulative weight
    share reaches one half."""
    v = np.asarray(values, float); w = np.asarray(weights, float)
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])


def load_cell(root, state, year, covariates, outcome_def, sample_requires=None,
              schooling="code", income_var="PINCP", fulltime=False,
              occupation=False, adjinc=False, weighted_median_threshold=False,
              age_range=(18, None), shift_female_schl=0.0):
    """One state-year analysis frame: X (with COW one-hot where present),
    y, group (1 = female), person weights.

    Sample: AGEP >= 18 and PINCP > 0, then complete cases on the covariates
    in use plus sample_requires (columns that must be observed even when
    not used as covariates). These filters do NOT mirror folktables
    ACSIncome, which uses AGEP > 16, PINCP > 100, WKHP > 0 and a valid COW
    for every sample; the premarket sample here therefore includes income
    recipients reporting no hours or class of worker, and the covset
    contrast changes the estimation sample as well as the conditioning
    (docstring corrected 15 Sep 2026 after review). The harmonised arm
    (REGISTERED_CHANGES, 15 Sep 2026) passes sample_requires=["WKHP",
    "COW"] to hold the sample at the extended set's complete cases while
    conditioning only on the premarket covariates, so the flip decomposes
    into sample and conditioning components. Defaults reproduce the T6
    grid's behaviour bit for bit. The cellmedian threshold is the
    unweighted median of the sample in use (disclosed limitation).

    Options for the arms added after the internal review of 22 Sep 2026
    (REGISTERED_CHANGES R4-R11; every default reproduces the registered
    behaviour):
      schooling="years"            SCHL recoded through specs.SCHL_TO_YEARS
                                   (the column keeps its name, so the
                                   proxy-column logic is unchanged)
      income_var="WAGP"            wage and salary income as the outcome
      fulltime=True                WKHP >= 35 and full-year weeks (WKW == 1
                                   up to 2018, WKWN >= 50 from 2019)
      occupation=True              OCCP major-group dummies appended to X
      adjinc=True                  income times ADJINC / 1e6 before use
      weighted_median_threshold    PWGTP-weighted cell median
      age_range=(25, 65)           inclusive age bounds
      shift_female_schl=delta      women's schooling column shifted by delta
                                   (mean-error arm R11), in the coding in use
    """
    path = fetch_acs_cell(root, state, year)
    base = ["AGEP", "SCHL", "SEX", "PINCP", "PWGTP", "WKHP", "COW"]
    if income_var != "PINCP":
        base.append(income_var)
    if fulltime:
        base += ["WKW", "WKWN"]
    if occupation:
        base.append("OCCP")
    if adjinc:
        base.append("ADJINC")
    df = pd.read_csv(path, usecols=lambda c: c in base, low_memory=False)
    lo_age, hi_age = age_range
    df = df[(df["AGEP"] >= lo_age) & (df[income_var].notna()) & (df[income_var] > 0)]
    if hi_age is not None:
        df = df[df["AGEP"] <= hi_age]
    if fulltime:
        df = df[df["WKHP"].notna() & (df["WKHP"] >= 35)]
        if "WKWN" in df.columns and df["WKWN"].notna().any():
            df = df[df["WKWN"].notna() & (df["WKWN"] >= 50)]
        elif "WKW" in df.columns and df["WKW"].notna().any():
            df = df[df["WKW"].notna() & (df["WKW"] == 1)]
        else:
            raise ValueError(f"{state} {year}: neither WKW nor WKWN carries values; "
                             "cannot impose the full-year restriction")
    need = list(dict.fromkeys(list(covariates) + list(sample_requires or [])))
    df = df.dropna(subset=[c for c in need if c != "COW"])
    if "COW" in need:
        df = df[df["COW"].notna()]
    if occupation:
        df = df[df["OCCP"].notna()]
        df = df.assign(OCC_GROUP=[occp_major_group(c) for c in df["OCCP"]])
        df = df[df["OCC_GROUP"].notna()]
    if schooling == "years":
        from .specs import SCHL_TO_YEARS
        df = df.assign(SCHL=df["SCHL"].astype(int).map(SCHL_TO_YEARS).astype(float))
        if df["SCHL"].isna().any():
            raise ValueError(f"{state} {year}: SCHL code outside 1-24 met the years crosswalk")
    elif schooling != "code":
        raise ValueError(schooling)
    if shift_female_schl:
        df = df.assign(SCHL=df["SCHL"].astype(float)
                       + shift_female_schl * (df["SEX"].to_numpy(float) == 2))
    y_cont = df[income_var].to_numpy(float)
    if adjinc:
        y_cont = y_cont * df["ADJINC"].to_numpy(float) / 1e6
    if outcome_def == "acs50k":
        y = (y_cont > 50000).astype(float)
    elif outcome_def == "cellmedian":
        thr = (weighted_median(y_cont, df["PWGTP"].to_numpy(float))
               if weighted_median_threshold else np.median(y_cont))
        y = (y_cont > thr).astype(float)
    else:
        raise ValueError(outcome_def)
    cols = [df[c].to_numpy(float) for c in covariates if c != "COW"]
    names = [c for c in covariates if c != "COW"]
    if "COW" in covariates:
        d = pd.get_dummies(df["COW"].astype(int), prefix="COW", drop_first=True)
        cols += [d[c].to_numpy(float) for c in d.columns]
        names += list(d.columns)
    if occupation:
        d = pd.get_dummies(df["OCC_GROUP"].astype(str), prefix="OCC", drop_first=True)
        cols += [d[c].to_numpy(float) for c in d.columns]
        names += list(d.columns)
    X = np.column_stack(cols)
    g = (df["SEX"].to_numpy(float) == 2).astype(int)     # 1 = female
    w = df["PWGTP"].to_numpy(float)
    options = {k: v for k, v in (("schooling", schooling), ("income_var", income_var),
                                 ("fulltime", fulltime), ("occupation", occupation),
                                 ("adjinc", adjinc),
                                 ("weighted_median_threshold", weighted_median_threshold),
                                 ("age_range", list(age_range)),
                                 ("shift_female_schl", shift_female_schl))
               if v not in ("code", "PINCP", False, 0.0, [18, None])}
    return {"X": X, "names": names, "y": y, "y_cont": y_cont,
            "g": g, "w": w, "n": len(df), "options": options}
