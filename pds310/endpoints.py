from typing import Tuple, List
import pandas as pd

ID_COL = "SUBJID"
STUDY_COL = "STUDYID"


def derive_os_from_adsl(
    adsl: pd.DataFrame,
    death_flag_col: str = "DTHX",
    time_col: str = "DTHDYX",
    trtsdt_col: str = None,
    dthdt_col: str = None,
) -> pd.DataFrame:
    """Derive OS labels from ADSL.

    Priority order for time:
      1) use numeric `DTHADY` when available
      2) else compute (DTHDT - TRTSDT).days when both dates exist
      3) else drop row (no OS time)

    Event is 1 if DTHFL indicates death (Y/YES/1/TRUE), else 0.
    """
    df = adsl.copy()

    # Event
    ev = df.get(death_flag_col)
    if ev is None:
        ev = 0
    else:
        # In 310, DTHX appears to be 0/1 or Y/N. Normalize to 0/1
        s = df[death_flag_col]
        if s.dtype.kind in ("i", "u", "f"):
            ev = (pd.to_numeric(s, errors="coerce") > 0).fillna(0).astype(int)
        else:
            ev = s.astype("string").str.upper().isin(["Y", "YES", "1", "TRUE"]).astype(int)

    # Time from preferred column
    t = pd.to_numeric(df.get(time_col), errors="coerce") if time_col in df.columns else pd.Series([pd.NA] * len(df))

    # Fallback: compute from dates if needed
    need = t.isna() if hasattr(t, "isna") else pd.Series([False] * len(df))
    if need.any() and trtsdt_col and dthdt_col and trtsdt_col in df.columns and dthdt_col in df.columns:
        # Ensure datetime
        trt = pd.to_datetime(df[trtsdt_col], errors="coerce")
        dth = pd.to_datetime(df[dthdt_col], errors="coerce")
        delta = (dth - trt).dt.days
        t = t.fillna(delta)

    out = df[[ID_COL, STUDY_COL]].copy()
    out["time"] = pd.to_numeric(t, errors="coerce")
    out["event"] = ev
    out = out.dropna(subset=["time"]).reset_index(drop=True)
    out["time"] = out["time"].astype(float)
    out["event"] = out["event"].astype(int)
    return out



def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def derive_eot_from_adlb(adlb: pd.DataFrame) -> pd.DataFrame:
    """Proxy EOT time from ADLB last observed visit day per subject.

    Returns columns: SUBJID, STUDYID, time
    """
    if adlb is None or adlb.empty:
        return pd.DataFrame(columns=[ID_COL, STUDY_COL, "time"])  # empty
    df = adlb.copy()
    if "VISITDY" not in df.columns:
        return pd.DataFrame(columns=[ID_COL, STUDY_COL, "time"])  # cannot derive
    df["VISITDY"] = pd.to_numeric(df["VISITDY"], errors="coerce")
    agg = (
        df.dropna(subset=["VISITDY"])  # drop rows without day
          .groupby([ID_COL, STUDY_COL], as_index=False)["VISITDY"].max()
          .rename(columns={"VISITDY": "time"})
    )
    agg["time"] = pd.to_numeric(agg["time"], errors="coerce").astype(float)
    agg = agg.dropna(subset=["time"]).reset_index(drop=True)
    return agg


def derive_ae_from_adae(adae: pd.DataFrame) -> pd.DataFrame:
    """Derive earliest AE start day per subject from ADAE.

    Tries common ADaM columns for AE start day: AESTDY, AESTDYI, ASTDY, ASTDYI.

    Returns columns: SUBJID, STUDYID, ae_time, event_ae
    """
    if adae is None or adae.empty:
        return pd.DataFrame(columns=[ID_COL, STUDY_COL, "ae_time", "event_ae"])  # empty
    df = adae.copy()
    day_col = _first_existing_column(df, ["AESTDY", "AESTDYI", "ASTDY", "ASTDYI", "AESTDYX"]) or ""
    if day_col == "":
        return pd.DataFrame(columns=[ID_COL, STUDY_COL, "ae_time", "event_ae"])  # cannot derive
    df[day_col] = pd.to_numeric(df[day_col], errors="coerce")
    # earliest AE per subject
    agg = (
        df.dropna(subset=[day_col])
          .groupby([ID_COL, STUDY_COL], as_index=False)[day_col].min()
          .rename(columns={day_col: "ae_time"})
    )
    agg["ae_time"] = pd.to_numeric(agg["ae_time"], errors="coerce").astype(float)
    agg = agg.dropna(subset=["ae_time"]).reset_index(drop=True)
    agg["event_ae"] = 1
    return agg


def prepare_eot_competing_from_adam(adsl: pd.DataFrame, adlb: pd.DataFrame, adae: pd.DataFrame) -> pd.DataFrame:
    """Build EOT times and AE cause flag for PDS310 using ADaM tables.

    - EOT time is proxied by last lab visit day from ADLB (max VISITDY)
    - AE event flag is 1 if an AE occurs on/before EOT (from ADAE earliest start day)

    Returns columns: SUBJID, STUDYID, time, event_ae, event_prog, event_complete
    """
    eot = derive_eot_from_adlb(adlb)
    ae = derive_ae_from_adae(adae)
    if eot is None or eot.empty:
        return pd.DataFrame(columns=[ID_COL, STUDY_COL, "time", "event_ae", "event_prog", "event_complete"])  # empty
    out = eot.copy()
    if ae is not None and not ae.empty:
        out = out.merge(ae[[ID_COL, STUDY_COL, "ae_time"]], on=[ID_COL, STUDY_COL], how="left")
        out["event_ae"] = ((out["ae_time"].astype(float) <= out["time"].astype(float))).fillna(False).astype(int)
        out = out.drop(columns=[c for c in ["ae_time"] if c in out.columns])
    else:
        out["event_ae"] = 0
    # We cannot distinguish other causes; mark remainder as non-AE complete
    out["event_prog"] = 0  # unknown in ADaM provided
    out["event_complete"] = (1 - out["event_ae"]).astype(int)
    return out[[ID_COL, STUDY_COL, "time", "event_ae", "event_prog", "event_complete"]]


