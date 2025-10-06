from typing import Tuple
import pandas as pd

ID_COL = "RPT"
STUDY_COL = "STUDYID"


def derive_os_from_core(core: pd.DataFrame,
                        time_col: str = "LKADT_P",
                        death_col: str = "DEATH",
                        time_unit_col: str = "LKADT_PER") -> pd.DataFrame:
    df = core[[ID_COL, STUDY_COL, time_col, death_col, time_unit_col]].copy()
    df.rename(columns={time_col: "time", death_col: "event", time_unit_col: "time_unit"}, inplace=True)
    # Event mapping: YES -> 1, else 0; treat NA as 0 (censored)
    ev = df["event"].astype("string").str.upper().eq("YES")
    df["event"] = ev.fillna(False).astype(int)
    # Time as numeric days; rows with missing time are dropped
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    return df


def derive_ae_discontinuation(core: pd.DataFrame,
                              reason_col: str = "ENDTRS_C",
                              time_col: str = "ENTRT_PC",
                              ae_reasons: Tuple[str, ...] = ("AE", "possible_AE")) -> pd.DataFrame:
    df = core[[ID_COL, STUDY_COL, reason_col, time_col]].copy()
    df.rename(columns={time_col: "time", reason_col: "reason"}, inplace=True)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = df["reason"].astype("string").isin(ae_reasons).fillna(False).astype(int)
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    return df
