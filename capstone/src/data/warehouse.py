"""Warehouse access layer — DuckDB over HuggingFace parquets.

Reuses the patterns from w03/w04 notebooks. All queries run against locally
cached parquet files (no repeated downloads). Token stays in the environment,
never in a cell.
"""
import os
import duckdb
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download, get_token

DS = "FlyRank/internship-warehouse"
MONTHS = ["2026-02", "2026-03", "2026-04"]
DECISION_MOMENT = pd.Timestamp("2026-03-31")


def _get_parquet_paths() -> dict[str, str]:
    """Download (or use cache) the monthly parquet files and dimension tables."""
    paths = {}
    for m in MONTHS:
        paths[m] = hf_hub_download(
            DS, f"fact_content_daily_performance/month={m}/data_0.parquet",
            repo_type="dataset",
        )
    paths["dim_content"] = hf_hub_download(DS, "dim_content.parquet", repo_type="dataset")
    paths["dim_clients"] = hf_hub_download(DS, "dim_clients.parquet", repo_type="dataset")
    return paths


def load_population() -> pd.DataFrame:
    """Build the study population: one row per content page, 8 features + label.

    Returns the same 125,573-row frame as w03/w04/w05, with columns:
      content_hash_id, client_hash_id, content_type, search_volume,
      mar_impressions, mar_clicks, ctr_mar, mar_avg_position,
      momentum_feb_to_mar_pct, feb_impressions,
      has_clicks, has_feb_data, has_ga4, engagement_rate_mar,
      is_declining, content_created_date, is_published, is_deleted,
      pos_band, baseline_score (added later by models).
    """
    assert get_token(), (
        "No HF read token. Set HF_TOKEN in your environment or .env."
    )
    P = _get_parquet_paths()
    con = duckdb.connect()
    FACT3 = "[" + ",".join(f"'{P[m]}'" for m in MONTHS) + "]"

    raw = con.execute(f"""
        WITH f AS (
          SELECT content_hash_id,
                 MAX(client_hash_id) AS client_hash_id,
                 SUM(gsc_impressions) FILTER (WHERE month='2026-02' AND gsc_data_available IS TRUE) AS feb_impressions,
                 SUM(gsc_impressions) FILTER (WHERE month='2026-03' AND gsc_data_available IS TRUE) AS mar_impressions,
                 SUM(gsc_clicks)      FILTER (WHERE month='2026-03' AND gsc_data_available IS TRUE) AS mar_clicks,
                 SUM(gsc_sum_position) FILTER (WHERE month='2026-03' AND gsc_data_available IS TRUE
                                                     AND gsc_avg_position > 0) AS mar_sum_position,
                 SUM(gsc_impressions)  FILTER (WHERE month='2026-03' AND gsc_data_available IS TRUE
                                                     AND gsc_avg_position > 0) AS mar_impr_with_position,
                 SUM(ga4_engaged_sessions) FILTER (WHERE month='2026-03' AND ga4_data_available IS TRUE) AS mar_engaged,
                 SUM(ga4_sessions)         FILTER (WHERE month='2026-03' AND ga4_data_available IS TRUE) AS mar_sessions,
                 SUM(gsc_impressions) FILTER (WHERE month='2026-04' AND gsc_data_available IS TRUE) AS apr_impressions
          FROM read_parquet({FACT3})
          GROUP BY content_hash_id
        )
        SELECT f.*, d.content_created_date, d.content_type, d.search_volume,
               d.is_published, d.is_deleted
        FROM f LEFT JOIN read_parquet('{P["dim_content"]}') d USING (content_hash_id)
    """).df()

    for c in ["feb_impressions", "mar_impressions", "mar_clicks", "apr_impressions",
              "mar_engaged", "mar_sessions"]:
        raw[c] = raw[c].fillna(0)

    raw["mar_avg_position"] = np.where(
        raw.mar_impr_with_position.fillna(0) > 0,
        raw.mar_sum_position / raw.mar_impr_with_position, np.nan,
    )
    raw["ctr_mar"] = np.where(
        raw.mar_impressions > 0,
        raw.mar_clicks / raw.mar_impressions * 100, 0.0,
    )
    raw["momentum_feb_to_mar_pct"] = np.where(
        raw.feb_impressions > 0,
        (raw.mar_impressions - raw.feb_impressions)
        / raw.feb_impressions.replace(0, np.nan) * 100,
        np.nan,
    )
    raw["pos_band"] = pd.cut(
        raw.mar_avg_position,
        bins=[0, 3, 10, 20, 50, 1e9],
        labels=["<3", "3-10", "10-20", "20-50", "50+"],
    )
    raw["is_declining"] = (
        raw.apr_impressions < 0.8 * raw.mar_impressions
    ).astype(int)

    pop = raw[
        (raw.mar_impressions >= 30)
        & (raw.is_published == True)
        & (raw.is_deleted == False)
    ].copy()

    return pop


def query_page_detail(content_hash_id: str) -> dict | None:
    """Query all daily rows for a single content page across Feb+Mar+Apr.

    Returns a dict with monthly aggregates for the Investigator.
    """
    P = _get_parquet_paths()
    con = duckdb.connect()
    FACT3 = "[" + ",".join(f"'{P[m]}'" for m in MONTHS) + "]"

    row = con.execute(f"""
        SELECT content_hash_id,
               MAX(client_hash_id) AS client_hash_id,
               SUM(gsc_impressions) FILTER (WHERE month='2026-02' AND gsc_data_available IS TRUE) AS feb_impressions,
               SUM(gsc_impressions) FILTER (WHERE month='2026-03' AND gsc_data_available IS TRUE) AS mar_impressions,
               SUM(gsc_clicks)      FILTER (WHERE month='2026-03' AND gsc_data_available IS TRUE) AS mar_clicks,
               SUM(gsc_impressions) FILTER (WHERE month='2026-04' AND gsc_data_available IS TRUE) AS apr_impressions,
               SUM(ga4_engaged_sessions) FILTER (WHERE month='2026-03' AND ga4_data_available IS TRUE) AS mar_engaged,
               SUM(ga4_sessions)         FILTER (WHERE month='2026-03' AND ga4_data_available IS TRUE) AS mar_sessions,
               COUNT(*) FILTER (WHERE month='2026-03' AND gsc_data_available IS TRUE) AS mar_days_with_data
        FROM read_parquet({FACT3})
        WHERE content_hash_id = '{content_hash_id}'
        GROUP BY content_hash_id
    """).fetchone()

    if row is None:
        return {}

    cols = [
        "content_hash_id", "client_hash_id", "feb_impressions", "mar_impressions",
        "mar_clicks", "apr_impressions", "mar_engaged", "mar_sessions",
        "mar_days_with_data",
    ]
    d = dict(zip(cols, row))

    for c in ["feb_impressions", "mar_impressions", "mar_clicks",
              "apr_impressions", "mar_engaged", "mar_sessions"]:
        d[c] = d[c] or 0

    d["ctr_mar"] = (d["mar_clicks"] / d["mar_impressions"] * 100) if d["mar_impressions"] > 0 else 0.0
    d["momentum"] = (
        ((d["mar_impressions"] - d["feb_impressions"]) / d["feb_impressions"] * 100)
        if d["feb_impressions"] > 0 else None
    )
    d["is_declining"] = d["apr_impressions"] < 0.8 * d["mar_impressions"] if d["mar_impressions"] > 0 else None

    return d


def query_position_band_ctr() -> pd.Series:
    """Return mean CTR per position band, computed from March data.

    Used by the baseline rule and Investigator.
    """
    P = _get_parquet_paths()
    con = duckdb.connect()
    FACT3 = "[" + ",".join(f"'{P[m]}'" for m in MONTHS) + "]"

    df = con.execute(f"""
        WITH page AS (
          SELECT content_hash_id,
                 SUM(gsc_clicks)      FILTER (WHERE month='2026-03' AND gsc_data_available IS TRUE) AS mar_clicks,
                 SUM(gsc_impressions) FILTER (WHERE month='2026-03' AND gsc_data_available IS TRUE) AS mar_impressions,
                 SUM(gsc_sum_position) FILTER (WHERE month='2026-03' AND gsc_data_available IS TRUE
                                                     AND gsc_avg_position > 0) AS mar_sum_position,
                 SUM(gsc_impressions)  FILTER (WHERE month='2026-03' AND gsc_data_available IS TRUE
                                                     AND gsc_avg_position > 0) AS mar_impr_with_position
          FROM read_parquet({FACT3})
          GROUP BY content_hash_id
          HAVING mar_impressions >= 30
        )
        SELECT content_hash_id,
               CASE WHEN mar_impr_with_position > 0
                    THEN mar_sum_position / mar_impr_with_position
                    ELSE NULL END AS mar_avg_position,
               CASE WHEN mar_impressions > 0
                    THEN mar_clicks * 100.0 / mar_impressions
                    ELSE 0 END AS ctr_mar
        FROM page
    """).df()

    df["pos_band"] = pd.cut(
        df["mar_avg_position"],
        bins=[0, 3, 10, 20, 50, 1e9],
        labels=["<3", "3-10", "10-20", "20-50", "50+"],
    )
    return df.groupby("pos_band", observed=False)["ctr_mar"].mean()


def close():
    """Close any open DuckDB connections (cleanup)."""
    pass  # connections are local and GC'd
