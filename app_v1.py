# app.py
import os
import json
import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
from branca.element import Template, MacroElement, Html

from google.cloud import bigquery
from google.oauth2 import service_account

try:
    import tomllib  # py311+
except Exception:
    import tomli as tomllib  # py310 fallback


# ---------------------------
# Creds (same pattern as your current file)
# ---------------------------
def _load_sa_from_toml_files():
    candidates = [
        os.path.join(os.environ.get("USERPROFILE", ""), ".streamlit", "secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
    ]
    for path in candidates:
        try:
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    data = tomllib.load(f)
                sa = data.get("gcp_service_account")
                if sa:
                    return sa, f"file:{path}"
        except Exception:
            pass
    return None, None


def get_bq_client():
    sa_info = None
    try:
        sa_info = st.secrets.get("gcp_service_account", None)
    except Exception:
        sa_info = None

    if sa_info:
        if isinstance(sa_info, str):
            sa_info = json.loads(sa_info)
        creds = service_account.Credentials.from_service_account_info(sa_info)
        return bigquery.Client(credentials=creds, project=creds.project_id)

    sa_info, _ = _load_sa_from_toml_files()
    if sa_info:
        creds = service_account.Credentials.from_service_account_info(sa_info)
        return bigquery.Client(credentials=creds, project=creds.project_id)

    raise RuntimeError(
        "No BigQuery credentials found. Put secrets.toml in HOME/CWD or use Streamlit secrets."
    )


# ---------------------------
# UI helpers
# ---------------------------

from branca.element import Template, MacroElement

def add_title(folium_map, title_text: str):
    title_html = f"""
    <div style="
        position: fixed;
        top: 15px;
        left: 50%;
        transform: translateX(-50%);
        background-color: white;
        padding: 10px 20px;
        font-size: 18px;
        font-weight: 700;
        z-index: 9999;
        border-radius: 6px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
        {title_text}
    </div>
    """

    title = MacroElement()
    title._template = Template(
        "{% macro html(this, kwargs) %}" + title_html + "{% endmacro %}"
    )
    folium_map.get_root().add_child(title)
    return folium_map



# def add_title(folium_map, title_text: str):
#     title_html = f"""
#     <div style="
#         position: fixed;
#         top: 15px;
#         left: 50%;
#         transform: translateX(-50%);
#         background-color: white;
#         padding: 10px 20px;
#         font-size: 16px;
#         font-weight: bold;
#         z-index: 9999;
#         border-radius: 5px;
#         box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
#     ">
#         {title_text}
#     </div>
#     """
#     folium_map.get_root().html.add_child(Html(title_html))
#     return folium_map


# def add_legend(folium_map, legend_title, color_map):
#     legend_html = f"""
#     <div style="
#         position: fixed;
#         top: 80px;
#         right: 50px;
#         width: 240px;
#         background-color: white;
#         z-index: 9999;
#         font-size: 13px;
#         padding: 10px;
#         border-radius: 5px;
#         box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
#     ">
#         <strong>{legend_title}</strong><br>
#     """
#     for label, color in color_map.items():
#         legend_html += (
#             f'<div style="display:flex;align-items:center;margin:5px 0;">'
#             f'<div style="width:15px;height:15px;background:{color};margin-right:8px;"></div>'
#             f'{label}</div>'
#         )
#     legend_html += "</div>"

#     legend = MacroElement()
#     legend._template = Template(f"{{% macro html(this, kwargs) %}}{legend_html}{{% endmacro %}}")
#     folium_map.get_root().add_child(legend)
#     return folium_map

def add_legend(
    folium_map,
    reference_metric_ui,
    achievement_metric_ui,
    ref_bins_desc,
    ach_bins_desc,
    
    ordered_legend_items
):
    legend_html = f"""
    <div style="
        position: fixed;
        top: 80px;
        right: 40px;
        width: 300px;
        background-color: white;
        z-index: 9999;
        font-size: 13px;
        padding: 12px;
        border-radius: 6px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">

    <strong>Reference Metric: {reference_metric_ui}</strong><br>
    <ul style="margin:6px 0 10px 15px; padding:0;">
        <li><b>High</b>: {ref_bins_desc['high']}</li>
        <li><b>Medium</b>: {ref_bins_desc['med']}</li>
        <li><b>Low</b>: {ref_bins_desc['low']}</li>
    </ul>

    <strong>Achievement Metric: {achievement_metric_ui}</strong><br>
    <ul style="margin:6px 0 10px 15px; padding:0;">
        <li><b>High</b>: {ach_bins_desc['high']}</li>
        <li><b>Medium</b>: {ach_bins_desc['med']}</li>
        <li><b>Low</b>: {ach_bins_desc['low']}</li>
    </ul>

    <strong>Classification Legend</strong><br>
    """

    for label, color in ordered_legend_items:
        legend_html += f"""
        <div style="display:flex; align-items:center; margin:4px 0;">
            <div style="width:14px; height:14px; background:{color};
                        margin-right:8px; border:1px solid #999;"></div>
            {label}
        </div>
        """

    legend_html += "</div>"

    legend = MacroElement()
    legend._template = Template(
        "{% macro html(this, kwargs) %}" + legend_html + "{% endmacro %}"
    )
    folium_map.get_root().add_child(legend)
    return folium_map


# ---------------------------
# Month options
# ---------------------------
def build_month_options(start_year=2024, start_month=4):
    today = datetime.date.today()
    end_year = today.year
    end_month = today.month - 1 if today.month > 1 else 12
    if today.month == 1:
        end_year -= 1

    options = {}
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        first_day = datetime.date(y, m, 1)
        label = first_day.strftime("%B %Y")
        value = first_day.strftime("%Y-%m-%d")
        options[label] = value
        if m == 12:
            m = 1
            y += 1
        else:
            m += 1
    return options


# ---------------------------
# Metric config (MVP: only AEPS market size + market share)
# ---------------------------
METRICS = {
    "AEPS Market Size": "AEPS_MARKET_SIZE",
    "Market Share": "SM_AEPS_MARKET_SHARE",
    "Trans Density (per 10k)": "TRANS_DENSITY_10K",
    "SP Density (per 10k)": "SP_DENSITY_10K",
    "CMS GTV": "CMS_GTV",
    "Monthly Visit Coverage" : "MONTHLY_VISIT_COVERAGE",
    "Partner Presence" : "PARTNER_PRESENCE",
    "Field Presence" : "FIELD_PRESENCE",
    "SP Winback (as % of Potential SP)" :  "SP_WINBACK_RATIO"
}



# # Reference bins (Cr)
# REF_BINS = {
#     "AEPS Market Size": {
#         "low":  (-np.inf, 5),
#         "med":  (5, 25),
#         "high": (25, np.inf),
#     }
# }

# # Achievement bins (market share)
# ACH_BINS = {
#     "Market Share": {
#         "low":  (-np.inf, 0.10),
#         "med":  (0.10, 0.20),
#         "high": (0.20, np.inf),
#     }
# }


METRIC_BINS = {
    "AEPS Market Size": {
        "low":  (-np.inf, 5),
        "med":  (5, 25),
        "high": (25, np.inf),
    },
    "Market Share": {
        "low":  (-np.inf, 0.10),
        "med":  (0.10, 0.20),
        "high": (0.20, np.inf),
    },
    "Trans Density (per 10k)": {
        "low":  (-np.inf, 2),
        "med":  (2, 5),
        "high": (5, np.inf),
    },
    "SP Density (per 10k)": {
        "low":  (-np.inf, 0.5),
        "med":  (0.5, 1),
        "high": (1, np.inf),
    },
    "CMS_GTV": {
    "low":  (-np.inf, 1),
    "med":  (1, 5),
    "high": (5, np.inf),
    },

    "Monthly Visit Coverage" : {
    "low":  (-np.inf, 4),     # <4%
    "med":  (4, 10),          # 4–10%
    "high": (10, np.inf),     # >10%
    },

    "Partner Presence" : {
    "low":  (-np.inf, 2),   # <=1  (because med starts at 2)
    "med":  (2, 11),        # 2–10 (hi exclusive, so use 11)
    "high": (11, np.inf),   # >=11 (i.e., >10)
    },

    "Field Presence" : {
    "low":  (-np.inf, 1),   # 0 goes here
    "med":  (1, 2),         # exactly 1 goes here (>=1 and <2)
    "high": (2, np.inf),    # >=2
    },

    "SP Winback (as % of Potential SP)": {
    "low":  (-np.inf, 10),      # <10%
    "med":  (10, 25),           # 10–25%
    "high": (25, np.inf),       # >=25% (see note below)
}

}



# 3x3 color matrix (reference rows: low/med/high ; achievement cols: low/med/high)
# Labels match your sheet intent; you can tweak hexes if you want exact shades.
MATRIX_COLOR = {
    ("low",  "low"):  ("Light Red",  "#ffb3b3"),
    ("low",  "med"):  ("Yellow",     "#ffff66"),
    ("low",  "high"): ("Light Green","#b7ffb7"),

    ("med",  "low"):  ("Red",        "#ff3333"),
    ("med",  "med"):  ("Mustard",    "#ffcc33"),
    ("med",  "high"): ("Green",      "#33cc33"),

    ("high", "low"):  ("Dark Red",   "#8b0000"),
    ("high", "med"):  ("Orange",     "#ff7f00"),
    ("high", "high"): ("Dark Green", "#006400"),
}

# Desired legend order (by class label)
LEGEND_ORDER = [
    "Light Red",
    "Red",
    "Dark Red",
    "Yellow",
    "Mustard",
    "Orange",
    "Light Green",
    "Green",
    "Dark Green",
    "No Data",
]

BIN_PRETTY = {"low": "Low", "med": "Med", "high": "High"}


def assign_bin(value: float, bin_def: dict) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "na"
    for k, (lo, hi) in bin_def.items():
        # low inclusive, high exclusive for med/low; high bin includes upper tail
        if k == "high":
            if value >= lo:
                return k
        else:
            if (value >= lo) and (value < hi):
                return k
    return "na"


# ---------------------------
# Data fetch (district-level only for now)
# ---------------------------

from google.cloud import bigquery

from google.cloud import bigquery

def fetch_district_data(month_year: str) -> pd.DataFrame:
    client = get_bq_client()

    month_str = pd.to_datetime(month_year).strftime("%b_%Y").lower()
    review_table = f"spicemoney-dwh.analytics_dwh.sm_business_review_{month_str}"

    # -----------------------------
    # 1) Base: market size + gtv (for market share)
    # -----------------------------
    q_base = f"""
        SELECT
          DISTRICT_NAME,
          STATE,
          AEPS_GTV,
          AEPS_MARKET_SIZE
        FROM `{review_table}`
    """
    df = client.query(q_base).result().to_dataframe()

    # Normalize district strings to reduce merge mismatches
    df["DISTRICT_NAME"] = df["DISTRICT_NAME"].astype(str).str.upper().str.strip()

    # Market share (keep your current logic)
    df["SM_AEPS_MARKET_SHARE"] = (df["AEPS_GTV"] / 1e7) / df["AEPS_MARKET_SIZE"]

    # -----------------------------
    # 2) Densities: Trans Density per 10k + SP Density per 10k
    # -----------------------------
    q_density = """
    WITH pop AS (
      SELECT District, SUM(Population) AS Population
      FROM `spicemoney-dwh.analytics_dwh.pincode_acqusition_plan`
      GROUP BY District
    ),

    trans_agents AS (
      SELECT
        final_district AS District,
        COUNT(DISTINCT t1.agent_id) AS num_trxn_SMAs
      FROM `spicemoney-dwh.analytics_dwh.csp_monthly_timeline_with_tu` t1
      LEFT JOIN `spicemoney-dwh.analytics_dwh.v_client_pincode` t2
        ON t1.agent_id = t2.retailer_id
      WHERE t1.month_year = @month_year
        AND t1.total_gtv_amt > 0
        AND final_district IS NOT NULL
      GROUP BY District
    ),

    sp_agents AS (
      SELECT
        final_district AS District,
        COUNT(DISTINCT t1.agent_id) AS num_sp_SMAs
      FROM `spicemoney-dwh.analytics_dwh.csp_monthly_timeline_with_tu` t1
      LEFT JOIN `spicemoney-dwh.analytics_dwh.v_client_pincode` t2
        ON t1.agent_id = t2.retailer_id
      WHERE t1.month_year = @month_year
        AND (t1.total_gtv_amt - t1.cms_gtv_success) > 250000
        AND final_district IS NOT NULL
      GROUP BY District
    )

    SELECT
      p.District AS District,
      COALESCE(ROUND(SAFE_DIVIDE(t.num_trxn_SMAs, p.Population) * 10000, 2), 0) AS TRANS_DENSITY_10K,
      COALESCE(ROUND(SAFE_DIVIDE(s.num_sp_SMAs, p.Population) * 10000, 2), 0) AS SP_DENSITY_10K
    FROM pop p
    LEFT JOIN trans_agents t USING (District)
    LEFT JOIN sp_agents s USING (District)
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("month_year", "DATE", month_year)
        ]
    )
    df_den = client.query(q_density, job_config=job_config).result().to_dataframe()
    df_den["District"] = df_den["District"].astype(str).str.upper().str.strip()

    # Merge densities into base
    df = df.merge(df_den, left_on="DISTRICT_NAME", right_on="District", how="left")
    df.drop(columns=["District"], inplace=True, errors="ignore")

    # -----------------------------
    # 3) CMS_GTV (Cr) at district level
    # -----------------------------
    q_cms = """
    SELECT
      final_district AS District,
      COALESCE(SUM(cms_gtv_success) / 1e7, 0) AS CMS_GTV
    FROM `spicemoney-dwh.analytics_dwh.csp_monthly_timeline_with_tu` t1
    LEFT JOIN `spicemoney-dwh.analytics_dwh.v_client_pincode` t2
      ON t1.agent_id = t2.retailer_id
    WHERE t1.month_year = @month_year
      AND final_district IS NOT NULL
    GROUP BY District
    """

    q_mvc = """
    WITH interaction_data AS (
    SELECT DISTINCT userid, sma_id
    FROM (
        SELECT * FROM (
        SELECT
            DATE(date) AS date,
            userid,
            sma_id,
            CASE WHEN interaction_mode IN ("onCall","Call") THEN "Call" ELSE "Meeting" END AS interaction_type
        FROM (
            SELECT DATE(date) AS date, userid, sma_id, interaction_mode
            FROM `spicemoney-dwh.impact_reports.dl_interaction_data_revamp`
            WHERE DATE_TRUNC(date(date), MONTH) = @month_year

            UNION ALL

            SELECT DATE(date) AS date, userid, sma_id, interaction_type
            FROM `spicemoney-dwh.impact_reports.dl_interaction_data`
            WHERE DATE_TRUNC(date(date), MONTH) = @month_year
        ) t1
        LEFT JOIN (
            SELECT DISTINCT agent_id, agent_name user_name, user_role
            FROM `impact.agent_login`
            WHERE user_role NOT IN ("testing","Technology & Research")
            UNION ALL
            SELECT DISTINCT agent_id, agent_name user_name, user_role
            FROM `spicemoney-dwh.impact.agent_login_inactive_user`
        ) t2
        ON t1.userid = t2.agent_id
        )
        WHERE interaction_type = "Meeting"

        UNION ALL

        SELECT * FROM (
        SELECT DATE(date) AS date, partner_id AS userid, sma_id, interaction_type
        FROM `spicemoney-dwh.impact_reports.partner_interaction_data`
        WHERE DATE_TRUNC(date(date), MONTH) = @month_year
        )
        WHERE interaction_type = "Meeting"
    )
    ),

    location_data AS (
    SELECT DISTINCT retailer_id, final_district AS District
    FROM `spicemoney-dwh.analytics_dwh.v_client_pincode`
    ),

    transacting_sma_data AS (
    SELECT District, COUNT(DISTINCT agent_id) AS Trxn_SMAs
    FROM (
        SELECT agent_id, final_district AS District
        FROM `spicemoney-dwh.analytics_dwh.csp_monthly_timeline_with_tu` t1
        LEFT JOIN `spicemoney-dwh.analytics_dwh.v_client_pincode` t2
        ON t1.agent_id = t2.retailer_id
        WHERE t1.month_year = @month_year
        AND t1.total_gtv_amt > 0
        AND final_district IS NOT NULL
    )
    GROUP BY District
    ),

    visited_data AS (
    SELECT
        District,
        COALESCE(COUNT(DISTINCT sma_id), 0) AS VISITED_SMAs
    FROM interaction_data t1
    LEFT JOIN location_data t2
        ON t1.sma_id = t2.retailer_id
    GROUP BY District
    )

    SELECT
    v.District,
    ROUND(SAFE_DIVIDE(VISITED_SMAs, Trxn_SMAs) * 100, 1) AS MONTHLY_VISIT_COVERAGE
    FROM visited_data v
    LEFT JOIN transacting_sma_data t
    ON v.District = t.District
    """

    q_partner = """
    SELECT
    district AS District,
    COALESCE(COUNT(DISTINCT distributor_id), 0) AS PARTNER_PRESENCE
    FROM (
    SELECT distributor_id, district
    FROM `spicemoney-dwh.impact.distributor_master`
    WHERE distributor_id IN (
        SELECT client_id
        FROM `prod_dwh.client_wallet`
        WHERE status = 'active'
    )
    AND distributor_id IN (
        SELECT DISTINCT retailer_id
        FROM `prod_dwh.client_details`
        WHERE client_type IN ('CME', 'distributor')
    )
    AND district IS NOT NULL
    AND distributor_id IN (
        SELECT DISTINCT CAST(Distributor_ID AS STRING) AS partner_id
        FROM `spicemoney-dwh.analytics_dwh.partner_monthly_commission`
        WHERE month = @month_year
        AND TOTAL_COMMISSION > 0
    )
    )
    GROUP BY District
    """

    q_field = """
    SELECT
    District,
    COUNT(DISTINCT user_id) AS FIELD_PRESENCE
    FROM (
    SELECT
        t2.user_id,
        t1.district AS District
    FROM (
        SELECT DISTINCT
        district,
        COALESCE(dl_email, bl_email) AS user_email
        FROM `spicemoney-dwh.analytics_dwh.sales_hierarchy_mapping_native_logs `
        WHERE DATE_TRUNC(DATE(insert_date), MONTH) = @month_year
    ) AS t1
    LEFT JOIN (
        SELECT user_id, email_id
        FROM `analytics_dwh.impact_user_state_mapping`
    ) AS t2
    ON t1.user_email = t2.email_id
    )
    GROUP BY District
    """

    q_sp_winback = """
    WITH params AS (
    SELECT
        @month_year AS focus_month,
        DATE_SUB(@month_year, INTERVAL 1 MONTH) AS prev_month,
        DATE_SUB(@month_year, INTERVAL 2 MONTH) AS prev2_month,
        DATE_SUB(@month_year, INTERVAL 14 MONTH) AS hist_start
    ),

    sp_winback_data AS (
    SELECT
        District,
        COUNT(DISTINCT agent_id) AS SP_WINBACK
    FROM (
        SELECT c.agent_id, d.final_district AS District
        FROM (
        SELECT DISTINCT agent_id
        FROM (
            SELECT
            agent_id,

            COALESCE(COUNT(DISTINCT CASE
                WHEN month_year <= (SELECT prev2_month FROM params)
                AND month_year >= (SELECT hist_start FROM params)
                AND GTV > 250000
                THEN month_year END), 0) AS _2_5_LAC_hist,

            COALESCE(COUNT(DISTINCT CASE
                WHEN month_year = (SELECT prev_month FROM params)
                AND GTV > 250000
                THEN month_year END), 0) AS Prev_month,

            COALESCE(COUNT(DISTINCT CASE
                WHEN month_year = (SELECT focus_month FROM params)
                AND GTV > 250000
                THEN month_year END), 0) AS Curr_month

            FROM (
            SELECT
                agent_id,
                month_year,
                total_gtv_amt - cms_gtv_success AS GTV
            FROM `spicemoney-dwh.analytics_dwh.csp_monthly_timeline_with_tu`
            WHERE month_year <= (SELECT focus_month FROM params)
            )
            GROUP BY agent_id
        )
        WHERE _2_5_LAC_hist > 0 AND Prev_month = 0 AND Curr_month = 1
        ) c
        LEFT JOIN `spicemoney-dwh.analytics_dwh.v_client_pincode` d
        ON c.agent_id = d.retailer_id
    )
    WHERE District IS NOT NULL
    GROUP BY District
    ),

    potential_sp_data AS (
    SELECT
        District,
        COUNT(DISTINCT agent_id) AS POTENTIAL_SPs
    FROM (
        SELECT c.agent_id, d.final_district AS District
        FROM (
        SELECT agent_id
        FROM (
            SELECT
            agent_id,
            SUM(CASE
                WHEN month_year BETWEEN DATE_SUB((SELECT focus_month FROM params), INTERVAL 6 MONTH)
                                AND DATE_SUB((SELECT focus_month FROM params), INTERVAL 1 MONTH)
                AND (total_gtv_amt - cms_gtv_success) >= 250000
                THEN 1 ELSE 0 END
            ) AS two_five_lac_history,

            SUM(CASE
                WHEN month_year = (SELECT focus_month FROM params)
                AND (total_gtv_amt - cms_gtv_success) < 250000
                AND (total_gtv_amt - cms_gtv_success) != 0
                THEN 1 ELSE 0 END
            ) AS focus_month_gtv

            FROM `spicemoney-dwh.analytics_dwh.csp_monthly_timeline_with_tu`
            WHERE month_year <= (SELECT focus_month FROM params)
            GROUP BY agent_id
        )
        WHERE two_five_lac_history >= 1 AND focus_month_gtv = 1
        ) c
        LEFT JOIN `spicemoney-dwh.analytics_dwh.v_client_pincode` d
        ON c.agent_id = d.retailer_id
    )
    WHERE District IS NOT NULL
    GROUP BY District
    )

    SELECT
    a.District,
    a.SP_WINBACK,
    b.POTENTIAL_SPs,
    ROUND(SAFE_DIVIDE(a.SP_WINBACK, b.POTENTIAL_SPs) * 100, 1) AS SP_WINBACK_RATIO
    FROM sp_winback_data a
    JOIN potential_sp_data b
    ON a.District = b.District
    """

    df_spw = client.query(q_sp_winback, job_config=job_config).result().to_dataframe()
    df_spw["District"] = df_spw["District"].astype(str).str.upper().str.strip()

    df = df.merge(df_spw[["District", "SP_WINBACK_RATIO"]], left_on="DISTRICT_NAME", right_on="District", how="left")
    df.drop(columns=["District"], inplace=True, errors="ignore")
    df["SP_WINBACK_RATIO"] = df["SP_WINBACK_RATIO"].fillna(0)

    df_field = client.query(q_field, job_config=job_config).result().to_dataframe()

    # Normalize district naming for merge consistency
    df_field["District"] = df_field["District"].astype(str).str.upper().str.strip()

    df = df.merge(df_field, left_on="DISTRICT_NAME", right_on="District", how="left")
    df.drop(columns=["District"], inplace=True, errors="ignore")
    df["FIELD_PRESENCE"] = df["FIELD_PRESENCE"].fillna(0)


    df_partner = client.query(q_partner, job_config=job_config).result().to_dataframe()
    df_partner["District"] = df_partner["District"].astype(str).str.upper().str.strip()

    df = df.merge(df_partner, left_on="DISTRICT_NAME", right_on="District", how="left")
    df.drop(columns=["District"], inplace=True, errors="ignore")
    df["PARTNER_PRESENCE"] = df["PARTNER_PRESENCE"].fillna(0)


    df_mvc = client.query(q_mvc, job_config=job_config).result().to_dataframe()
    df_mvc["District"] = df_mvc["District"].astype(str).str.upper().str.strip()

    df = df.merge(df_mvc, left_on="DISTRICT_NAME", right_on="District", how="left")
    df.drop(columns=["District"], inplace=True, errors="ignore")
    df["MONTHLY_VISIT_COVERAGE"] = df["MONTHLY_VISIT_COVERAGE"].fillna(0)


    df_cms = client.query(q_cms, job_config=job_config).result().to_dataframe()
    df_cms["District"] = df_cms["District"].astype(str).str.upper().str.strip()

    # Merge CMS_GTV into base
    df = df.merge(df_cms, left_on="DISTRICT_NAME", right_on="District", how="left")
    df.drop(columns=["District"], inplace=True, errors="ignore")

    # Fill missing KPI values as 0 to avoid "na" bins due to nulls
    # for col in ["TRANS_DENSITY_10K", "SP_DENSITY_10K", "CMS_GTV"]:
    #     if col in df.columns:
    #         df[col] = df[col].fillna(0)
    for col in ["TRANS_DENSITY_10K", "SP_DENSITY_10K", "CMS_GTV", "MONTHLY_VISIT_COVERAGE",  "PARTNER_PRESENCE", "FIELD_PRESENCE",
    "SP_WINBACK_RATIO"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)


    return df




# def fetch_district_data(month_year: str) -> pd.DataFrame:
#     """
#     Pull minimal columns needed to compute:
#       - AEPS_MARKET_SIZE
#       - Market Share = (AEPS_GTV / 1e6) / AEPS_MARKET_SIZE
#     from sm_business_review_<mon_year> district table.

#     IMPORTANT:
#     - If your AEPS_MARKET_SIZE unit is not "Cr", adjust conversion below.
#     """
#     client = get_bq_client()
#     month_str = pd.to_datetime(month_year).strftime("%b_%Y").lower()
#     table_name = f"spicemoney-dwh.analytics_dwh.sm_business_review_{month_str}"

#     q = f"""
#         SELECT
#           DISTRICT_NAME,
#           STATE,
#           AEPS_GTV,
#           AEPS_MARKET_SIZE
#         FROM `{table_name}`
#     """
#     df = client.query(q).result().to_dataframe()

#     # --- conversions ---
#     # If AEPS_GTV is in rupees and you want "Cr", you might do /1e7.
#     # Your existing code uses /1e6 in the market share formula; keep consistent.
#     # Market share definition here matches your existing pattern:
#     # SM_AEPS_MARKET_SHARE = (AEPS_GTV / 1e6) / AEPS_MARKET_SIZE
#     df["SM_AEPS_MARKET_SHARE"] = (df["AEPS_GTV"] / 1e7) / df["AEPS_MARKET_SIZE"]

#     # If AEPS_MARKET_SIZE is not already "Cr", convert it here.
#     # For now, we assume it matches your bin thresholds (5, 25).
#     return df


# ---------------------------
# Map generation (cross metric classification)
# ---------------------------
def generate_geo_spatial_map(
    month_year: str,
    geography: str,
    state: str,
    reference_metric_ui: str,
    achievement_metric_ui: str,
):
    # Shapefile
    shape_file = "India_District_Boundaries.shp"
    gdf = gpd.read_file(shape_file)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    gdf = gdf.to_crs(epsg=4326)

    # Data
    df = fetch_district_data(month_year)
    print("****************", df.head(1))
    print("****************", gdf.head(1))
    # Merge
    merged = gdf.merge(df, left_on="District", right_on="DISTRICT_NAME", how="left")
    print("****************", merged.head(1))
    if (geography == "State")  and (state != "All States"):
        merged = merged[merged["STATE_x"] == state]

    
    # Compute bins
    ref_col = METRICS[reference_metric_ui]
    ach_col = METRICS[achievement_metric_ui]

    # ref_bins = REF_BINS[reference_metric_ui]
    # ach_bins = ACH_BINS[achievement_metric_ui]

    ref_bins = METRIC_BINS[reference_metric_ui]
    ach_bins = METRIC_BINS[achievement_metric_ui]


    merged["ref_bin"] = merged[ref_col].apply(lambda x: assign_bin(x, ref_bins))
    merged["ach_bin"] = merged[ach_col].apply(lambda x: assign_bin(x, ach_bins))

    def class_and_color(row):
        rb, ab = row["ref_bin"], row["ach_bin"]
        if rb == "na" or ab == "na":
            return ("No Data", "#d9d9d9")
        label, color = MATRIX_COLOR[(rb, ab)]
        # Add a compact label that helps in tooltip/legend
        pretty = f"{label} (Ref={rb.title()}, Ach={ab.title()})"
        return (pretty, color)

    merged[["class_label", "fill_color"]] = merged.apply(
        lambda r: pd.Series(class_and_color(r)), axis=1
    )

    # Map center
    center = [merged.geometry.centroid.y.mean(), merged.geometry.centroid.x.mean()]
    folium_map = folium.Map(location=center, zoom_start=6, tiles="cartodb positron")

    # Style
    def style_fn(feature):
        c = feature["properties"].get("fill_color", "#d9d9d9")
        return {"fillColor": c, "color": "black", "weight": 1, "fillOpacity": 1}

    tooltip_fields = ["DISTRICT_NAME", "STATE_x", ref_col, ach_col, "class_label"]
    tooltip_aliases = ["District:", "State:", f"{reference_metric_ui}:", f"{achievement_metric_ui}:", "Class:"]

    folium.GeoJson(
        merged,
        name="GeoSpatial Map",
        style_function=style_fn,
        tooltip=GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=False,
            labels=True,
            style="background-color: white; color: black; font-weight: bold;",
        ),
    ).add_to(folium_map)

    # Legend: only show the 9 matrix classes + No Data
    # legend_map = {v[0]: v[1] for _, v in MATRIX_COLOR.items()}
    # legend_map["No Data"] = "#d9d9d9"

    # Build label -> (color, ref_bin, ach_bin)
    class_meta = {}
    for (ref_bin, ach_bin), (label, color) in MATRIX_COLOR.items():
        class_meta[label] = (color, ref_bin, ach_bin)

    # Now create ordered legend entries with bin info appended
    ordered_legend_items = []
    for label in LEGEND_ORDER:
        if label == "No Data":
            ordered_legend_items.append(("No Data", "#d9d9d9"))
            continue

        if label in class_meta:
            color, ref_bin, ach_bin = class_meta[label]
            pretty = f"{label} (Ref={BIN_PRETTY[ref_bin]}, Ach={BIN_PRETTY[ach_bin]})"
            ordered_legend_items.append((pretty, color))


    # title = (
    #     f"{reference_metric_ui} (Reference) × {achievement_metric_ui} (Achievement)"
    #     + (f" — {state}" if geography == "State" else " — National")
    #     + f" — {pd.to_datetime(month_year).strftime('%b %Y')}"
    # )

    scope = "National" if (state == "All States" or geography == "National") else state
    title = f"{reference_metric_ui} (Reference) × {achievement_metric_ui} (Achievement) — {scope} — {pd.to_datetime(month_year).strftime('%b %Y')}"


    folium_map = add_title(folium_map, title)

    BIN_DESC = {
    "AEPS Market Size": {"high": "> 25 Cr", "med": "5 – 25 Cr", "low": "< 5 Cr"},
    "Market Share": {"high": "> 20%", "med": "10 – 20%", "low": "< 10%"},
    "Trans Density (per 10k)": {"high": "> 5", "med": "2 – 5", "low": "< 2"},
    "SP Density (per 10k)": {"high": "> 1", "med": "0.5 – 1", "low": "< 0.5"},
    "CMS_GTV": {"high": "> 5 Cr", "med": "1 – 5 Cr", "low": "< 1 Cr"},
    "Monthly Visit Coverage" : {
    "high": "> 10%",
    "med": "4 – 10%",
    "low": "< 4%",
    },

    "Partner Presence" : {
    "high": "> 10",
    "med": "2 – 10",
    "low": "≤ 1",
        },

    "Field Presence" : {
    "high": "≥ 2",
    "med": "1",
    "low": "0",
    },

    "SP Winback (as % of Potential SP)": {
    "high": "> 25%",
    "med": "10 – 25%",
    "low": "< 10%",
    }
    }

    ref_bins_desc = BIN_DESC[reference_metric_ui]
    ach_bins_desc = BIN_DESC[achievement_metric_ui]

    # ref_bins_desc = {
    #     "high": "> 25 Cr",
    #     "med": "5 – 25 Cr",
    #     "low": "< 5 Cr",
    # }

    # ach_bins_desc = {
    #     "high": "> 20%",
    #     "med": "10 – 20%",
    #     "low": "< 10%",
    # }

    folium_map = add_legend(
        folium_map,
        reference_metric_ui,
        achievement_metric_ui,
        ref_bins_desc,
        ach_bins_desc,
        
        ordered_legend_items
    )


    # folium_map = add_legend(folium_map, "Classification Legend", legend_map)

    # file_name = f"GEO_MAP_{geography}_{state if geography=='State' else 'National'}_{reference_metric_ui}_x_{achievement_metric_ui}_{month_year}.html"
    scope_slug = "National" if (state == "All States" or geography == "National") else state
    file_name = f"GEO_MAP_{scope_slug}_{reference_metric_ui}_x_{achievement_metric_ui}_{month_year}.html"

    return folium_map, file_name


# ---------------------------
# Streamlit App
# ---------------------------
def _init_session_state():
    defaults = {
        "last_map_html": None,
        "last_map_file_name": None,
        "map_file_bytes": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    _init_session_state()

    st.set_page_config(page_title="District GeoSpatial Map Generator", layout="wide")

    with st.sidebar:
        st.header("Configuration")

        geography = st.selectbox("Select Geography", ["State", "National"], index=0)

        states = ["ANDHRA PRADESH", "KERALA", 
            "TAMIL NADU", "UTTAR PRADESH", "WEST BENGAL", "MADHYA PRADESH",
            "MAHARASHTRA", "KARNATAKA", "ODISHA", "CHHATTISGARH",
            "JHARKHAND", "PUNJAB", "DELHI_NCR", "HARYANA","BIHAR"
        ]
        state_list = ["All States"] + sorted(states)
        state = st.selectbox("Select State", state_list, index=0)

        # month_map = build_month_options()
        # month_label = st.selectbox("Select Month–Year", list(month_map.keys()), index=len(month_map) - 1)
        # month_value = month_map[month_label]

        month_map = build_month_options()

        month_labels_desc = list(month_map.keys())[::-1]  # reverse order

        month_label = st.selectbox(
            "Select Month–Year",
            month_labels_desc,
            index=0  # latest month selected by default
        )

        month_value = month_map[month_label]


        # reference_metric_ui = st.selectbox("Reference Metric", ["AEPS Market Size"], index=0)
        # achievement_metric_ui = st.selectbox("Achievement Metric", ["Market Share"], index=0)

        all_kpis = list(METRICS.keys())

        reference_metric_ui = st.selectbox("Reference Metric", all_kpis, index=0)
        achievement_metric_ui = st.selectbox("Achievement Metric", all_kpis, index=1 if len(all_kpis) > 1 else 0)


        generate_clicked = st.button("Generate Map", type="primary", use_container_width=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🗺️ District GeoSpatial Map Generator")

    with col2:
        map_ready = st.session_state.map_file_bytes is not None
        if map_ready:
            st.download_button(
                label="⬇️ Download HTML Map",
                data=st.session_state.map_file_bytes,
                file_name=st.session_state.last_map_file_name or "map.html",
                mime="text/html",
                use_container_width=True,
            )
        else:
            st.download_button(
                label="⬇️ Download HTML Map",
                data=b"",
                file_name="map.html",
                mime="text/html",
                disabled=True,
                use_container_width=True,
            )

    if generate_clicked:
        with st.spinner("Generating geo spatial map…"):
            try:
                folium_map, file_name = generate_geo_spatial_map(
                    month_year=month_value,
                    geography=geography,
                    state=state,
                    reference_metric_ui=reference_metric_ui,
                    achievement_metric_ui=achievement_metric_ui,
                )

                map_html = folium_map._repr_html_()
                st.session_state.last_map_html = map_html
                st.session_state.last_map_file_name = file_name
                st.session_state.map_file_bytes = map_html.encode("utf-8")

            except Exception as e:
                st.session_state.last_map_html = None
                st.session_state.last_map_file_name = None
                st.session_state.map_file_bytes = None
                st.error(f"❌ Error while generating map: {e}")

    if st.session_state.last_map_html:
        st_html(st.session_state.last_map_html, height=720, scrolling=False)
    else:
        st.info("Pick Reference + Achievement metrics, month, geography, then click **Generate Map**.")


if __name__ == "__main__":
    main()
