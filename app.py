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
}

# Reference bins (Cr)
REF_BINS = {
    "AEPS Market Size": {
        "low":  (-np.inf, 5),
        "med":  (5, 25),
        "high": (25, np.inf),
    }
}

# Achievement bins (market share)
ACH_BINS = {
    "Market Share": {
        "low":  (-np.inf, 0.10),
        "med":  (0.10, 0.20),
        "high": (0.20, np.inf),
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
def fetch_district_data(month_year: str) -> pd.DataFrame:
    """
    Pull minimal columns needed to compute:
      - AEPS_MARKET_SIZE
      - Market Share = (AEPS_GTV / 1e6) / AEPS_MARKET_SIZE
    from sm_business_review_<mon_year> district table.

    IMPORTANT:
    - If your AEPS_MARKET_SIZE unit is not "Cr", adjust conversion below.
    """
    client = get_bq_client()
    month_str = pd.to_datetime(month_year).strftime("%b_%Y").lower()
    table_name = f"spicemoney-dwh.analytics_dwh.sm_business_review_{month_str}"

    q = f"""
        SELECT
          DISTRICT_NAME,
          STATE,
          AEPS_GTV,
          AEPS_MARKET_SIZE
        FROM `{table_name}`
    """
    df = client.query(q).result().to_dataframe()

    # --- conversions ---
    # If AEPS_GTV is in rupees and you want "Cr", you might do /1e7.
    # Your existing code uses /1e6 in the market share formula; keep consistent.
    # Market share definition here matches your existing pattern:
    # SM_AEPS_MARKET_SHARE = (AEPS_GTV / 1e6) / AEPS_MARKET_SIZE
    df["SM_AEPS_MARKET_SHARE"] = (df["AEPS_GTV"] / 1e7) / df["AEPS_MARKET_SIZE"]

    # If AEPS_MARKET_SIZE is not already "Cr", convert it here.
    # For now, we assume it matches your bin thresholds (5, 25).
    return df


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

    ref_bins = REF_BINS[reference_metric_ui]
    ach_bins = ACH_BINS[achievement_metric_ui]

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

    ref_bins_desc = {
        "high": "> 25 Cr",
        "med": "5 – 25 Cr",
        "low": "< 5 Cr",
    }

    ach_bins_desc = {
        "high": "> 20%",
        "med": "10 – 20%",
        "low": "< 10%",
    }

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

        month_map = build_month_options()
        month_label = st.selectbox("Select Month–Year", list(month_map.keys()), index=len(month_map) - 1)
        month_value = month_map[month_label]

        reference_metric_ui = st.selectbox("Reference Metric", ["AEPS Market Size"], index=0)
        achievement_metric_ui = st.selectbox("Achievement Metric", ["Market Share"], index=0)

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
