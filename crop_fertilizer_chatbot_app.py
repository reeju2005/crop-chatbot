from typing import Optional, Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import difflib
import os
from datetime import datetime

# --------------------------- Utilities ---------------------------

def normalize(s: Optional[str]) -> str:
    return (str(s).strip().lower() if s is not None else "").replace('\u200b','')


def fuzzy_match(query: str, choices: List[str], cutoff: float = 0.6) -> Optional[str]:
    """Return best fuzzy match from choices or None."""
    if not query or not choices:
        return None
    query = normalize(query)
    choices_norm = {normalize(c): c for c in choices}
    keys = list(choices_norm.keys())
    matches = difflib.get_close_matches(query, keys, n=1, cutoff=cutoff)
    if matches:
        return choices_norm[matches[0]]
    # try partial contains
    for k, orig in choices_norm.items():
        if query in k:
            return orig
    return None

# --------------------------- Data loading ---------------------------

@st.cache_data
def load_local_prices(path: str = "Crop_Prices_Dataset.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        st.warning(f"Local price CSV not found at {path}. Returning empty DataFrame.")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding='utf-8', low_memory=False)
    return df

@st.cache_data
def load_local_fertilizers(path: str = "Fertilizer_Recommendations.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        st.warning(f"Local fertilizer CSV not found at {path}. Returning empty DataFrame.")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding='utf-8', low_memory=False)
    return df

# --------------------------- Fertilizer functions ---------------------------

def get_fertilizer_info(crop: str, fert_df: pd.DataFrame) -> str:
    if fert_df is None or fert_df.empty:
        return "No fertilizer recommendations available (local CSV missing)."
    # try to detect crop column
    crop_cols = [c for c in fert_df.columns if 'crop' in c.lower()]
    crop_col = crop_cols[0] if crop_cols else fert_df.columns[0]
    available = fert_df[crop_col].astype(str).tolist()
    match = fuzzy_match(crop, available, cutoff=0.55)
    if not match:
        # try removing parenthesis or variety
        short = crop.split('(')[0].strip()
        match = fuzzy_match(short, available, cutoff=0.5)
        if not match:
            return f"No fertilizer recommendation found for '{crop}'. Try other crop names or check spelling."
    row = fert_df[fert_df[crop_col].astype(str).str.strip() == match].iloc[0]
    pieces = [f"Fertilizer recommendations for: {match}"]
    # collect common nutrient columns
    for nutrient in ['nitrogen', 'phosphorus', 'potassium', 'sulfur', 'n', 'p2o5', 'k2o', 's']:
        cols = [c for c in fert_df.columns if nutrient in c.lower()]
        if cols:
            pieces.append(f"{cols[0]}: {row[cols[0]]}")
    # Application notes
    for note_col in ['application', 'notes', 'application notes', 'special considerations']:
        cols = [c for c in fert_df.columns if note_col in c.lower()]
        if cols:
            pieces.append(f"{cols[0]}: {row[cols[0]]}")
    return "\n".join(pieces)

# --------------------------- Price functions (local) ---------------------------

def get_price_stats_local(crop: str, prices_df: pd.DataFrame) -> Any:
    if prices_df is None or prices_df.empty:
        return "Local price dataset not available."
    # detect crop column
    crop_cols = [c for c in prices_df.columns if 'crop' in c.lower() or 'commodity' in c.lower()]
    crop_col = crop_cols[0] if crop_cols else prices_df.columns[0]
    available = prices_df[crop_col].astype(str).tolist()
    match = fuzzy_match(crop, available, cutoff=0.55)
    if not match:
        # try short name
        short = crop.split('(')[0].strip()
        match = fuzzy_match(short, available, cutoff=0.5)
        if not match:
            return f"No local price records found for '{crop}'."
    df = prices_df[prices_df[crop_col].astype(str).str.strip() == match]
    # find numeric price columns
    price_cols = [c for c in df.columns if any(k in c.lower() for k in ('price', 'modal', 'msp', 'market'))]
    # prefer modal or market price
    chosen = None
    for pref in ['modal', 'market price', 'market', 'msp', 'price']:
        for c in price_cols:
            if pref in c.lower():
                chosen = c
                break
        if chosen:
            break
    if not chosen and price_cols:
        chosen = price_cols[0]
    if not chosen:
        return f"No price column found for '{match}'."

    import re
    def parse_price_value(x):
        if pd.isna(x):
            return None
        s = str(x).replace(',','')
        # find numbers (handles ranges like '2300-2400' -> [2300,2400])
        nums = re.findall(r"\d+(?:\.\d+)?", s)
        if not nums:
            return None
        vals = [float(n) for n in nums]
        # if the cell contained a range or multiple numbers, average them
        return float(sum(vals)/len(vals))

    series = df[chosen].apply(parse_price_value).dropna()
    if series.empty:
        return f"No price records found for {match}."
    return {
        'crop': match,
        'mean': float(series.mean()),
        'min': float(series.min()),
        'max': float(series.max()),
        'latest': float(series.iloc[-1])
    }

# --------------------------- Live fetch (Agmarknet best-effort) ---------------------------

def fetch_agmarknet_state_prices(state: str = 'West Bengal', commodity: Optional[str] = None, limit_markets: int = 10) -> pd.DataFrame:
    """
    Best-effort fetch of recent modal prices for a state from AGMARKNET.
    This function tries to query AGMARKNET search pages and parse tables. It may fail if the site blocks requests or changes layout.

    Returns a DataFrame with columns: State, District, Market, Commodity, Variety, Date, MinPrice, ModalPrice, MaxPrice
    """
    base = 'https://agmarknet.gov.in/SearchCmmMkt.aspx'
    params = {
        'Fr_Date': '',
        'To_Date': '',
        'DateFrom': '',
        'DateTo': '',
        'Tx_District': 0,
        'Tx_Market': 0,
        'Tx_Trend': 2,  # trend maybe
    }
    # AGMARKNET expects Tx_State param as state code (e.g., WB). We'll try both full name and code.
    # We'll attempt two queries: one with Tx_State=<state> and one with Tx_State=<shortcode>.

    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117 Safari/537.36'
    }
    results = []
    # Approach: iterate over commodity list pages if commodity is provided; otherwise try a state-wide report page
    try:
        if commodity:
            # try querying by commodity name by opening the CommodityDailyStateWise page with query params
            qparams = {
                'Tx_State': state,
                'Tx_CommodityHead': commodity,
                'Tx_Commodity': '',
                'Tx_District': 0,
                'Tx_Market': 0,
                'Tx_Trend': 2
            }
            r = session.get(base, params=qparams, headers=headers, timeout=20)
            r.raise_for_status()
            tables = pd.read_html(r.text)
            # heuristics: pick tables that have price columns
            for t in tables:
                cols = [c.lower() for c in t.columns.astype(str)]
                if any('modal' in c or 'price' in c for c in cols):
                    results.append(t)
        else:
            # try the commodity daily state wise page
            url = 'https://agmarknet.gov.in/PriceAndArrivals/CommodityDailyStateWise.aspx'
            r = session.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            tables = pd.read_html(r.text)
            # combine tables
            for t in tables:
                cols = [c.lower() for c in t.columns.astype(str)]
                if any('modal' in c or 'price' in c for c in cols):
                    # filter by state column if present
                    if 'state' in cols:
                        t = t[t['State'].astype(str).str.contains(state, case=False, na=False)]
                    results.append(t)
    except Exception as e:
        # fail gracefully - return empty DF
        st.info(f"Live fetch from AGMARKNET failed: {e}")
        return pd.DataFrame()

    if not results:
        return pd.DataFrame()
    df_all = pd.concat(results, ignore_index=True, sort=False)
    # normalize column names
    df_all.columns = [str(c).strip() for c in df_all.columns]
    # keep only useful columns
    keep = [c for c in df_all.columns if any(k in c.lower() for k in ('state', 'district', 'market', 'commodity', 'variety', 'modal', 'min', 'max', 'date'))]
    df_all = df_all[keep] if keep else df_all
    return df_all.head(limit_markets)

# --------------------------- Formatting ---------------------------

def format_price_stats(stats: Any) -> str:
    if isinstance(stats, str):
        return stats
    if isinstance(stats, dict):
        return (f"Price summary for {stats['crop']}: Mean {stats['mean']:.2f}, "
                f"Min {stats['min']:.2f}, Max {stats['max']:.2f}, Latest {stats['latest']:.2f}")
    return "No data"

# --------------------------- Streamlit UI ---------------------------

def main():
    st.set_page_config(page_title="WB Crop & Fertilizer Chatbot (Fixed)", layout='wide')

    # Custom CSS for larger fonts
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            font-size: 20px !important;  /* increase base font size */
        }
        h1, .stTitle {font-size: 42px !important;}
        h2 {font-size: 32px !important;}
        h3 {font-size: 26px !important;}
        </style>
        """,
        unsafe_allow_html=True
    )


    st.title("Crop Fertilizer & Price Assistant — West Bengal (fixed)")



    st.markdown("""
    This app answers two types of questions:
    1. Fertilizer recommendations for a crop (using the local `Fertilizer_Recommendations.csv`).
    2. Price statistics for a crop (using local `Crop_Prices_Dataset.csv`, or try "Live (Agmarknet)").

    *Note:* Live Agmarknet access is best-effort and may fail due to remote site restrictions. If live fetch fails, the app falls back to the local CSV.
    """)

    col1, col2 = st.columns([2,1])
    with col2:
        source = st.radio("Price source:", ['Local CSV (default)', 'Live (Agmarknet, best-effort)'])
        state = st.text_input("State for live fetch (used only when Live selected)", value='West Bengal')
        lang = st.selectbox("Language (response)", ['English', 'Bengali', 'Hindi'])

    with col1:
        st.subheader("Ask a question — examples:")
        st.write("- fertilizer for rice\n- price of tomato\n- fertilizer coconut\n- price of wheat")
        user_q = st.text_input("Your question (type crop & intent)")

    # load local data
    prices_df = load_local_prices()
    fert_df = load_local_fertilizers()

    # parse intent and crop from user_q (very simple rule-based)
    intent = None
    crop_term = None
    q = user_q.lower()
    if q:
        if 'fert' in q or 'fertilizer' in q or 'manure' in q:
            intent = 'fertilizer'
        elif 'price' in q or 'rate' in q or 'ms p' in q.replace(' ', ''):
            intent = 'price'
        # extract crop by removing common words
        # naive approach: find longest word sequence after keywords
        words = user_q.replace('?', '').split()
        keywords = ['fertilizer','fertiliser','price','of','for','what','is','the']
        candidate = ' '.join([w for w in words if w.lower() not in keywords])
        crop_term = candidate.strip()
        if not crop_term:
            # fallback: take last word
            crop_term = words[-1] if words else ''

    if st.button('Get Answer'):
        if not user_q:
            st.warning('Please type a question.')
            return
        if not intent:
            st.info("Couldn't detect intent automatically. I'll try both fertilizer and price answers.")

        if intent == 'fertilizer' or intent is None:
            ans = get_fertilizer_info(crop_term if crop_term is not None else "", fert_df)
            st.subheader('Fertilizer Information (local CSV)')
            st.text(ans)

        if intent == 'price' or intent is None:
            if not crop_term:
                st.warning('Could not detect crop name. Please specify a crop in your question.')
            else:
                if source.startswith('Local'):
                    stats = get_price_stats_local(crop_term, prices_df)
                    st.subheader('Price (from local CSV)')
                    st.text(format_price_stats(stats))
                else:
                    st.subheader('Price (Live: AGMARKNET - best effort)')
                    with st.spinner('Fetching live prices from AGMARKNET (may take a few seconds)...'):
                        df_live = fetch_agmarknet_state_prices(state=state, commodity=crop_term or None)
                    if df_live is None or df_live.empty:
                        st.info('Live fetch failed or returned no data; falling back to local CSV.')
                        stats = get_price_stats_local(crop_term, prices_df)
                        st.text(format_price_stats(stats))
                    else:
                        st.write(df_live)

if __name__ == '__main__':
    main()
