"""
Crop Prices & Fertilizer Chatbot (Streamlit)

- Loads two CSVs:
    - Crop_Prices_Dataset.csv
    - Fertilizer_Recommendations.csv
- Provides a chat-like UI where users can ask about:
    - Fertilizer recommendations for a crop
    - Price statistics (avg/min/max/latest) for a crop
- Supports English, Hindi, Bengali (with googletrans if available).
"""

import streamlit as st
import pandas as pd
import numpy as np
from dateutil import parser
import difflib

# optional translation
try:
    from googletrans import Translator
    translator = Translator()
    HAVE_TRANSLATOR = True
except Exception:
    translator = None
    HAVE_TRANSLATOR = False

# UI text
UI_TEXT = {
    'en': {
        'title': 'Crop Prices & Fertilizer Chatbot',
        'instruction': 'Ask about fertilizer recommendations or crop prices. Examples:\n- "Fertilizer for wheat"\n- "Price of rice in Karnal"\n- "Average price of tomato in 2024"',
        'no_data': 'No data found for the crop you asked about.',
        'loading': 'Loading datasets...'
    },
    'hi': {
        'title': 'फसल कीमतें और उर्वरक चैटबॉट',
        'instruction': 'उर्वरक सिफ़ारिशों या फसल की कीमतों के बारे में पूछें। उदाहरण:\n- "गेहूं के लिए उर्वरक"\n- "कर्नाल में चावल की कीमत"',
        'no_data': 'आपने जिस फसल के बारे में पूछा है, उसके लिए कोई डेटा नहीं मिला।',
        'loading': 'डेटासेट लोड हो रहे हैं...'
    },
    'bn': {
        'title': 'ফসল মূল্য ও সার চ্যাটবট',
        'instruction': 'সার সুপারিশ বা ফসলের দাম সম্পর্কে জিজ্ঞাসা করুন। উদাহরণ:\n- "গমের জন্য সার"\n- "কর্ণালে চালের দাম"',
        'no_data': 'আপনি যে ফসলটির সম্পর্কে প্রশ্ন করেছেন তার কোনো ডেটা মিলেনি।',
        'loading': 'ডেটাসেটগুলি লোড করা হচ্ছে...'
    }
}

# Paths to datasets
PRICES_PATH = 'Crop_Prices_Dataset.csv'
FERT_PATH = 'Fertilizer_Recommendations.csv'

@st.cache_data
def load_data():
    prices, fert = None, None
    try:
        prices = pd.read_csv(PRICES_PATH)
    except Exception:
        try:
            prices = pd.read_csv(PRICES_PATH, encoding='latin1')
        except Exception:
            prices = None
    try:
        fert = pd.read_csv(FERT_PATH)
    except Exception:
        try:
            fert = pd.read_csv(FERT_PATH, encoding='latin1')
        except Exception:
            fert = None
    return prices, fert

# Fuzzy match
def match_crop(crop_query, crop_list, cutoff=0.6):
    if not crop_query or crop_list is None or len(crop_list)==0:
        return None
    crop_query = str(crop_query).strip().lower()
    choices = [str(c).strip() for c in crop_list if pd.notna(c)]
    matches = difflib.get_close_matches(crop_query, choices, n=5, cutoff=cutoff)
    if matches:
        return matches[0]
    return None

# Fertilizer info
def get_fertilizer_info(crop_name, fert_df):
    if fert_df is None:
        return None
    if 'Crop' in fert_df.columns:
        match = match_crop(crop_name, fert_df['Crop'].unique())
        if not match:
            return None
        row = fert_df[fert_df['Crop'].astype(str).str.strip()==match].iloc[0]
        info_items = []
        for c in fert_df.columns:
            if c=='Crop': continue
            val = row.get(c, '')
            if pd.isna(val) or str(val).strip()=='':
                continue
            info_items.append(f"{c}: {val}")
        return "\n".join(info_items) if info_items else f"No detailed fertilizer recommendations found for {match}."
    return None

# Price info
def get_price_stats(crop_name, prices_df):
    if prices_df is None:
        return None
    df = prices_df.copy()
    crop_cols = [c for c in df.columns if c.lower() in ('crop','commodity','cropname','commodityname')]
    crop_col = crop_cols[0] if crop_cols else df.columns[0]
    match = match_crop(crop_name, df[crop_col].unique())
    if not match:
        return None
    df = df[df[crop_col].astype(str).str.strip()==match]
    price_cols = [c for c in df.columns if 'price' in c.lower() or 'modal' in c.lower()]
    price_col = price_cols[0] if price_cols else None
    if not price_col:
        return f"No price column found for {match}."
    series = df[price_col].dropna().astype(float)
    if series.empty:
        return f"No price records found for {match}."
    return {
        'crop': match,
        'mean': float(series.mean()),
        'min': float(series.min()),
        'max': float(series.max()),
        'latest': float(series.iloc[-1])
    }

# Translate response
def format_response(resp_text, lang='en'):
    if lang=='en' or resp_text is None:
        return resp_text
    if HAVE_TRANSLATOR:
        try:
            return translator.translate(resp_text, dest=lang).text
        except Exception:
            return resp_text
    return resp_text

# Intent detection
def detect_intent(user_text):
    t = user_text.lower()
    if 'fertil' in t or 'urea' in t:
        return 'fertilizer'
    if 'price' in t or 'rate' in t or 'कीमत' in t or 'দাম' in t:
        return 'price'
    return 'unknown'

# Streamlit app
st.set_page_config(page_title='Crop & Fert Chatbot', layout='wide')
lang = st.sidebar.selectbox('Language / भाषा / ভাষা', options=['en','hi','bn'], index=0)
texts = UI_TEXT.get(lang, UI_TEXT['en'])

st.title(texts['title'])
st.write(texts['instruction'])

prices_df, fert_df = load_data()

if prices_df is None and fert_df is None:
    st.error('Failed to load both datasets.')
else:
    with st.expander('Datasets (preview)'):
        if prices_df is not None:
            st.subheader('Prices dataset')
            st.dataframe(prices_df.head(5))
        if fert_df is not None:
            st.subheader('Fertilizer recommendations')
            st.dataframe(fert_df.head(5))

    st.subheader('Chat')
    user_input = st.text_input('Enter question / प्रश्न / প্রশ্ন')
    if st.button('Ask') and user_input.strip()!='':
        intent = detect_intent(user_input)
        crop_candidates = []
        if fert_df is not None:
            crop_candidates += [str(c).strip() for c in fert_df.iloc[:,0].unique() if pd.notna(c)]
        if prices_df is not None:
            crop_col = prices_df.columns[0]
            crop_candidates += [str(c).strip() for c in prices_df[crop_col].unique() if pd.notna(c)]
        matched_crop = match_crop(user_input, crop_candidates)

        response = ""
        if intent=='fertilizer':
            response = get_fertilizer_info(matched_crop, fert_df) if matched_crop else texts['no_data']
        elif intent=='price':
            stats = get_price_stats(matched_crop, prices_df) if matched_crop else None
            if isinstance(stats, dict):
                response = f"Price summary for {stats['crop']}: Avg {stats['mean']:.2f}, Min {stats['min']:.2f}, Max {stats['max']:.2f}, Latest {stats['latest']:.2f}"
            else:
                response = stats or texts['no_data']
        else:
            response = "Ask about 'fertilizer for <crop>' or 'price of <crop>'."
        out = format_response(response, lang)
        st.text_area('Response', value=out, height=200)
