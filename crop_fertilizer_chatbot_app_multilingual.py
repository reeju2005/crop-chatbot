# crop_fertilizer_chatbot_app_multilingual.py
# Modified to add UI-level multilingual support (English, Hindi, Bengali primarily)
# Uses deep-translator for translations of dynamic answers and for translating UI labels.
# Keeps voice widget and gTTS audio options.
# Requirements (pip):
# pip install streamlit pandas numpy requests beautifulsoup4 deep-translator gTTS

from typing import Optional, Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import difflib
import os
from datetime import datetime
import streamlit.components.v1 as components
# translation & tts (using deep-translator)
from deep_translator import GoogleTranslator, exceptions as dt_exceptions
from gtts import gTTS
import io

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
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117 Safari/537.36'
    }
    results = []
    try:
        if commodity:
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
            for t in tables:
                cols = [c.lower() for c in t.columns.astype(str)]
                if any('modal' in c or 'price' in c for c in cols):
                    results.append(t)
        else:
            url = 'https://agmarknet.gov.in/PriceAndArrivals/CommodityDailyStateWise.aspx'
            r = session.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            tables = pd.read_html(r.text)
            for t in tables:
                cols = [c.lower() for c in t.columns.astype(str)]
                if any('modal' in c or 'price' in c for c in cols):
                    if 'state' in cols:
                        t = t[t['State'].astype(str).str.contains(state, case=False, na=False)]
                    results.append(t)
    except Exception as e:
        st.info(f"Live fetch from AGMARKNET failed: {e}")
        return pd.DataFrame()

    if not results:
        return pd.DataFrame()
    df_all = pd.concat(results, ignore_index=True, sort=False)
    df_all.columns = [str(c).strip() for c in df_all.columns]
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

# --------------------------- Translation helpers (deep-translator) ---------------------------

@st.cache_data
def get_language_map() -> Dict[str, str]:
    """
    Return dict of language_code -> language_name using deep-translator's supported languages.
    If that fails, fall back to a conservative built-in subset.
    """
    try:
        langs = GoogleTranslator.get_supported_languages(as_dict=True)  # type: ignore
        if isinstance(langs, dict) and langs:
            return {k: v.title() for k, v in langs.items()}
        langs_list = GoogleTranslator.get_supported_languages()  # type: ignore
        if isinstance(langs_list, list):
            raise Exception("deep-translator returned language list without codes; using fallback.")
    except Exception:
        return {
            'en': 'English',
            'hi': 'Hindi',
            'bn': 'Bengali',
            'mr': 'Marathi',
            'te': 'Telugu',
            'ta': 'Tamil',
            'gu': 'Gujarati',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'pa': 'Punjabi',
            'ur': 'Urdu'
        }
    return {
        'en': 'English',
        'hi': 'Hindi',
        'bn': 'Bengali',
        'mr': 'Marathi',
        'te': 'Telugu',
        'ta': 'Tamil',
        'gu': 'Gujarati',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'pa': 'Punjabi',
        'ur': 'Urdu'
    }

@st.cache_data
def translate_text(text: str, dest_lang_code: str) -> str:
    """Translate text using deep-translator's GoogleTranslator. dest_lang_code should be like 'bn' or 'hi' or 'en'."""
    if not text or dest_lang_code in ('en', ''):
        return text
    try:
        translated = GoogleTranslator(source='auto', target=dest_lang_code).translate(text)
        return translated
    except dt_exceptions.NotValidPayload as e:
        st.warning(f"Translation payload invalid: {e}")
        return text
    except dt_exceptions.LanguageNotSupportedException as e:
        st.warning(f"Language not supported for TTS/translation: {e}")
        return text
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text


def tts_audio_bytes(text: str, lang_code: str = "en") -> bytes:
    """Return MP3 bytes for text using gTTS."""
    try:
        tts = gTTS(text=text, lang=lang_code)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        st.warning(f"TTS failed: {e}")
        return b""

# --------------------------- UI translation caching helpers ---------------------------

# English originals for UI strings
_UI_ORIGINALS = {
    'page_title': "WB Crop & Fertilizer Chatbot (Translate + Voice)",
    'description': "This app answers fertilizer and price questions.\n- Use the **microphone widget** (below) to speak your question (best in Chrome/Edge).\n- Choose a **target language** using the Google Translate list to receive the answer in that language.\n- Optionally play the answer as audio (gTTS).",
    'price_source': "Price source:",
    'state_input': "State for live fetch (used only when Live selected)",
    'response_language': "Response language (translate using Google via deep-translator)",
    'tts_enable': "Provide audio of response (gTTS)",
    'examples_header': "Ask a question — examples:",
    'examples_list': "- fertilizer for rice\n- price of tomato\n- fertilizer coconut\n- price of wheat",
    'voice_instruction': "Voice input (start microphone, then copy recognized text into the question box):",
    'question_placeholder': "Your question (type crop & intent)",
    'get_answer': "Get Answer",
    'no_question_warning': "Please type a question or use the voice widget and paste into the question box.",
    'couldnt_detect_intent': "Couldn't detect intent automatically. I'll try both fertilizer and price answers.",
    'fert_subheader': 'Fertilizer Information (local CSV)',
    'price_subheader': 'Price (from local CSV)',
    'price_live_subheader': 'Price (Live: AGMARKNET - best effort)',
    'live_fetch_spinner': "Fetching live prices from AGMARKNET (may take a few seconds)...",
    'live_fetch_failed': "Live fetch failed or returned no data; falling back to local CSV.",
    'no_crop_warning': "Could not detect crop name. Please specify a crop in your question.",
}


def ui_text(key: str, dest_code: str) -> str:
    """Return translated UI text for `key` into dest_code (caches translations in session_state)."""
    if 'ui_cache' not in st.session_state:
        st.session_state['ui_cache'] = {}
    cache = st.session_state['ui_cache']
    cache_key = f"{key}::{dest_code}"
    if cache_key in cache:
        return cache[cache_key]
    base = _UI_ORIGINALS.get(key, key)
    if dest_code in ('en', ''):
        cache[cache_key] = base
        return base
    translated = translate_text(base, dest_code)
    cache[cache_key] = translated
    return translated

# Map app language code -> Web Speech API locale for speech input
_SPEECH_LOCALE = {
    'en': 'en-IN',
    'hi': 'hi-IN',
    'bn': 'bn-IN'
}

# --------------------------- Speech widget (browser) ---------------------------

def speech_input_widget(locale: str = 'en-IN'):
    """
    A small HTML widget using the browser's Web Speech API to capture speech and return text.
    The locale parameter sets recognition.lang (e.g. 'en-IN', 'hi-IN').
    """
    html = f"""
    <div>
      <button id="recBtn">🎤 Start/Stop Speech</button>
      <button id="clearBtn">✖ Clear</button>
      <div><small id="status">Status: idle</small></div>
      <textarea id="result" rows="4" style="width:100%;"></textarea>
      <script>
      const btn = document.getElementById('recBtn');
      const clear = document.getElementById('clearBtn');
      const result = document.getElementById('result');
      const status = document.getElementById('status');
      let recognizing = false;
      let recognition = null;
      if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {{
        status.innerText = 'Status: Web Speech API not supported in this browser.';
      }} else {{
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.lang = '{locale}'; // dynamic locale
        recognition.interimResults = true;
        recognition.continuous = true;
        recognition.onstart = () => {{ recognizing = true; status.innerText='Status: listening...'; btn.innerText='⏹ Stop'; }}
        recognition.onend = () => {{ recognizing = false; status.innerText='Status: idle'; btn.innerText='🎤 Start/Stop Speech'; }}
        recognition.onerror = (e) => {{ status.innerText='Error: ' + e.error; }}
        recognition.onresult = (event) => {{
          let interim = '';
          let final = '';
          for (let i = event.resultIndex; i < event.results.length; ++i) {{
            if (event.results[i].isFinal) {{
              final += event.results[i][0].transcript;
            }} else {{
              interim += event.results[i][0].transcript;
            }}
          }}
          result.value = (result.value ? result.value + ' ' : '') + final + interim;
        }};
      }}
      btn.onclick = () => {{
        if (!recognition) return;
        if (recognizing) {{
          recognition.stop();
        }} else {{
          try {{ recognition.start(); }} catch(e) {{ status.innerText='Status: '+e.message; }}
        }}
      }};
      clear.onclick = () => {{ result.value=''; }}
      </script>
    </div>
    """
    return components.html(html, height=220)

# --------------------------- Streamlit UI ---------------------------

def main():
    st.set_page_config(page_title="WB Crop & Fertilizer Chatbot (Translate + Voice)", layout='wide')

    # Custom CSS for larger fonts
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            font-size: 18px !important;  /* increase base font size */
        }
        h1, .stTitle {font-size: 38px !important;}
        h2 {font-size: 30px !important;}
        h3 {font-size: 24px !important;}
        </style>
        """,
        unsafe_allow_html=True
    )

    # language selector early so we can translate UI immediately
    lang_map = get_language_map()  # dict code->name
    # keep primary languages first
    primary = ['en', 'bn', 'hi']
    lang_items = []
    for code in primary:
        if code in lang_map:
            lang_items.append(f"{code} — {lang_map[code]}")
    # add remaining
    for code, name in lang_map.items():
        if code not in primary:
            lang_items.append(f"{code} — {name}")

    lang_choice = st.sidebar.selectbox("App language / ভাষা / भाषा (UI)", lang_items, index=0) or lang_items[0]
    dest_code = lang_choice.split(' — ')[0] if ' — ' in lang_choice else 'en'

    # helper to get translated UI strings
    def U(k):
        return ui_text(k, dest_code)

    st.title(U('page_title'))

    st.markdown(U('description'))

    col1, col2 = st.columns([2,1])
    with col2:
        source = st.radio(U('price_source'), ['Local CSV (default)', 'Live (Agmarknet, best-effort)'])
        state = st.text_input(U('state_input'), value='West Bengal')
        # language selector using deep-translator supported languages (for responses)
        lang_map = get_language_map()  # dict code->name
        lang_items = [f"{code} — {name}" for code, name in lang_map.items()]
        options = [f"en — English"] + sorted([li for li in lang_items if not li.startswith("en —")])
        lang_choice_resp = st.selectbox(U('response_language'), options, index=0)
        # extract code from selection
        resp_code = lang_choice_resp.split(' — ')[0] if ' — ' in lang_choice_resp else 'en'
        tts_enable = st.checkbox(U('tts_enable'), value=False)

    with col1:
        st.subheader(U('examples_header'))
        st.write(U('examples_list'))
        st.markdown(U('voice_instruction'))

        # choose speech locale for widget
        speech_locale = _SPEECH_LOCALE.get(dest_code, 'en-IN')
        recognized_text_area = speech_input_widget(locale=speech_locale)
        user_q = st.text_input(U('question_placeholder'))

    # load local data
    prices_df = load_local_prices()
    fert_df = load_local_fertilizers()

    # parse intent and crop from user_q (very simple rule-based)
    intent = None
    crop_term = None
    q = user_q.lower() if user_q else ""
    if q:
        if 'fert' in q or 'fertilizer' in q or 'manure' in q:
            intent = 'fertilizer'
        elif 'price' in q or 'rate' in q or 'msp' in q.replace(' ', ''):
            intent = 'price'
        # extract crop by removing common words
        words = user_q.replace('?', '').split()
        keywords = ['fertilizer','fertiliser','price','of','for','what','is','the']
        candidate = ' '.join([w for w in words if w.lower() not in keywords])
        crop_term = candidate.strip()
        if not crop_term:
            crop_term = words[-1] if words else ''

    if st.button(U('get_answer')):
        if not user_q:
            st.warning(U('no_question_warning'))
            return
        if not intent:
            st.info(U('couldnt_detect_intent'))

        full_answer_parts = []

        if intent == 'fertilizer' or intent is None:
            ans = get_fertilizer_info(crop_term if crop_term is not None else "", fert_df)
            st.subheader(U('fert_subheader'))
            translated = translate_text(ans, resp_code)
            st.text(translated)
            full_answer_parts.append(translated)

        if intent == 'price' or intent is None:
            if not crop_term:
                st.warning(U('no_crop_warning'))
            else:
                if source.startswith('Local'):
                    stats = get_price_stats_local(crop_term, prices_df)
                    st.subheader(U('price_subheader'))
                    formatted = format_price_stats(stats)
                    translated = translate_text(formatted, resp_code)
                    st.text(translated)
                    full_answer_parts.append(translated)
                else:
                    st.subheader(U('price_live_subheader'))
                    with st.spinner(U('live_fetch_spinner')):
                        df_live = fetch_agmarknet_state_prices(state=state, commodity=crop_term or None)
                    if df_live is None or df_live.empty:
                        st.info(U('live_fetch_failed'))
                        stats = get_price_stats_local(crop_term, prices_df)
                        formatted = format_price_stats(stats)
                        translated = translate_text(formatted, resp_code)
                        st.text(translated)
                        full_answer_parts.append(translated)
                    else:
                        st.write(df_live)
                        try:
                            sample = df_live.iloc[0].to_dict()
                            summary = "Live sample row: " + "; ".join([f"{k}: {v}" for k, v in sample.items() if k.lower() in ('market','commodity','modal price','modal','date') or isinstance(v, (str, int, float))][:6])
                            translated = translate_text(summary, resp_code)
                            st.text(translated)
                            full_answer_parts.append(translated)
                        except Exception:
                            pass

        # If TTS desired: assemble and play combined answer
        if tts_enable and full_answer_parts:
            combined = "\n\n".join(full_answer_parts)
            # choose language code for tts (gTTS expects certain codes; fallback to 'en' if unsupported)
            tts_code = resp_code if resp_code in ("en","hi","bn","mr","te","ta","gu","kn","ml","pa","ur") else "en"
            mp3_bytes = tts_audio_bytes(combined, lang_code=tts_code)
            if mp3_bytes:
                st.audio(mp3_bytes, format="audio/mp3")

if __name__ == '__main__':
    main()
