Crop Prices & Fertilizer Chatbot - Package
=========================================

Contents:
- crop_fertilizer_chatbot_app.py  : Streamlit single-file app
- requirements.txt                : Python dependencies
- README.md                       : This file

Description:
This Streamlit app loads two CSV files expected at:
    /mnt/data/Crop_Prices_Dataset.csv
    /mnt/data/Fertilizer_Recommendations.csv

It provides a simple chat-like interface to ask about fertilizer recommendations and crop price summaries
(mean/min/max/latest). The app supports English, Hindi and Bengali UI labels and will attempt translation
via googletrans when available.

Quick start (Windows):
1. Ensure your CSVs are placed at the paths mentioned above or update the PRICES_PATH and FERT_PATH variables inside the app.
2. Create a virtual environment and activate it:
   python -m venv venv
   venv\Scripts\activate
3. Install dependencies:
   pip install -r requirements.txt
4. Run the app:
   streamlit run crop_fertilizer_chatbot_app.py
5. Open the displayed URL (typically http://localhost:8501) in your browser.

Notes & tips:
- If googletrans fails due to network or version mismatch, the app still returns English responses and UI labels.
- For better crop matching, provide clear crop names (e.g., 'wheat', 'rice', 'tomato').
- The app uses fuzzy matching; if it doesn't find an exact crop, try a slightly different wording.
- If the CSVs have different column names, edit the app to point to the correct crop/price columns.

Want more?
- I can add: a backend API (FastAPI) + React frontend, semantic search with embeddings for natural questions, or packaging as a Docker image.
