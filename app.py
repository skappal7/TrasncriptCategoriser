import io
import math
import time
import gc
import re
import logging
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Dict, List, Optional, Set

import streamlit as st
import polars as pl
import pandas as pd
import orjson
import requests
from streamlit_lottie import st_lottie
from flashtext import KeywordProcessor

# =========================
# ---- LOGGING & CONFIG ----
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Fast Transcript Classifier", page_icon="⚡", layout="wide")

# =========================
# ---- CSS / UI THEME ----
# =========================
st.markdown("""
<style>
    /* Main Background & Font */
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    
    /* Custom Card Styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        text-align: center;
    }
    .metric-card h3 {margin: 0; color: #555; font-size: 1rem; font-weight: 500;}
    .metric-card h2 {margin: 5px 0; color: #1f77b4; font-size: 1.8rem; font-weight: 700;}
    
    /* Headers */
    .big-title {font-size: 2.2rem; font-weight: 800; color: #1E1E1E; margin-bottom: 0px;}
    .sub-title {font-size: 1.1rem; color: #666; margin-bottom: 2rem; font-weight: 400;}
    
    /* Success Box */
    .success-box {
        padding: 1.5rem; 
        background-color: #f0fdf4; 
        border-radius: 12px; 
        border: 1px solid #bbf7d0;
        margin-bottom: 20px;
    }
    
    /* Custom Button */
    .stButton>button {
        background: linear-gradient(45deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 10px;
        height: 50px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# =========================
# ---- LOTTIE LOADER ----
# =========================
@st.cache_data(show_spinner=False)
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Professional Animations (No Balloons!)
LOTTIE_AI_SIDEBAR = "https://lottie.host/98f8045a-6945-4201-92bc-70a92e448455/53w9f7w8X7.json" # Tech Neural Network
LOTTIE_PROCESSING = "https://lottie.host/b0d77839-2c7c-473d-82d0-65e903f56e9c/K5S5q2y2F2.json" # Document Scanning
LOTTIE_SUCCESS = "https://lottie.host/7c7e9c91-9e8c-4867-8854-99214a1413df/v8s022g7YF.json"    # Clean Checkmark
LOTTIE_IDLE = "https://lottie.host/677b1029-4b13-40e1-bb5c-915008544d67/g7f6w3x4z5.json"       # Data Analysis

# =========================
# ---- CONSTANTS ----
# =========================
PARQUET_COMPRESSION = "zstd"
CHUNK_ROWS = 25_000
MAX_FILE_SIZE_MB = 1000
MAX_ROWS = 2_000_000
CLEAN_REGEX = re.compile(r"\[.*?\]|\d{4}-\d{2}-\d{2}.*?\+\d{4}|(?i)(Consumer:|AGENT:|CUSTOMER:|System:)|[\|\n]")

# =========================
# ---- CORE LOGIC ----
# =========================
def validate_file_size(file_bytes: bytes, max_mb: int = MAX_FILE_SIZE_MB) -> bool:
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        st.error(f"File too large ({size_mb:.1f}MB). Maximum allowed: {max_mb}MB")
        return False
    return True

def fast_clean_text(text: str) -> str:
    if not text: return ""
    text = str(text)
    text = CLEAN_REGEX.sub(" ", text)
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

@st.cache_data(show_spinner=False, max_entries=2)
def read_categories(file_bytes: bytes, filename: str) -> pl.DataFrame:
    try:
        suffix = Path(filename).suffix.lower()
        if suffix == ".csv":
            df = pl.read_csv(io.BytesIO(file_bytes), infer_schema_length=0)
        elif suffix in [".xls", ".xlsx"]:
            df = pl.from_pandas(pd.read_excel(io.BytesIO(file_bytes), dtype=str))
        else:
            return None
        return df.rename({col: col.strip() for col in df.columns})
    except Exception as e:
        st.error(f"Error reading categories: {e}")
        return None

@st.cache_data(show_spinner=False, max_entries=2)
def read_transcripts_streaming(file_bytes: bytes, filename: str) -> pl.DataFrame:
    try:
        suffix = Path(filename).suffix.lower()
        f_obj = io.BytesIO(file_bytes)
        if suffix == ".parquet":
            return pl.read_parquet(f_obj)
        elif suffix == ".csv":
            return pl.read_csv(f_obj, truncate_ragged_lines=False, ignore_errors=True, infer_schema_length=0)
        return None
    except Exception as e:
        st.error(f"Error reading transcripts: {e}")
        return None

@st.cache_resource(show_spinner=False)
def build_keyword_processor(cat_df: pl.DataFrame, phrase_col: str) -> Tuple[KeywordProcessor, int]:
    kp = KeywordProcessor(case_sensitive=False)
    phrase_map = defaultdict(list)
    rows = cat_df.to_dicts()
    count = 0
    for row in rows:
        phrase = row.get(phrase_col)
        if not phrase or str(phrase).strip() in ["", "nan", "None", "null"]: continue
        clean_phrase = str(phrase).strip().lower()
        if len(clean_phrase) < 2: continue
        
        path = {k: str(row.get(k, "")).strip() or None for k in ["L1", "L2", "L3", "L4"]}
        phrase_map[clean_phrase].append(path)
        count += 1
        
    for phrase, paths in phrase_map.items():
        kp.add_keyword(phrase, orjson.dumps({"p": phrase, "h": paths}).decode())
    return kp, count

def categorize_chunk(chunk_df: pl.DataFrame, kp: KeywordProcessor, id_col: str, text_col: str) -> List[Dict]:
    results = []
    for row in chunk_df.iter_rows(named=True):
        rid = row[id_col]
        clean_text = fast_clean_text(row[text_col])
        if not clean_text: continue
        try:
            hits = kp.extract_keywords(clean_text, span_info=False)
        except: hits = []
        if not hits: continue
        
        seen_paths = set()
        for hit_json in hits:
            data = orjson.loads(hit_json)
            for path in data.get("h", []):
                path_tuple = (path["L1"], path["L2"], path["L3"], path["L4"])
                if path_tuple not in seen_paths:
                    seen_paths.add(path_tuple)
                    results.append({
                        id_col: str(rid),
                        "matched_keyword": data.get("p"),
                        "L1": path["L1"], "L2": path["L2"], "L3": path["L3"], "L4": path["L4"]
                    })
    return results

# =========================
# ---- SIDEBAR SETUP ----
# =========================
with st.sidebar:
    # Sidebar Animation (Tech/AI)
    lottie_sidebar = load_lottieurl(LOTTIE_AI_SIDEBAR)
    if lottie_sidebar:
        st_lottie(lottie_sidebar, height=120, key="sidebar_anim")
        
    st.header("📂 Data Upload")
    
    cat_file = st.file_uploader("1. Categories (CSV/XLSX)", type=["csv", "xlsx"])
    cat_df, phrase_col = None, None
    if cat_file:
        cat_df = read_categories(cat_file.read(), cat_file.name)
        if cat_df is not None:
            st.success(f"✅ {cat_df.height:,} Rules Loaded")
            phrase_col = st.selectbox("Keyword Column", cat_df.columns)
            
    st.markdown("---")
    
    trn_file = st.file_uploader("2. Transcripts (CSV/Parquet)", type=["csv", "parquet"])
    trn_df, id_col, text_col = None, None, None
    if trn_file:
        bytes_data = trn_file.read()
        if validate_file_size(bytes_data):
            trn_df = read_transcripts_streaming(bytes_data, trn_file.name)
            if trn_df is not None:
                st.success(f"✅ {trn_df.height:,} Transcripts Loaded")
                cols = trn_df.columns
                id_col = st.selectbox("ID Column", cols)
                text_col = st.selectbox("Text Column", cols, index=len(cols)-1 if len(cols)>1 else 0)

    st.markdown("---")
    ready_to_run = cat_df is not None and trn_df is not None
    run_btn = st.button("🚀 Start Classification", type="primary", disabled=not ready_to_run, use_container_width=True)

# =========================
# ---- MAIN LAYOUT ----
# =========================

col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown('<div class="big-title">Transcript Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">High-Performance Hierarchical Classification Engine</div>', unsafe_allow_html=True)
with col_head2:
    # Small header animation if sidebar one fails or just for style
    pass 

if not run_btn and not ready_to_run:
    # IDLE STATE: Show cool analytics animation
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        lottie_idle = load_lottieurl(LOTTIE_IDLE)
        if lottie_idle:
            st_lottie(lottie_idle, height=300, key="idle")
        st.info("👈 Please upload your Categories and Transcripts in the sidebar to begin.")

# =========================
# ---- PROCESSING ----
# =========================
if run_btn and ready_to_run:
    start_time = time.time()
    
    # 1. VISUAL: Processing Animation
    processing_placeholder = st.empty()
    with processing_placeholder.container():
        c1, c2, c3 = st.columns([1,1,1])
        with c2:
            lottie_proc = load_lottieurl(LOTTIE_PROCESSING)
            if lottie_proc:
                st_lottie(lottie_proc, height=200, loop=True)
            st.markdown("<h3 style='text-align: center;'>Analyzing Transcripts...</h3>", unsafe_allow_html=True)
    
    # 2. Build Engine
    kp, rule_count = build_keyword_processor(cat_df, phrase_col)
    
    # 3. Process
    total_rows = trn_df.height
    num_chunks = math.ceil(total_rows / CHUNK_ROWS)
    all_results = []
    
    prog_bar = st.progress(0)
    
    for i in range(num_chunks):
        offset = i * CHUNK_ROWS
        length = min(CHUNK_ROWS, total_rows - offset)
        chunk = trn_df.slice(offset, length)
        all_results.extend(categorize_chunk(chunk, kp, id_col, text_col))
        prog_bar.progress((i + 1) / num_chunks)
        del chunk
        if i % 10 == 0: gc.collect()
            
    # Clear processing animation
    processing_placeholder.empty()
    prog_bar.empty()
    
    # =========================
    # ---- RESULTS ----
    # =========================
    process_time = time.time() - start_time
    
    if not all_results:
        st.warning("Analysis complete, but no matches were found.")
    else:
        result_df = pl.from_dicts(all_results, infer_schema_length=None)
        unique_ids = result_df.select(pl.col(id_col)).n_unique()
        coverage = (unique_ids / total_rows) * 100
        
        # SUCCESS HEADER
        st.markdown('<div class="success-box">Analysis Completed Successfully!</div>', unsafe_allow_html=True)

        # METRICS ROW (With Animation)
        m_col1, m_col2, m_col3, m_col4 = st.columns([1.5, 1, 1, 1])
        
        with m_col1:
            lottie_succ = load_lottieurl(LOTTIE_SUCCESS)
            if lottie_succ:
                st_lottie(lottie_succ, height=100, loop=False)
        
        with m_col2:
            st.markdown(f"""<div class="metric-card"><h3>Time Taken</h3><h2>{process_time:.1f}s</h2></div>""", unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""<div class="metric-card"><h3>Tagged Calls</h3><h2>{unique_ids:,}</h2></div>""", unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""<div class="metric-card"><h3>Coverage</h3><h2>{coverage:.1f}%</h2></div>""", unsafe_allow_html=True)

        st.markdown("---")
        
        # DATA PREVIEW & DOWNLOAD
        c_left, c_right = st.columns([2, 1])
        
        with c_left:
            st.subheader("📊 Category Distribution")
            if "L1" in result_df.columns:
                chart_data = result_df.group_by("L1").len().sort("len", descending=True).head(10)
                st.bar_chart(chart_data.to_pandas().set_index("L1"), color="#1f77b4")
                
        with c_right:
            st.subheader("📥 Download")
            st.write("Get your classified data:")
            
            buf = io.BytesIO()
            result_df.write_parquet(buf, compression=PARQUET_COMPRESSION)
            st.download_button(
                "⬇️ Download Parquet", 
                data=buf.getvalue(), 
                file_name="classified.parquet", 
                mime="application/octet-stream", 
                use_container_width=True
            )
            
            csv_data = result_df.head(500000).write_csv().encode('utf-8')
            st.download_button(
                "⬇️ Download CSV (Top 500k)", 
                data=csv_data, 
                file_name="classified.csv", 
                mime="text/csv", 
                use_container_width=True
            )

        with st.expander("🔎 View Raw Data Preview"):
            st.dataframe(result_df.head(100).to_pandas(), use_container_width=True)
