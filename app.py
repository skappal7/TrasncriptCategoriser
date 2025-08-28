# app.py
import io, os, tempfile, math, orjson, time, gc, re
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Dict, List, Optional
import logging

import streamlit as st
import polars as pl
import pandas as pd
from flashtext import KeywordProcessor

# =========================
# ---- LOGGING SETUP ----
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# ---- UI / THEME CSS  ----
# =========================
st.set_page_config(page_title="Transcript Categorizer", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.big-title {font-size: 1.6rem; font-weight: 700; padding-top: 12px; margin-bottom: .25rem;}
.subtle {color:#6c757d;}
.card {border:1px solid #e9ecef; padding:1rem 1.25rem; border-radius:16px;
       box-shadow:0 1px 6px rgba(0,0,0,0.04); background:linear-gradient(180deg,#ffffff 0%, #fafbff 100%);}
.kpi {border-radius:16px; padding:1rem 1.25rem;
     background:linear-gradient(180deg,#f8f9ff 0%, #f1f3ff 100%); border:1px solid #e9e7ff;}
hr {border: none; border-top: 1px solid #eee; margin: 1.25rem 0;}
.stProgress > div > div > div > div { background-image: linear-gradient(to right, #6a11cb, #2575fc); }
.dataframe tbody tr:hover {background-color:#fafafa;}
button[kind="primary"] {border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🧠 Transcript Categorizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Upload categories & transcripts → Select columns → Process & Download results</div>', unsafe_allow_html=True)
st.write("")

# =========================
# ---- CONSTANTS/KNOBS ----
# =========================
PARQUET_COMPRESSION = "zstd"
CHUNK_ROWS = 10_000  # Reduced for stability
MAX_FILE_SIZE_MB = 500  # File size limit
MAX_ROWS = 1_000_000   # Row limit for safety

# =========================
# ---- HELPERS / CACHE ----
# =========================
def validate_file_size(file_bytes: bytes, max_mb: int = MAX_FILE_SIZE_MB) -> bool:
    """Validate file size to prevent memory issues."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        st.error(f"File too large ({size_mb:.1f}MB). Maximum allowed: {max_mb}MB")
        return False
    return True

def clean_text_for_matching(text: str) -> str:
    """Enhanced text cleaning for better matching."""
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text).strip()
    
    # Normalize whitespace and remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common punctuation that might interfere with matching
    text = re.sub(r'[^\w\s-]', ' ', text)
    
    # Convert to lowercase for case-insensitive matching
    text = text.lower()
    
    return text

@st.cache_data(show_spinner=False, max_entries=2)
def read_categories(file_bytes: bytes, filename: str) -> pl.DataFrame:
    """Read categories CSV/XLSX into Polars DataFrame with validation."""
    try:
        if not validate_file_size(file_bytes):
            return None
            
        suffix = Path(filename).suffix.lower()
        
        if suffix == ".csv":
            df = pl.read_csv(io.BytesIO(file_bytes))
        elif suffix in [".xls", ".xlsx"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            
            try:
                xls = pd.ExcelFile(tmp_path)
                frames = []
                for sheet in xls.sheet_names:
                    sheet_df = pd.read_excel(tmp_path, sheet_name=sheet, dtype=str)
                    frames.append(sheet_df)
                
                if frames:
                    pdf = pd.concat(frames, ignore_index=True)
                    df = pl.from_pandas(pdf)
                else:
                    raise ValueError("No valid sheets found in Excel file")
                    
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            raise ValueError("Category file must be .csv or .xlsx")
        
        # Validate DataFrame
        if df.height == 0:
            raise ValueError("Categories file is empty")
            
        if df.height > MAX_ROWS:
            raise ValueError(f"Categories file too large ({df.height:,} rows). Maximum: {MAX_ROWS:,}")
        
        # Clean up column names
        df = df.rename({col: col.strip() for col in df.columns})
        
        return df
        
    except Exception as e:
        logger.error(f"Error reading categories file: {e}")
        st.error(f"Error reading categories file: {e}")
        return None

@st.cache_data(show_spinner=False, max_entries=2)
def convert_csv_to_parquet_streaming(csv_bytes: bytes, filename: str, compression: str = PARQUET_COMPRESSION) -> bytes:
    """Polars streaming CSV -> Parquet with memory management."""
    try:
        with tempfile.TemporaryDirectory() as tmpd:
            csv_path = os.path.join(tmpd, Path(filename).name)
            parquet_path = os.path.join(tmpd, Path(filename).with_suffix(".parquet").name)
            
            with open(csv_path, "wb") as f:
                f.write(csv_bytes)
            
            # Use streaming to handle large files
            pl.scan_csv(csv_path).sink_parquet(parquet_path, compression=compression)
            
            with open(parquet_path, "rb") as f:
                return f.read()
                
    except Exception as e:
        logger.error(f"Error converting CSV to Parquet: {e}")
        raise

@st.cache_data(show_spinner=False, max_entries=2)
def read_transcripts_any(file_bytes: bytes, filename: str) -> pl.DataFrame:
    """Read transcripts CSV/Parquet with validation."""
    try:
        if not validate_file_size(file_bytes):
            return None
            
        suffix = Path(filename).suffix.lower()
        
        if suffix == ".parquet":
            df = pl.read_parquet(io.BytesIO(file_bytes))
        elif suffix == ".csv":
            pq_bytes = convert_csv_to_parquet_streaming(file_bytes, filename)
            df = pl.read_parquet(io.BytesIO(pq_bytes))
        else:
            raise ValueError("Transcripts must be .csv or .parquet")
        
        # Validate DataFrame
        if df.height == 0:
            raise ValueError("Transcripts file is empty")
            
        if df.height > MAX_ROWS:
            raise ValueError(f"Transcripts file too large ({df.height:,} rows). Maximum: {MAX_ROWS:,}")
        
        # Clean up column names
        df = df.rename({col: col.strip() for col in df.columns})
        
        return df
        
    except Exception as e:
        logger.error(f"Error reading transcripts file: {e}")
        st.error(f"Error reading transcripts file: {e}")
        return None

@st.cache_resource(show_spinner=False, max_entries=1)
def build_keyword_processor(cat_df: pl.DataFrame, phrase_col: str) -> Tuple[KeywordProcessor, int]:
    """
    Build FlashText index with better phrase processing.
    Returns (processor, phrase_count)
    """
    try:
        kp = KeywordProcessor(case_sensitive=False)
        
        # Collect unique phrases and their paths
        phrase_map: Dict[str, List[Dict[str, Optional[str]]]] = defaultdict(list)
        processed_phrases = 0
        
        for row in cat_df.iter_rows(named=True):
            phrase = row.get(phrase_col)
            
            # Skip empty/null phrases
            if not phrase or str(phrase).strip() == "" or str(phrase).lower() in ["nan", "null", "none"]:
                continue
            
            # Clean and normalize phrase
            phrase_clean = clean_text_for_matching(str(phrase))
            if len(phrase_clean.strip()) < 2:  # Skip very short phrases
                continue
            
            # Build path from available levels
            path = {}
            for level in ["L1", "L2", "L3", "L4"]:
                if level in row:
                    value = row[level]
                    path[level] = str(value).strip() if value and str(value).lower() not in ["nan", "null", "none"] else None
                else:
                    path[level] = None
            
            phrase_map[phrase_clean].append(path)
            processed_phrases += 1
        
        # Add keywords to processor
        for phrase_clean, paths in phrase_map.items():
            value_json = orjson.dumps({"phrase": phrase_clean, "paths": paths}).decode("utf-8")
            kp.add_keyword(phrase_clean, value_json)
        
        logger.info(f"Built keyword processor with {len(phrase_map)} unique phrases from {processed_phrases} total phrases")
        return kp, len(phrase_map)
        
    except Exception as e:
        logger.error(f"Error building keyword processor: {e}")
        st.error(f"Error building keyword processor: {e}")
        raise

def categorize_chunk(chunk_df: pl.DataFrame, kp: KeywordProcessor, id_col: str, text_col: str) -> List[Dict[str, Optional[str]]]:
    """Categorize a chunk of transcripts."""
    rows = []
    
    for row_data in chunk_df.iter_rows(named=True):
        rid = row_data[id_col]
        text_raw = row_data[text_col]
        
        # Clean text for matching
        text = clean_text_for_matching(text_raw) if text_raw else ""
        
        if not text:
            rows.append({
                id_col: str(rid),
                "matched_phrase": None,
                "L1": None, "L2": None, "L3": None, "L4": None
            })
            continue
        
        # Extract keywords
        try:
            hits = kp.extract_keywords(text, span_info=True)
        except Exception as e:
            logger.warning(f"Error extracting keywords for ID {rid}: {e}")
            hits = []
        
        if not hits:
            rows.append({
                id_col: str(rid),
                "matched_phrase": None,
                "L1": None, "L2": None, "L3": None, "L4": None
            })
            continue
        
        # Process hits
        for val_json, _start, _end in hits:
            try:
                obj = orjson.loads(val_json) if isinstance(val_json, str) else val_json
                matched_phrase = obj.get("phrase", "")
                paths = obj.get("paths", [])
                
                if not isinstance(paths, list):
                    paths = [{"L1": None, "L2": None, "L3": None, "L4": None}]
                
                if not paths:
                    paths = [{"L1": None, "L2": None, "L3": None, "L4": None}]
                
                for path in paths:
                    rows.append({
                        id_col: str(rid),
                        "matched_phrase": matched_phrase,
                        "L1": path.get("L1"),
                        "L2": path.get("L2"),
                        "L3": path.get("L3"),
                        "L4": path.get("L4"),
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing match for ID {rid}: {e}")
                rows.append({
                    id_col: str(rid),
                    "matched_phrase": str(val_json) if val_json else None,
                    "L1": None, "L2": None, "L3": None, "L4": None
                })
    
    return rows

def categorize_df(transcripts_df: pl.DataFrame, kp: KeywordProcessor, id_col: str, text_col: str) -> pl.DataFrame:
    """
    Categorize transcripts with better memory management and error handling.
    """
    try:
        n = transcripts_df.height
        if n == 0:
            return pl.DataFrame(
                schema={id_col: pl.Utf8, "matched_phrase": pl.Utf8, "L1": pl.Utf8, "L2": pl.Utf8, "L3": pl.Utf8, "L4": pl.Utf8}
            )
        
        batches = math.ceil(n / CHUNK_ROWS)
        out_frames: List[pl.DataFrame] = []
        
        prog = st.progress(0.0, text="Categorizing transcripts…")
        
        for b in range(batches):
            try:
                start = b * CHUNK_ROWS
                stop = min((b + 1) * CHUNK_ROWS, n)
                chunk_df = transcripts_df.slice(start, stop - start)
                
                # Process chunk
                rows = categorize_chunk(chunk_df, kp, id_col, text_col)
                
                if rows:
                    chunk_result = pl.DataFrame(rows)
                    out_frames.append(chunk_result)
                
                # Update progress
                progress = (b + 1) / batches
                prog.progress(progress, text=f"Categorizing transcripts… {b + 1}/{batches} batches")
                
                # Force garbage collection every 10 batches
                if b % 10 == 0:
                    gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing batch {b + 1}: {e}")
                st.warning(f"Error processing batch {b + 1}, skipping...")
                continue
        
        prog.empty()
        
        if not out_frames:
            return pl.DataFrame(
                schema={id_col: pl.Utf8, "matched_phrase": pl.Utf8, "L1": pl.Utf8, "L2": pl.Utf8, "L3": pl.Utf8, "L4": pl.Utf8}
            )
        
        result = pl.concat(out_frames)
        
        # Force cleanup
        del out_frames
        gc.collect()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in categorize_df: {e}")
        st.error(f"Categorization failed: {e}")
        raise

def make_downloads(joined: pl.DataFrame, summary: pl.DataFrame) -> Tuple[bytes, bytes, bytes]:
    """Create download files with better memory management."""
    try:
        # Parquet
        pq_buf = io.BytesIO()
        joined.write_parquet(pq_buf, compression=PARQUET_COMPRESSION)
        pq_bytes = pq_buf.getvalue()
        pq_buf.close()

        # Excel (limit size for stability)
        if joined.height > 100_000:
            st.warning("Excel output limited to first 100,000 rows due to size constraints. Use Parquet for complete data.")
            joined_excel = joined.head(100_000)
        else:
            joined_excel = joined
        
        xls_buf = io.BytesIO()
        with pd.ExcelWriter(xls_buf, engine="xlsxwriter", options={'strings_to_numbers': True}) as xl:
            joined_excel.to_pandas(use_pyarrow_extension_array=True).to_excel(
                xl, index=False, sheet_name="raw_with_categories"
            )
            summary.to_pandas(use_pyarrow_extension_array=True).to_excel(
                xl, index=False, sheet_name="summary_by_category"
            )
        xls_bytes = xls_buf.getvalue()
        xls_buf.close()

        # CSV via pandas (limit size for stability)
        if joined.height > 200_000:
            st.warning("CSV output limited to first 200,000 rows due to size constraints. Use Parquet for complete data.")
            joined_csv = joined.head(200_000)
        else:
            joined_csv = joined
            
        csv_buf = io.StringIO()
        joined_csv.to_pandas(use_pyarrow_extension_array=True).to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode("utf-8")
        csv_buf.close()

        return pq_bytes, xls_bytes, csv_bytes
        
    except Exception as e:
        logger.error(f"Error creating downloads: {e}")
        st.error(f"Error creating download files: {e}")
        raise

# =========================
# ---- SIDEBAR / INPUT ----
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # File size warnings
    st.info(f"📊 Limits: {MAX_FILE_SIZE_MB}MB per file, {MAX_ROWS:,} rows max")
    
    cat_file = st.file_uploader("Categories (.csv or .xlsx)", type=["csv", "xlsx"])
    trn_file = st.file_uploader("Transcripts (.csv or .parquet)", type=["csv", "parquet"])

    phrase_col, id_col, text_col = None, None, None
    cat_df, trn_df = None, None

    # Categories upload -> show column selectors immediately
    if cat_file:
        try:
            cat_df = read_categories(cat_file.read(), cat_file.name)
            if cat_df is not None:
                st.success(f"✅ Categories: {cat_df.height:,} rows, {len(cat_df.columns)} columns")
                
                # Show available columns
                available_cols = list(cat_df.columns)
                st.write("**Available columns:**", ", ".join(available_cols))
                
                phrase_col = st.selectbox(
                    "Select Phrase column", 
                    options=available_cols,
                    help="Column containing the phrases/keywords to match"
                )
                
                # Show preview of selected phrase column
                if phrase_col:
                    phrase_preview = cat_df.select(phrase_col).head(5).to_pandas()
                    st.write(f"**Preview of {phrase_col}:**")
                    st.dataframe(phrase_preview, use_container_width=True)
            else:
                st.error("Failed to load categories file")
        except Exception as e:
            st.error(f"Error loading categories: {e}")

    # Transcripts upload -> show column selectors immediately
    if trn_file:
        try:
            trn_df = read_transcripts_any(trn_file.read(), trn_file.name)
            if trn_df is not None:
                st.success(f"✅ Transcripts: {trn_df.height:,} rows, {len(trn_df.columns)} columns")
                
                # Show available columns
                cols = list(trn_df.columns)
                st.write("**Available columns:**", ", ".join(cols))
                
                id_col = st.selectbox(
                    "Select ID column (or <auto-generate>)", 
                    options=["<auto-generate>"] + cols,
                    help="Unique identifier for each transcript"
                )
                text_col = st.selectbox(
                    "Select Transcript Text column", 
                    options=cols,
                    help="Column containing the text to categorize"
                )
                
                # Show preview of selected text column
                if text_col:
                    text_preview = trn_df.select(text_col).head(3).to_pandas()
                    st.write(f"**Preview of {text_col}:**")
                    st.dataframe(text_preview, use_container_width=True)
            else:
                st.error("Failed to load transcripts file")
        except Exception as e:
            st.error(f"Error loading transcripts: {e}")

    # Only show Process when all selections are available
    go = False
    if cat_file and trn_file and phrase_col and text_col and cat_df is not None and trn_df is not None:
        st.markdown("---")
        st.markdown("### 🚀 Ready to Process")
        st.write(f"• Categories: {cat_df.height:,} rows")
        st.write(f"• Transcripts: {trn_df.height:,} rows") 
        st.write(f"• Phrase column: {phrase_col}")
        st.write(f"• Text column: {text_col}")
        
        go = st.button("🚀 Start Processing", type="primary", use_container_width=True)

# =========================
# --------- MAIN ----------
# =========================
if go and cat_df is not None and trn_df is not None:
    start_time = time.time()
    
    try:
        # 1) Build keyword index
        st.info("🔨 Building keyword index...")
        kp, phrase_count = build_keyword_processor(cat_df, phrase_col)
        st.success(f"✅ Built index with {phrase_count:,} unique phrases")

        # 2) If no ID column, generate one
        if id_col == "<auto-generate>":
            trn_df = trn_df.with_row_index(name="row_id")
            id_col = "row_id"
            st.info("🔢 Generated row IDs")

        # 3) Categorize
        st.info("🏷️ Starting categorization...")
        cat_only = categorize_df(trn_df.select([id_col, text_col]), kp, id_col, text_col)
        
        # Calculate match statistics
        total_transcripts = trn_df.height
        matched_transcripts = cat_only.filter(pl.col("matched_phrase").is_not_null()).select(id_col).n_unique()
        match_rate = (matched_transcripts / total_transcripts) * 100 if total_transcripts > 0 else 0
        
        st.success(f"✅ Categorization complete!")
        st.info(f"📊 Match Rate: {matched_transcripts:,} / {total_transcripts:,} transcripts ({match_rate:.1f}%)")

        # 4) Join results
        st.info("🔗 Joining results...")
        trn_df = trn_df.with_columns(pl.col(id_col).cast(pl.Utf8))
        cat_only = cat_only.with_columns(pl.col(id_col).cast(pl.Utf8))
        joined = trn_df.join(cat_only, on=id_col, how="left")

        # 5) Create summary
        summary = (
            joined.group_by(["L1", "L2", "L3", "L4"])
                  .agg([
                      pl.len().alias("n_conversations"),
                      pl.col("matched_phrase").n_unique().alias("unique_phrases_matched")
                  ])
                  .sort("n_conversations", descending=True)
        )

        # Remove the all-null row from summary for display
        summary_display = summary.filter(
            pl.col("L1").is_not_null() | 
            pl.col("L2").is_not_null() | 
            pl.col("L3").is_not_null() | 
            pl.col("L4").is_not_null()
        )

        # 6) Show results
        processing_time = time.time() - start_time
        st.success(f"🎉 Processing completed in {processing_time:.1f} seconds")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transcripts", f"{total_transcripts:,}")
        with col2:
            st.metric("Matched Transcripts", f"{matched_transcripts:,}")
        with col3:
            st.metric("Match Rate", f"{match_rate:.1f}%")
        with col4:
            st.metric("Categories Found", f"{summary_display.height:,}")

        st.write("### 📊 Top Categories")
        if summary_display.height > 0:
            st.dataframe(
                summary_display.head(25).to_pandas(), 
                use_container_width=True, 
                height=420
            )
        else:
            st.warning("No categorized matches found. Check your phrase column and text data quality.")

        # 7) Prepare downloads
        st.info("📦 Preparing download files...")
        pq_bytes, xls_bytes, csv_bytes = make_downloads(joined, summary)
        
        # Download buttons
        st.write("### 📥 Download Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "⬇️ Download Parquet", 
                data=pq_bytes, 
                file_name=f"categorized_output_{int(time.time())}.parquet",
                mime="application/octet-stream", 
                use_container_width=True,
                help="Best format for large datasets"
            )
        with col2:
            st.download_button(
                "⬇️ Download Excel", 
                data=xls_bytes, 
                file_name=f"categorized_output_{int(time.time())}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                help="Includes summary sheet"
            )
        with col3:
            st.download_button(
                "⬇️ Download CSV", 
                data=csv_bytes, 
                file_name=f"categorized_output_{int(time.time())}.csv",
                mime="text/csv", 
                use_container_width=True,
                help="Compatible with most tools"
            )

        st.success("✅ All files ready for download! Refresh to process new files.")
        
        # Cleanup
        del kp, cat_only, joined, summary
        gc.collect()

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        st.error(f"❌ Processing failed: {e}")
        st.write("Please check your input files and try again.")

else:
    # Show instructions
    st.markdown("""
    ## 📋 Instructions
    
    1. **Upload Categories File** (.csv or .xlsx)
       - Must contain phrases/keywords to match
       - Can have hierarchical levels (L1, L2, L3, L4)
    
    2. **Upload Transcripts File** (.csv or .parquet) 
       - Must contain text data to categorize
       - Should have an ID column (or auto-generate)
    
    3. **Select Columns**
       - Choose the phrase column from categories
       - Choose ID and text columns from transcripts
    
    4. **Process & Download**
       - Click "Start Processing" 
       - Download results in your preferred format
    
    ### 💡 Tips for Better Results
    - Clean your phrase data (remove duplicates, fix typos)
    - Ensure text column contains meaningful content
    - Use Parquet format for large files (faster processing)
    - Check match rates - low rates may indicate data quality issues
    """)
    
    if not (cat_file and trn_file):
        st.info("👆 Upload both files in the sidebar to get started")
    elif not (phrase_col and text_col):
        st.info("👆 Select the required columns in the sidebar")
