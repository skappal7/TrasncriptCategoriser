# app.py
import io, os, tempfile, math, orjson, time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Tuple, Dict, List

import streamlit as st
import polars as pl
import pandas as pd
from flashtext import KeywordProcessor

# =========================
# ---- UI / THEME CSS  ----
# =========================
st.set_page_config(page_title="Transcript Categorizer", page_icon="🧠", layout="wide")
st.markdown("""
<style>
/* App-wide polish */
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.big-title {font-size: 2.0rem; font-weight: 800; margin-bottom: .25rem;}
.subtle {color:#6c757d;}
.card {border:1px solid #e9ecef; padding:1rem 1.25rem; border-radius:16px; box-shadow:0 1px 6px rgba(0,0,0,0.04); background:linear-gradient(180deg,#ffffff 0%, #fafbff 100%);}
.kpi {border-radius:16px; padding:1rem 1.25rem; background:linear-gradient(180deg,#f8f9ff 0%, #f1f3ff 100%); border:1px solid #e9e7ff;}
hr {border: none; border-top: 1px solid #eee; margin: 1.25rem 0;}
.stProgress > div > div > div > div { background-image: linear-gradient(to right, #6a11cb, #2575fc); }
.dataframe tbody tr:hover {background-color:#fafafa;}
button[kind="primary"] {border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🧠 Transcript Categorizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Upload categories & transcripts, fast-convert CSV → Parquet (Polars streaming), categorize, and download results.</div>', unsafe_allow_html=True)
st.write("")

# =========================
# ---- CONSTANTS/KNOBS ----
# =========================
PHRASE_COL = "L4"                           # L4 contains the matchable phrase
CATEGORY_PATH_COLS = ["L1", "L2", "L3", "L4"]  # reporting path includes L4
PARQUET_COMPRESSION = "zstd"
CHUNK_ROWS = 50_000                         # batch size during categorization
ENABLE_LEMMATIZE = False                    # kept off to avoid NLTK downloads in Streamlit
ID_COL_CANDIDATES   = ["id","call_id","conversation_id","ticket_id"]
TEXT_COL_CANDIDATES = ["text","transcript","conversation","content","dialogue","utterances"]

# =========================
# ---- CACHE HELPERS   ----
# =========================
@st.cache_data(show_spinner=False)
def read_categories(file_bytes: bytes, filename: str) -> pl.DataFrame:
    """
    Accepts CSV/XLSX with L1..L4 (multi-sheet ok) OR category/phrase.
    Returns: Polars DF with columns: L1,L2,L3,L4, category_path, phrase
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        df = pl.read_csv(io.BytesIO(file_bytes))
    elif suffix in [".xls", ".xlsx"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        xls = pd.ExcelFile(tmp_path)
        frames = [pd.read_excel(tmp_path, sheet_name=s, dtype=str) for s in xls.sheet_names]
        pdf = pd.concat(frames, ignore_index=True)
        df = pl.from_pandas(pdf)
        os.remove(tmp_path)
    else:
        raise ValueError("Category file must be .csv or .xlsx")

    lower = {c.lower(): c for c in df.columns}
    # L1..L4 path
    if all(k in lower for k in ["l1","l2","l3","l4"]):
        df = df.rename({lower["l1"]:"L1", lower["l2"]:"L2", lower["l3"]:"L3", lower["l4"]:"L4"})
        for c in ["L1","L2","L3","L4"]:
            if c in df.columns:
                df = df.with_columns(pl.col(c).cast(pl.Utf8).str.strip())
        df = df.with_columns(
            pl.concat_str([pl.col(c).fill_null("") for c in CATEGORY_PATH_COLS], separator=" > ")
              .alias("category_path"),
            pl.col(PHRASE_COL).alias("phrase")
        )
    # category/phrase flat
    elif "category" in lower and "phrase" in lower:
        df = df.rename({lower["category"]:"category", lower["phrase"]:"phrase"})
        df = df.with_columns(
            pl.col("category").cast(pl.Utf8).str.strip(),
            pl.col("phrase").cast(pl.Utf8).str.strip(),
        ).with_columns(
            pl.col("category").alias("category_path"),
            pl.lit(None).alias("L1"),
            pl.lit(None).alias("L2"),
            pl.lit(None).alias("L3"),
            pl.lit(None).alias("L4"),
        )
    else:
        raise ValueError("Expected columns L1..L4 OR category, phrase in categories file.")

    df = df.filter(pl.col("phrase").is_not_null() & (pl.col("phrase").str.len_chars() > 0))
    return df.select(["L1","L2","L3","L4","category_path","phrase"])

@st.cache_resource(show_spinner=False)
def build_keyword_processor(cat_df: pl.DataFrame):
    kp = KeywordProcessor(case_sensitive=False)
    phrase_to_paths: Dict[str, set] = defaultdict(set)
    # Simple normalization (lowercase); lemmatization skipped for speed & no external downloads
    def normalize(s: str) -> str:
        return (s or "").strip().lower()
    for r in cat_df.iter_rows(named=True):
        phrase = normalize(r["phrase"])
        phrase_to_paths[phrase].add(r["category_path"])
    for phrase, paths in phrase_to_paths.items():
        kp.add_keyword(phrase, tuple(sorted(paths)))
    return kp

def _detect_cols(df: pl.DataFrame, manual_id: str|None=None, manual_text: str|None=None) -> Tuple[str,str]:
    if manual_id and manual_id in df.columns:
        id_col = manual_id
    else:
        lower = {c.lower(): c for c in df.columns}
        id_col = next((lower[c] for c in ID_COL_CANDIDATES if c in lower), None)

    if manual_text and manual_text in df.columns:
        text_col = manual_text
    else:
        lower = {c.lower(): c for c in df.columns}
        text_col = next((lower[c] for c in TEXT_COL_CANDIDATES if c in lower), None)

    if not id_col or not text_col:
        raise ValueError(f"Could not detect ID/TEXT columns. Found: {df.columns}")
    return id_col, text_col

@st.cache_data(show_spinner=False)
def convert_csv_to_parquet_streaming(csv_bytes: bytes, filename: str, compression: str = PARQUET_COMPRESSION) -> bytes:
    """
    Uses polars scan_csv(...).sink_parquet(...) to stream-convert without loading whole CSV.
    Returns Parquet file bytes.
    """
    with tempfile.TemporaryDirectory() as tmpd:
        csv_path = os.path.join(tmpd, Path(filename).name)
        with open(csv_path, "wb") as f:
            f.write(csv_bytes)
        parquet_path = os.path.join(tmpd, Path(filename).with_suffix(".parquet").name)
        # streaming conversion (fast for large text)
        pl.scan_csv(csv_path).sink_parquet(parquet_path, compression=compression)
        with open(parquet_path, "rb") as f:
            return f.read()

@st.cache_data(show_spinner=False)
def read_transcripts_any(file_bytes: bytes, filename: str) -> pl.DataFrame:
    """
    Accepts CSV or Parquet. If CSV -> stream convert to Parquet first.
    Returns Polars DataFrame (full materialized, but downstream we can chunk if needed).
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(io.BytesIO(file_bytes))
    elif suffix == ".csv":
        pq_bytes = convert_csv_to_parquet_streaming(file_bytes, filename)
        return pl.read_parquet(io.BytesIO(pq_bytes))
    else:
        raise ValueError("Transcripts must be .csv or .parquet")

def _summarize_hits(hits) -> Tuple[List[str], Dict[str,int], str|None]:
    counter = Counter()
    for h in hits:
        # FlashText span_info returns (value, start, end) OR (value, matched, start, end)
        val = h[0]
        paths = val if isinstance(val, tuple) else (str(val),)
        for p in paths:
            counter[p] += 1
    all_paths = list(counter.keys())
    top_path = max(counter.items(), key=lambda kv: kv[1])[0] if counter else None
    return all_paths, dict(counter), top_path

def _normalize_text_basic(s: str) -> str:
    return (s or "").strip().lower()

def categorize_df(transcripts_df: pl.DataFrame, kp: KeywordProcessor, id_col: str, text_col: str) -> pl.DataFrame:
    """
    Chunked categorization to keep memory OK with very long rows.
    """
    n = transcripts_df.height
    batches = math.ceil(n / CHUNK_ROWS)
    out_frames = []
    prog = st.progress(0.0, text="Categorizing transcripts…")
    for b in range(batches):
        start = b * CHUNK_ROWS
        stop = min((b + 1) * CHUNK_ROWS, n)
        df = transcripts_df.slice(start, stop - start)
        rows = []
        for r in df.iter_rows(named=True):
            rid = str(r[id_col])
            t   = _normalize_text_basic(r[text_col] if r[text_col] is not None else "")
            hits = kp.extract_keywords(t, span_info=True)
            all_paths, cnts, top_path = _summarize_hits(hits)
            rows.append({
                id_col: rid,
                "all_categories_path": all_paths,
                "top_category_path": top_path,
                "category_path_counts_json": orjson.dumps(cnts).decode()
            })
        out_frames.append(pl.DataFrame(rows))
        prog.progress((b+1)/batches, text=f"Categorizing transcripts… {b+1}/{batches} batches")
    prog.empty()
    return pl.concat(out_frames) if out_frames else pl.DataFrame(schema={id_col: pl.Utf8,
                                                                       "all_categories_path": pl.List(pl.Utf8),
                                                                       "top_category_path": pl.Utf8,
                                                                       "category_path_counts_json": pl.Utf8})

def make_downloads(joined: pl.DataFrame, summary: pl.DataFrame) -> Tuple[bytes, bytes, bytes]:
    """
    Returns (parquet_bytes, excel_bytes, csv_zip_bytes(optional)) -> we'll return parquet & excel & csv (two files merged into one CSV? better: two CSVs in zip).
    To keep it simple: Parquet + Excel. Also CSV (raw_with_categories) only.
    """
    # Parquet
    pq_buf = io.BytesIO()
    joined.write_parquet(pq_buf, compression=PARQUET_COMPRESSION)
    pq_bytes = pq_buf.getvalue()

    # Excel with two sheets
    xls_buf = io.BytesIO()
    with pd.ExcelWriter(xls_buf, engine="xlsxwriter") as xl:
        joined.to_pandas(use_pyarrow_extension_array=True).to_excel(xl, index=False, sheet_name="raw_with_categories")
        summary.to_pandas(use_pyarrow_extension_array=True).to_excel(xl, index=False, sheet_name="summary_by_category")
    xls_bytes = xls_buf.getvalue()

    # CSV (raw_with_categories) for quick look
    csv_buf = io.StringIO()
    joined.write_csv(csv_buf)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    return pq_bytes, xls_bytes, csv_bytes

# =========================
# ---- SIDEBAR / INPUT ----
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.caption("L4 is used for matching. Reporting path = L1 > L2 > L3 > L4.")
    st.caption("CSV→Parquet uses Polars streaming for speed.")
    st.write("")
    st.markdown("**Upload Files**")
    cat_file = st.file_uploader("Categories (.csv or .xlsx with L1..L4 or category/phrase)", type=["csv","xlsx"])
    trn_file = st.file_uploader("Transcripts (.csv or .parquet)", type=["csv","parquet"])

    st.write("")
    st.markdown("**Column Overrides (optional)**")
    override_id = st.text_input("ID column (leave blank to auto-detect)")
    override_text = st.text_input("Text column (leave blank to auto-detect)")

    st.write("")
    go = st.button("🚀 Process", type="primary", use_container_width=True)

# =========================
# --------- MAIN ----------
# =========================
col1, col2, col3 = st.columns([1,1,1], gap="large")
with col1:
    st.markdown('<div class="kpi"><b>Step 1</b><br/>Upload your files</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="kpi"><b>Step 2</b><br/>Auto-convert CSV → Parquet</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="kpi"><b>Step 3</b><br/>Categorize & Download</div>', unsafe_allow_html=True)

st.write("")
st.markdown('<div class="card">', unsafe_allow_html=True)

if go:
    if not cat_file or not trn_file:
        st.error("Please upload both **Categories** and **Transcripts** files.")
    else:
        # Load categories
        try:
            with st.spinner("Reading categories…"):
                cat_df = read_categories(cat_file.read(), cat_file.name)
            st.success(f"Loaded {len(cat_df):,} phrases · {cat_df.select(pl.n_unique('category_path')).item()} unique category paths")
        except Exception as e:
            st.error(f"Failed to read categories: {e}")
            st.stop()

        # Build keyword processor
        try:
            with st.spinner("Building keyword index (FlashText)…"):
                kp = build_keyword_processor(cat_df)
            st.success("Keyword index ready")
        except Exception as e:
            st.error(f"Failed to build keyword index: {e}")
            st.stop()

        # Load transcripts (CSV -> Parquet streaming if needed)
        try:
            st.info("Loading transcripts (CSV will be converted to Parquet via Polars streaming)…")
            start = time.time()
            trn_bytes = trn_file.read()
            trn_df = read_transcripts_any(trn_bytes, trn_file.name)
            elapsed = time.time() - start
            st.success(f"Transcripts loaded in {elapsed:0.1f}s · Rows: {trn_df.height:,}")
        except Exception as e:
            st.error(f"Failed to read transcripts: {e}")
            st.stop()

        # Choose ID/TEXT columns (auto-detect + allow override)
        try:
            sample = trn_df.head(1)
            id_col, text_col = _detect_cols(sample,
                                            manual_id=override_id.strip() or None,
                                            manual_text=override_text.strip() or None)
        except Exception as e:
            st.error(f"Column detection error: {e}")
            st.stop()

        st.write("")
        st.markdown(f"**Detected columns** → ID: `{id_col}` · TEXT: `{text_col}`")

        # Categorize
        try:
            cat_only = categorize_df(trn_df.select([id_col, text_col]), kp, id_col, text_col)
            st.success(f"Categorized {cat_only.height:,} rows")
        except Exception as e:
            st.error(f"Categorization failed: {e}")
            st.stop()

        # Join back + summary
        joined = trn_df.join(cat_only, on=id_col, how="left")
        summary = (
            joined.with_columns(pl.col("all_categories_path").fill_null([]))
                  .explode("all_categories_path")
                  .group_by("all_categories_path")
                  .agg(pl.len().alias("n_conversations"))
                  .sort("n_conversations", descending=True)
                  .rename({"all_categories_path":"category_path"})
        )

        st.write("")
        st.markdown("### 📊 Summary by Category Path (Top 25)")
        st.dataframe(summary.head(25).to_pandas(), use_container_width=True, height=420)

        # Downloads
        pq_bytes, xls_bytes, csv_bytes = make_downloads(joined, summary)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("⬇️ Download Parquet", data=pq_bytes, file_name="categorized_output.parquet",
                               mime="application/octet-stream", use_container_width=True)
        with c2:
            st.download_button("⬇️ Download Excel (2 sheets)", data=xls_bytes, file_name="categorized_output.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        with c3:
            st.download_button("⬇️ Download CSV (raw_with_categories)", data=csv_bytes, file_name="categorized_output.csv",
                               mime="text/csv", use_container_width=True)

        st.success("Done! Refresh the browser to clear cache and start over.")
else:
    st.info("Upload files in the sidebar and click **Process** to begin.")

st.markdown('</div>', unsafe_allow_html=True)
