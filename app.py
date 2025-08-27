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
CHUNK_ROWS = 50_000

# =========================
# ---- HELPERS / CACHE ----
# =========================
@st.cache_data(show_spinner=False)
def read_categories(file_bytes: bytes, filename: str) -> pl.DataFrame:
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
    return df

@st.cache_resource(show_spinner=False)
def build_keyword_processor(cat_df: pl.DataFrame, phrase_col: str, category_path_cols: List[str]):
    kp = KeywordProcessor(case_sensitive=False)
    phrase_to_paths: Dict[str, set] = defaultdict(set)

    def normalize(s: str) -> str: return (s or "").strip().lower()
    for r in cat_df.iter_rows(named=True):
        if r[phrase_col] is None:
            continue
        phrase = normalize(r[phrase_col])
        path_parts = [str(r[c]) for c in category_path_cols if c in r and r[c] is not None]
        category_path = " > ".join(path_parts)
        phrase_to_paths[phrase].add(category_path)
    for phrase, paths in phrase_to_paths.items():
        kp.add_keyword(phrase, tuple(sorted(paths)))
    return kp

@st.cache_data(show_spinner=False)
def convert_csv_to_parquet_streaming(csv_bytes: bytes, filename: str, compression: str = PARQUET_COMPRESSION) -> bytes:
    with tempfile.TemporaryDirectory() as tmpd:
        csv_path = os.path.join(tmpd, Path(filename).name)
        with open(csv_path, "wb") as f:
            f.write(csv_bytes)
        parquet_path = os.path.join(tmpd, Path(filename).with_suffix(".parquet").name)
        pl.scan_csv(csv_path).sink_parquet(parquet_path, compression=compression)
        with open(parquet_path, "rb") as f:
            return f.read()

@st.cache_data(show_spinner=False)
def read_transcripts_any(file_bytes: bytes, filename: str) -> pl.DataFrame:
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
            rid = str(r[id_col]) if id_col else str(start + len(rows))
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
    return pl.concat(out_frames)

def make_downloads(joined: pl.DataFrame, summary: pl.DataFrame) -> Tuple[bytes, bytes, bytes]:
    pq_buf = io.BytesIO()
    joined.write_parquet(pq_buf, compression=PARQUET_COMPRESSION)
    pq_bytes = pq_buf.getvalue()

    xls_buf = io.BytesIO()
    with pd.ExcelWriter(xls_buf, engine="xlsxwriter") as xl:
        joined.to_pandas(use_pyarrow_extension_array=True).to_excel(xl, index=False, sheet_name="raw_with_categories")
        summary.to_pandas(use_pyarrow_extension_array=True).to_excel(xl, index=False, sheet_name="summary_by_category")
    xls_bytes = xls_buf.getvalue()

    csv_buf = io.StringIO()
    joined.write_csv(csv_buf)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    return pq_bytes, xls_bytes, csv_bytes

# =========================
# ---- SIDEBAR / INPUT ----
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    cat_file = st.file_uploader("Categories (.csv or .xlsx)", type=["csv","xlsx"])
    trn_file = st.file_uploader("Transcripts (.csv or .parquet)", type=["csv","parquet"])

    phrase_col, category_path_cols, id_col, text_col = None, [], None, None
    cat_df, trn_df = None, None

    if cat_file:
        cat_df = read_categories(cat_file.read(), cat_file.name)
        st.success(f"Categories loaded with {cat_df.height:,} rows")
        phrase_col = st.selectbox("Select Phrase column", options=cat_df.columns)
        available_levels = [c for c in ["L1","L2","L3","L4"] if c in cat_df.columns]
        category_path_cols = st.multiselect("Select Category Levels to include", options=available_levels, default=available_levels)

    if trn_file:
        trn_df = read_transcripts_any(trn_file.read(), trn_file.name)
        st.success(f"Transcripts loaded with {trn_df.height:,} rows")
        cols = trn_df.columns
        id_col = st.selectbox("Select ID column (or <none>)", options=["<none>"] + list(cols))
        text_col = st.selectbox("Select Transcript Text column", options=cols)

    go = False
    if cat_file and trn_file and phrase_col and text_col:
        go = st.button("🚀 Process", type="primary", use_container_width=True)

# =========================
# --------- MAIN ----------
# =========================
if go:
    try:
        kp = build_keyword_processor(cat_df, phrase_col, category_path_cols)
    except Exception as e:
        st.error(f"Failed to build keyword index: {e}")
        st.stop()

    if id_col == "<none>":
        trn_df = trn_df.with_row_index(name="row_id")
        id_col = "row_id"

    try:
        cat_only = categorize_df(trn_df.select([id_col, text_col]), kp, id_col, text_col)
        st.success(f"Categorized {cat_only.height:,} rows")
    except Exception as e:
        st.error(f"Categorization failed: {e}")
        st.stop()

    # ✅ FIX: cast join keys to Utf8 before joining
    trn_df = trn_df.with_columns(pl.col(id_col).cast(pl.Utf8))
    cat_only = cat_only.with_columns(pl.col(id_col).cast(pl.Utf8))

    joined = trn_df.join(cat_only, on=id_col, how="left")

    summary = (joined.with_columns(pl.col("all_categories_path").fill_null([]))
                      .explode("all_categories_path")
                      .group_by("all_categories_path")
                      .agg(pl.len().alias("n_conversations"))
                      .sort("n_conversations", descending=True)
                      .rename({"all_categories_path":"category_path"}))

    st.write("### 📊 Summary by Category Path (Top 25)")
    st.dataframe(summary.head(25).to_pandas(), use_container_width=True, height=420)

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
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="categorized_output.csv",
                           mime="text/csv", use_container_width=True)

    st.success("Done! Refresh the browser to clear cache and start over.")
else:
    st.info("Upload files in the sidebar, select columns, then click **Process**.")
