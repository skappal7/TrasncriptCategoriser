# app.py
import io, os, tempfile, math, orjson, time
from pathlib import Path
from collections import defaultdict
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
CHUNK_ROWS = 50_000  # categorization batch size

# =========================
# ---- HELPERS / CACHE ----
# =========================
@st.cache_data(show_spinner=False)
def read_categories(file_bytes: bytes, filename: str) -> pl.DataFrame:
    """Read categories CSV/XLSX into Polars DataFrame."""
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

@st.cache_data(show_spinner=False)
def convert_csv_to_parquet_streaming(csv_bytes: bytes, filename: str, compression: str = PARQUET_COMPRESSION) -> bytes:
    """Polars streaming CSV -> Parquet (fast, memory-light). Returns parquet bytes."""
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
    """Read transcripts CSV/Parquet. If CSV, stream-convert to Parquet first."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(io.BytesIO(file_bytes))
    elif suffix == ".csv":
        pq_bytes = convert_csv_to_parquet_streaming(file_bytes, filename)
        return pl.read_parquet(io.BytesIO(pq_bytes))
    else:
        raise ValueError("Transcripts must be .csv or .parquet")

@st.cache_resource(show_spinner=False)
def build_keyword_processor(cat_df: pl.DataFrame, phrase_col: str) -> KeywordProcessor:
    """
    Build FlashText index.
    Value stored for each phrase is a JSON string:
      {"phrase": <phrase>, "paths": [{"L1":..., "L2":..., "L3":..., "L4":...}, ...]}
    This supports one phrase mapping to multiple (L1..L4) paths.
    """
    kp = KeywordProcessor(case_sensitive=False)
    # normalize helper
    def norm(s: str) -> str:
        return (s or "").strip().lower()

    # Collect paths per phrase
    phrase_map: Dict[str, List[Dict[str, str | None]]] = defaultdict(list)
    # Ensure L1..L4 exist (fill with None if absent)
    present_levels = [c for c in ["L1", "L2", "L3", "L4"] if c in cat_df.columns]
    for r in cat_df.iter_rows(named=True):
        p = r.get(phrase_col)
        if p is None or str(p).strip() == "":
            continue
        path = {
            "L1": r["L1"] if "L1" in r else None,
            "L2": r["L2"] if "L2" in r else None,
            "L3": r["L3"] if "L3" in r else None,
            "L4": r["L4"] if "L4" in r else None,
        }
        phrase_map[norm(str(p))].append(path)

    # Add keywords (store JSON as value)
    for phrase_norm, paths in phrase_map.items():
        value_json = orjson.dumps({"phrase": phrase_norm, "paths": paths}).decode("utf-8")
        kp.add_keyword(phrase_norm, value_json)

    return kp

def _normalize_text_basic(s: str) -> str:
    return (s or "").strip().lower()

def categorize_df(transcripts_df: pl.DataFrame, kp: KeywordProcessor, id_col: str, text_col: str) -> pl.DataFrame:
    """
    Returns a row per (transcript, matched phrase, path), tabulated as:
      [id_col, matched_phrase, L1, L2, L3, L4]
    If no match for a transcript, returns one row with categories = None.
    """
    n = transcripts_df.height
    batches = math.ceil(n / CHUNK_ROWS)
    out_frames: List[pl.DataFrame] = []
    prog = st.progress(0.0, text="Categorizing transcripts…")

    for b in range(batches):
        start = b * CHUNK_ROWS
        stop = min((b + 1) * CHUNK_ROWS, n)
        df = transcripts_df.slice(start, stop - start)
        rows: List[Dict[str, str | None]] = []

        for r in df.iter_rows(named=True):
            rid = r[id_col]
            text = _normalize_text_basic(r[text_col] if r[text_col] is not None else "")
            hits = kp.extract_keywords(text, span_info=True)  # (value_json, start, end)

            if not hits:
                rows.append({
                    id_col: str(rid),
                    "matched_phrase": None,
                    "L1": None, "L2": None, "L3": None, "L4": None
                })
                continue

            for val_json, _s, _e in hits:
                # FlashText returns the stored value as first element
                try:
                    obj = orjson.loads(val_json) if isinstance(val_json, str) else val_json
                except Exception:
                    # fallback: treat value as string phrase
                    obj = {"phrase": str(val_json), "paths": [{"L1": None, "L2": None, "L3": None, "L4": None}]}

                matched_phrase = obj.get("phrase")
                paths = obj.get("paths", [])
                # Ensure paths is a list
                if not isinstance(paths, list):
                    paths = []

                if not paths:
                    rows.append({
                        id_col: str(rid),
                        "matched_phrase": matched_phrase,
                        "L1": None, "L2": None, "L3": None, "L4": None
                    })
                else:
                    for path in paths:
                        rows.append({
                            id_col: str(rid),
                            "matched_phrase": matched_phrase,
                            "L1": path.get("L1"),
                            "L2": path.get("L2"),
                            "L3": path.get("L3"),
                            "L4": path.get("L4"),
                        })

        out_frames.append(pl.DataFrame(rows))
        prog.progress((b + 1) / batches, text=f"Categorizing transcripts… {b + 1}/{batches} batches")

    prog.empty()
    return pl.concat(out_frames) if out_frames else pl.DataFrame(
        schema={id_col: pl.Utf8, "matched_phrase": pl.Utf8, "L1": pl.Utf8, "L2": pl.Utf8, "L3": pl.Utf8, "L4": pl.Utf8}
    )

def make_downloads(joined: pl.DataFrame, summary: pl.DataFrame) -> Tuple[bytes, bytes, bytes]:
    """Return (parquet_bytes, excel_bytes, csv_bytes). CSV via pandas to avoid Polars StringIO issues."""
    # Parquet
    pq_buf = io.BytesIO()
    joined.write_parquet(pq_buf, compression=PARQUET_COMPRESSION)
    pq_bytes = pq_buf.getvalue()

    # Excel (two sheets)
    xls_buf = io.BytesIO()
    with pd.ExcelWriter(xls_buf, engine="xlsxwriter") as xl:
        joined.to_pandas(use_pyarrow_extension_array=True).to_excel(xl, index=False, sheet_name="raw_with_categories")
        summary.to_pandas(use_pyarrow_extension_array=True).to_excel(xl, index=False, sheet_name="summary_by_category")
    xls_bytes = xls_buf.getvalue()

    # CSV via pandas
    csv_buf = io.StringIO()
    joined.to_pandas(use_pyarrow_extension_array=True).to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    return pq_bytes, xls_bytes, csv_bytes

# =========================
# ---- SIDEBAR / INPUT ----
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    cat_file = st.file_uploader("Categories (.csv or .xlsx)", type=["csv", "xlsx"])
    trn_file = st.file_uploader("Transcripts (.csv or .parquet)", type=["csv", "parquet"])

    phrase_col, id_col, text_col = None, None, None
    cat_df, trn_df = None, None

    # Categories upload -> show column selectors immediately
    if cat_file:
        cat_df = read_categories(cat_file.read(), cat_file.name)
        st.success(f"Categories loaded with {cat_df.height:,} rows")
        phrase_col = st.selectbox("Select Phrase column (usually L4 or 'phrase')", options=cat_df.columns)

    # Transcripts upload -> show column selectors immediately
    if trn_file:
        trn_df = read_transcripts_any(trn_file.read(), trn_file.name)
        st.success(f"Transcripts loaded with {trn_df.height:,} rows")
        cols = trn_df.columns
        id_col = st.selectbox("Select ID column (or <none>)", options=["<none>"] + list(cols))
        text_col = st.selectbox("Select Transcript Text column", options=cols)

    # Only show Process when all selections are available
    go = False
    if cat_file and trn_file and phrase_col and text_col:
        go = st.button("🚀 Process", type="primary", use_container_width=True)

# =========================
# --------- MAIN ----------
# =========================
if go:
    # 1) Build keyword index
    try:
        kp = build_keyword_processor(cat_df, phrase_col)
    except Exception as e:
        st.error(f"Failed to build keyword index: {e}")
        st.stop()

    # 2) If no ID column, generate one
    if id_col == "<none>":
        trn_df = trn_df.with_row_index(name="row_id")  # UInt32
        id_col = "row_id"

    # 3) Categorize (only need id + text columns)
    try:
        cat_only = categorize_df(trn_df.select([id_col, text_col]), kp, id_col, text_col)
        st.success(f"Categorized {cat_only.height:,} rows (matches expanded by paths)")
    except Exception as e:
        st.error(f"Categorization failed: {e}")
        st.stop()

    # 4) Cast join key to string on both sides, then join
    trn_df = trn_df.with_columns(pl.col(id_col).cast(pl.Utf8))
    cat_only = cat_only.with_columns(pl.col(id_col).cast(pl.Utf8))
    joined = trn_df.join(cat_only, on=id_col, how="left")

    # 5) Summary by L1..L4
    summary = (
        joined.group_by(["L1", "L2", "L3", "L4"])
              .agg(pl.len().alias("n_conversations"))
              .sort("n_conversations", descending=True)
    )

    # 6) Show preview + downloads
    st.write("### 📊 Summary by Category (Top 25)")
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
