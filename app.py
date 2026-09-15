"""
Data Cleaning Lab — Streamlit App
==================================
A Streamlit app that catches cleaning mistakes.

This single-file app implements, for the controlled test dataset:
    1. Inspection
    2. Validation
    3. Interactive cleaning (imputation)
    4. Age bands
    5. Verification & export

...and, for the Walmart dataset (downloaded via kagglehub):
    - Dataset inspection
    - Mining-task identification driven by what the columns ACTUALLY are
      (no hard-coded assumption about baskets / association rules)

Design principle used throughout:
    Streamlit re-runs this whole script top-to-bottom every time a widget
    changes. So we ALWAYS rebuild the cleaned dataframe starting from the
    original, untouched data on every run. This is what makes "change the
    imputation dropdown -> recompute from scratch" work for free, with no
    manual cache-busting logic.
"""

import io
import os
import glob
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# Page config + theme
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Cleaning Lab",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --accent: #14b8a6;      /* teal */
    --accent-dark: #0f766e;
    --bg-card: #ffffff;
    --bg-soft: #f4faf9;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
}
section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stFileUploader label {
    color: #f9fafb !important;
    font-weight: 600;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15);
}

/* Headings */
h1, h2, h3 {
    color: #0f172a;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--bg-soft);
    border: 1px solid #d9efec;
    border-radius: 12px;
    padding: 12px 16px;
}

/* Section badge */
.section-badge {
    display: inline-block;
    background: var(--accent);
    color: white;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    padding: 3px 10px;
    border-radius: 999px;
    margin-bottom: 6px;
}

/* Info / warning callouts reused across the app */
.lab-note {
    background: #fff7ed;
    border-left: 4px solid #f59e0b;
    padding: 10px 14px;
    border-radius: 6px;
    margin: 8px 0px;
    font-size: 0.92rem;
}
.lab-good {
    background: #ecfdf5;
    border-left: 4px solid var(--accent);
    padding: 10px 14px;
    border-radius: 6px;
    margin: 8px 0px;
    font-size: 0.92rem;
}

/* Buttons */
.stDownloadButton button, .stButton button {
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
.stDownloadButton button:hover, .stButton button:hover {
    background: var(--accent-dark);
    color: white;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def section_header(step_no: str, title: str):
    st.markdown(f'<span class="section-badge">{step_no}</span>', unsafe_allow_html=True)
    st.subheader(title)


# --------------------------------------------------------------------------------------
# Controlled test dataset (embedded directly in the app)
# --------------------------------------------------------------------------------------
CONTROLLED_CSV = """PassengerID,Age,Embarked
P01,0,S
P02,18,C
P03,,S
P04,40,Q
P05,65,S
P06,100,C
P07,180,S
P08,22,
P09,," S "
P10,35,C
"""

VALID_EMBARKED_CODES = {"S", "C", "Q"}
AGE_MIN, AGE_MAX = 0, 100
AGE_BAND_BINS = [-0.001, 18, 40, 65, 100]
AGE_BAND_LABELS = ["0–18", "19–40", "41–65", "66–100"]


@st.cache_data
def load_controlled_data() -> pd.DataFrame:
    """Loads the raw controlled test data, completely untouched."""
    return pd.read_csv(io.StringIO(CONTROLLED_CSV))


def inspect_dataset(df: pd.DataFrame, label: str):
    """Shared inspection view: original data, shape, dtypes, missing counts."""
    st.markdown(f"**Preview — {label}**")
    st.dataframe(df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", df.shape[0])
    with c2:
        st.metric("Columns", df.shape[1])
    with c3:
        st.metric("Total missing values", int(df.isna().sum().sum()))

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Column data types**")
        st.dataframe(
            df.dtypes.astype(str).rename("dtype").to_frame(),
            use_container_width=True,
        )
    with col_b:
        st.markdown("**Missing values per column**")
        st.dataframe(
            df.isna().sum().rename("missing_count").to_frame(),
            use_container_width=True,
        )


# --------------------------------------------------------------------------------------
# Controlled-dataset cleaning pipeline (pure function -> always rebuilt from original)
# --------------------------------------------------------------------------------------
def run_controlled_pipeline(raw_df: pd.DataFrame, age_method: str) -> dict:
    """
    Runs the full validate -> impute -> band pipeline starting from the
    ORIGINAL raw dataframe every single time it's called. Nothing here is
    cached/mutated in place, so switching age_method always recomputes
    everything from scratch, by design.
    """
    df = raw_df.copy()

    # ---- 1. VALIDATION -------------------------------------------------
    # 1a. Trim whitespace in Embarked (NaNs pass through .str methods untouched)
    embarked_before = df["Embarked"].copy()
    df["Embarked"] = df["Embarked"].str.strip()
    embarked_trim_changed = int(
        ((embarked_before != df["Embarked"]) & ~(embarked_before.isna() & df["Embarked"].isna())).sum()
    )

    # Anything left that isn't S/C/Q (and isn't NaN) is also invalid -> treat as missing
    embarked_bad_code_mask = df["Embarked"].notna() & ~df["Embarked"].isin(VALID_EMBARKED_CODES)
    embarked_bad_code_count = int(embarked_bad_code_mask.sum())
    df.loc[embarked_bad_code_mask, "Embarked"] = np.nan

    # 1b. Flag invalid ages (outside 0-100 inclusive) and convert to missing
    #     BEFORE any imputation statistic is computed — order matters, see README.
    invalid_age_mask = df["Age"].notna() & ((df["Age"] < AGE_MIN) | (df["Age"] > AGE_MAX))
    invalid_ages_found = df.loc[invalid_age_mask, "Age"].tolist()
    age_invalidated_count = int(invalid_age_mask.sum())
    df.loc[invalid_age_mask, "Age"] = np.nan

    validation_report = {
        "embarked_trim_changed": embarked_trim_changed,
        "embarked_bad_code_count": embarked_bad_code_count,
        "invalid_ages_found": invalid_ages_found,
        "age_invalidated_count": age_invalidated_count,
        "total_values_changed": embarked_trim_changed + embarked_bad_code_count + age_invalidated_count,
    }

    # ---- 2. INTERACTIVE CLEANING (IMPUTATION) ---------------------------
    missing_age_before_impute = int(df["Age"].isna().sum())
    missing_embarked_before_impute = int(df["Embarked"].isna().sum())

    valid_ages = df["Age"].dropna()
    if age_method == "mean":
        age_fill_value = float(valid_ages.mean())
    else:
        age_fill_value = float(valid_ages.median())
    df["Age"] = df["Age"].fillna(age_fill_value)

    valid_embarked = df["Embarked"].dropna()
    if len(valid_embarked) > 0:
        embarked_fill_value = valid_embarked.mode().iloc[0]
    else:
        embarked_fill_value = None
    if embarked_fill_value is not None:
        df["Embarked"] = df["Embarked"].fillna(embarked_fill_value)

    cleaning_report = {
        "age_method": age_method,
        "age_fill_value": age_fill_value,
        "age_filled_count": missing_age_before_impute,
        "embarked_fill_value": embarked_fill_value,
        "embarked_filled_count": missing_embarked_before_impute,
    }

    # ---- 3. AGE BANDS ----------------------------------------------------
    df["AgeBand"] = pd.cut(df["Age"], bins=AGE_BAND_BINS, labels=AGE_BAND_LABELS, include_lowest=True)

    # ---- 4. VERIFICATION --------------------------------------------------
    remaining_missing = df.isna().sum()
    unassigned_bands = int(df["AgeBand"].isna().sum())

    verification_report = {
        "remaining_missing": remaining_missing,
        "unassigned_bands": unassigned_bands,
    }

    return {
        "cleaned_df": df,
        "validation_report": validation_report,
        "cleaning_report": cleaning_report,
        "verification_report": verification_report,
    }


def render_controlled_page():
    st.title("🧪 Controlled Test Dataset")
    st.caption(
        "A tiny 10-row PassengerID / Age / Embarked dataset, hand-built to contain "
        "every kind of mistake this lab needs to catch."
    )

    raw_df = load_controlled_data()

    # ---------------- Inspection ----------------
    section_header("STEP 1", "Inspection")
    inspect_dataset(raw_df, "Original, untouched data")
    st.markdown(
        '<div class="lab-note">Notice: <b>Age</b> has 2 missing values and one out-of-range '
        'value (180). <b>Embarked</b> has 1 missing value and one value with stray spaces '
        '(<code>" S "</code>) that a naive missing-value count would miss entirely.</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ---------------- Sidebar control lives here logically, rendered in sidebar ----------------
    age_method = st.session_state.get("age_method", "mean")

    result = run_controlled_pipeline(raw_df, age_method)
    vr = result["validation_report"]
    cr = result["cleaning_report"]
    ver = result["verification_report"]
    cleaned_df = result["cleaned_df"]

    # ---------------- Validation ----------------
    section_header("STEP 2", "Validation")
    v1, v2, v3 = st.columns(3)
    with v1:
        st.metric("Embarked values trimmed", vr["embarked_trim_changed"])
    with v2:
        st.metric("Ages flagged invalid → missing", vr["age_invalidated_count"])
    with v3:
        st.metric("Total values changed by validation", vr["total_values_changed"])

    if vr["invalid_ages_found"]:
        st.markdown(
            f'<div class="lab-note"><b>Invalid ages caught:</b> '
            f'{", ".join(str(a) for a in vr["invalid_ages_found"])} '
            f'(outside the valid 0–{AGE_MAX} range) — converted to missing '
            f'<i>before</i> the mean/median is calculated, so they cannot skew the statistic.</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="lab-good">Zero-missing-count trap avoided: the app tracks invalid values '
        'separately from "originally missing" values, so a 0 in the missing-count column can never '
        'hide the fact that a value was actually thrown out for being invalid.</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ---------------- Interactive cleaning ----------------
    section_header("STEP 3", "Interactive Cleaning")
    st.write(
        "Choose how missing/invalid **Age** values should be imputed. "
        "Changing this option recomputes the *entire* pipeline from the original data."
    )
    chosen = st.radio(
        "Age imputation method",
        options=["mean", "median"],
        index=0 if age_method == "mean" else 1,
        horizontal=True,
        key="age_method_radio",
    )
    if chosen != age_method:
        st.session_state["age_method"] = chosen
        st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            f"Age fill value used ({cr['age_method']})",
            f"{cr['age_fill_value']:.2f}",
            help=f"Applied to {cr['age_filled_count']} row(s) that were missing/invalid.",
        )
    with c2:
        st.metric(
            "Embarked fill value used (mode)",
            str(cr["embarked_fill_value"]),
            help=f"Applied to {cr['embarked_filled_count']} row(s) that were missing after standardization.",
        )

    st.divider()

    # ---------------- Age bands ----------------
    section_header("STEP 4", "Age Bands")
    st.write("Bands: `[0,18]`, `(18,40]`, `(40,65]`, `(65,100]` — every valid/imputed age, including 0, gets one.")
    band_counts = cleaned_df["AgeBand"].value_counts().reindex(AGE_BAND_LABELS, fill_value=0)
    st.bar_chart(band_counts)

    st.divider()

    # ---------------- Verification & export ----------------
    section_header("STEP 5", "Verification & Export")
    st.markdown("**Cleaned data**")
    st.dataframe(cleaned_df, use_container_width=True)

    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Remaining missing values (should be 0 everywhere)**")
        st.dataframe(ver["remaining_missing"].rename("missing_count").to_frame(), use_container_width=True)
    with v2:
        st.metric("Rows with no Age Band assigned", ver["unassigned_bands"])
        if ver["unassigned_bands"] == 0:
            st.markdown(
                '<div class="lab-good">✅ Every row has a valid Age Band — 0 unassigned, '
                'and this is a genuine zero, not a hidden failure.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="lab-note">⚠️ Some rows have no band. This would mean an imputed '
                'age fell outside 0–100, which should not happen with a valid mean/median of '
                'in-range data — worth investigating.</div>',
                unsafe_allow_html=True,
            )

    csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download cleaned CSV",
        data=csv_bytes,
        file_name=f"cleaned_controlled_test_{cr['age_method']}.csv",
        mime="text/csv",
    )

    with st.expander("Why must validation → imputation → banding happen in that exact order?"):
        st.markdown(
            """
- **Validation before imputation:** the mean/median must be computed from
  *genuinely valid* ages only. If the out-of-range value (180) were left in,
  it would drag the mean upward before we even started cleaning — so we must
  flag and null it out first. The same logic applies to Embarked: the mode
  must be computed from *standardized* codes. If `" S "` (with spaces) were
  left untrimmed, it would count as a different category from `"S"`, and a
  naive mode calculation could get diluted or even pick the wrong code.
- **Imputation before banding:** age bands can only be assigned to a value
  that exists. If we tried to band the data before filling missing ages, every
  originally-missing row would get an "unassigned band" instead of an actual
  band — silently losing data from later analysis. Banding must be the last
  step so every row, including imputed ones, lands in a band.
            """
        )


# --------------------------------------------------------------------------------------
# Walmart dataset (uploaded zip / local folder / kagglehub download)
# --------------------------------------------------------------------------------------
def _walk_data_files(root_path: str) -> list:
    """Finds every csv/xlsx/xls/json/tsv file under the dataset directory.
    We never assume a returned path IS a data file — it's usually a folder."""
    exts = (".csv", ".xlsx", ".xls", ".json", ".tsv")
    found = []
    for dirpath, _dirs, filenames in os.walk(root_path):
        for fname in filenames:
            if fname.lower().endswith(exts):
                found.append(os.path.join(dirpath, fname))
    return sorted(found)


@st.cache_data(show_spinner=False)
def download_walmart_dataset():
    """Downloads the Kaggle dataset via kagglehub. Requires kaggle credentials
    to be configured on the machine running this app (see README)."""
    import kagglehub

    path = kagglehub.dataset_download("saurabhbadole/walmart-super-market-dataset")
    return path


def extract_uploaded_zip(uploaded_file) -> str:
    """Saves an uploaded zip to a temp dir and extracts it. Returns the extraction folder."""
    extract_dir = os.path.join(tempfile.gettempdir(), "walmart_upload_extract")
    if os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    zip_path = os.path.join(extract_dir, "upload.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


@st.cache_data(show_spinner=False)
def load_data_file(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if path.lower().endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    if path.lower().endswith(".json"):
        return pd.read_json(path)
    return pd.read_csv(path)


def guess_column(columns, keywords):
    """Returns the first column whose name contains any of the given keywords
    (case-insensitive). Used to reason about the dataset generically instead
    of hard-coding exact column names, since the real file must be inspected."""
    for col in columns:
        low = str(col).lower()
        for kw in keywords:
            if kw in low:
                return col
    return None


def _get_walmart_dataset_path():
    """Handles the three ways of getting the dataset onto disk and returns a
    folder/file path, or None if nothing is loaded yet."""
    st.markdown("**How do you want to load the Walmart dataset?**")
    mode = st.radio(
        "Load method",
        [
            "📦 Upload the dataset zip (e.g. archive.zip from Kaggle's website — no API key needed)",
            "☁️ Download with kagglehub (requires a Kaggle API key)",
            "📁 Point to a folder already extracted on this machine",
        ],
        label_visibility="collapsed",
    )

    if mode.startswith("📦"):
        uploaded = st.file_uploader("Upload the dataset .zip file", type=["zip"])
        if uploaded is None:
            st.info("Upload the .zip you downloaded from Kaggle's website to continue.")
            return None
        with st.spinner("Extracting zip..."):
            return extract_uploaded_zip(uploaded)

    if mode.startswith("☁️"):
        st.markdown(
            '<div class="lab-note">Needs a <code>kaggle.json</code> file (or '
            '<code>KAGGLE_USERNAME</code>/<code>KAGGLE_KEY</code> env vars) set up on this '
            'machine. See the README for setup steps. This is the standard kagglehub '
            'download integration.</div>',
            unsafe_allow_html=True,
        )
        if not st.button("📥 Download with kagglehub", type="primary"):
            st.info("Click the button to download the dataset via kagglehub.")
            return None
        try:
            with st.spinner("Downloading dataset via kagglehub..."):
                return download_walmart_dataset()
        except Exception as exc:  # noqa: BLE001 - surfaced directly to the student
            st.error(
                "Could not download the dataset via kagglehub. This usually means Kaggle "
                "credentials aren't configured, or there's no internet access from this "
                f"machine.\n\nDetails: {exc}"
            )
            return None

    # Local folder path
    folder = st.text_input(
        "Folder path where you already extracted the dataset",
        placeholder="e.g. C:/Users/you/Downloads/archive or /home/you/Downloads/archive",
    )
    if not folder:
        st.info("Paste a folder path above to continue.")
        return None
    if not os.path.exists(folder):
        st.error(f"That path doesn't exist: `{folder}`")
        return None
    return folder


def render_walmart_page():
    st.title("🛒 Walmart Dataset")
    st.caption(
        "Nothing about this dataset's structure is assumed in advance. Age/Embarked "
        "rules from the controlled dataset do NOT apply here."
    )

    dataset_path = _get_walmart_dataset_path()
    if not dataset_path:
        return

    st.success(f"Dataset ready at: `{dataset_path}`")

    # Do NOT assume the path IS a data file — inspect the directory.
    if os.path.isdir(dataset_path):
        data_files = _walk_data_files(dataset_path)
    elif os.path.isfile(dataset_path):
        data_files = [dataset_path]
    else:
        data_files = []

    if not data_files:
        st.error("No CSV/XLSX/JSON/TSV files were found in that location.")
        return

    st.markdown("**Files found:**")
    st.code("\n".join(os.path.relpath(f, dataset_path) if os.path.isdir(dataset_path) else f for f in data_files))

    if len(data_files) > 1:
        st.markdown(
            '<div class="lab-note">This dataset is split across <b>multiple related files</b> '
            'rather than one flat table. Pick one below to inspect — a real analysis would '
            'eventually join them together (e.g. on a shared <code>Store</code> / <code>Date</code> key) '
            'rather than treating each in isolation.</div>',
            unsafe_allow_html=True,
        )
        chosen_file = st.selectbox(
            "Choose which file to inspect:",
            options=data_files,
            format_func=lambda p: os.path.relpath(p, dataset_path) if os.path.isdir(dataset_path) else p,
        )
    else:
        chosen_file = data_files[0]

    st.markdown(f"**Selected file:** `{os.path.basename(chosen_file)}`")

    try:
        wdf = load_data_file(chosen_file)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load `{chosen_file}` into pandas. Details: {exc}")
        return

    st.divider()
    section_header("INSPECTION", "Walmart Dataset Overview")
    inspect_dataset(wdf, os.path.basename(chosen_file))

    st.markdown(
        '<div class="lab-note">A missing-value count of zero does <b>not</b> mean this data is '
        'clean — it only means pandas didn\'t detect a blank cell. Placeholder text like '
        '"NA" stored as a string, inconsistent capitalization, duplicate rows, or mixed units '
        'can all hide from <code>isna()</code>.</div>',
        unsafe_allow_html=True,
    )

    # ---------------- Mining task reasoning (driven by the ACTUAL columns) ----------------
    st.divider()
    section_header("REASONING", "Which data-mining task actually fits this data?")

    columns = list(wdf.columns)
    invoice_col = guess_column(columns, ["invoice", "transaction", "order", "receipt"])
    product_col = guess_column(columns, ["product", "item", "stockcode", "sku"])
    qty_col = guess_column(columns, ["quantity", "qty"])
    price_col = guess_column(columns, ["price", "unitprice", "unit price"])
    customer_col = guess_column(columns, ["customer", "gender", "membership"])
    rating_col = guess_column(columns, ["rating", "satisfaction"])
    target_col = guess_column(columns, ["sales", "revenue", "amount", "total"])
    date_col = guess_column(columns, ["date"])
    store_col = guess_column(columns, ["store", "branch", "shop"])
    dept_col = guess_column(columns, ["dept", "department", "category"])

    basket_supported = False
    if invoice_col is not None and product_col is not None:
        rows_per_invoice = wdf.groupby(invoice_col)[product_col].count()
        repeated_invoices = int((rows_per_invoice > 1).sum())
        basket_supported = repeated_invoices > 0
    else:
        repeated_invoices = None

    st.write(f"- Transaction/invoice-style column: **{invoice_col or 'none found'}**")
    st.write(f"- Product/item-style column: **{product_col or 'none found'}**")
    if repeated_invoices is not None:
        st.write(f"- Invoices/transactions with more than one product row: **{repeated_invoices}**")

    if invoice_col is None or product_col is None:
        st.markdown(
            '<div class="lab-note"><b>Association / market-basket mining is NOT supported by this file.</b> '
            'A basket-analysis task needs a transaction identifier plus the individual items bought '
            'within that transaction. This file has neither — a supermarket dataset does not '
            'automatically contain product-basket data.</div>',
            unsafe_allow_html=True,
        )
    elif basket_supported:
        st.markdown(
            '<div class="lab-good"><b>Association / market-basket mining IS supported:</b> '
            f'multiple rows share the same <code>{invoice_col}</code>, each naming a different '
            f'<code>{product_col}</code> — exactly the structure association rule mining '
            '(e.g. Apriori) needs.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="lab-note"><b>Association / market-basket mining is NOT supported here, even '
            f'though an <code>{invoice_col}</code> column exists</b> — every invoice only has one '
            'row/product in this file, so there\'s no record of what was bought together.</div>',
            unsafe_allow_html=True,
        )

    # ---- Alternative task, grounded in whatever columns actually exist ----
    if basket_supported:
        st.markdown(
            f"**Required columns:** `{invoice_col}` (transaction id) and `{product_col}` (item bought)"
            + (f", plus `{qty_col}`" if qty_col else "")
            + " to weight how strongly items co-occur."
        )
        st.markdown(
            "- *Limitation:* needs enough transactions with 2+ items for meaningful support/confidence — "
            "if most invoices only contain one product line, any discovered rule won't be very actionable."
        )
    elif target_col and (date_col or store_col):
        # This is the shape of the real Kaggle "Walmart Recruiting" style dataset: no
        # product-level detail at all, just a numeric target (sales) tracked over time
        # per store/department — a forecasting/regression problem, not a basket problem.
        req_cols = [c for c in [store_col, dept_col, date_col, target_col] if c]
        st.markdown(
            f'<div class="lab-good"><b>Sales forecasting / regression IS supported:</b> '
            f'<code>{target_col}</code> is a numeric outcome tracked per '
            f'{"store/department" if dept_col else "store"} over <code>{date_col or "an implied time order"}</code> — '
            'exactly what a regression or time-series forecasting task needs.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Required columns:** `{'`, `'.join(req_cols)}`" + (" (plus `IsHoliday` if present)" if "IsHoliday" in columns else ""))
        st.markdown(
            "- *Limitation:* this file alone only has the target and its keys — no external "
            "drivers of demand (promotions, pricing, weather, local economic conditions). "
            "If this dataset ships with companion files (e.g. `features.csv`, `stores.csv`), "
            f"those would need to be **joined on `{store_col}`{' / ' + date_col if date_col else ''}** "
            "to bring in that context before a forecasting model would be genuinely useful."
        )
    elif customer_col and (product_col or price_col or rating_col):
        alt_task_cols = [c for c in [product_col, qty_col, price_col, customer_col, rating_col] if c]
        st.markdown(
            f"**A task this data actually supports — customer segmentation / clustering** "
            f"using columns such as `{', '.join(alt_task_cols)}`, grouping customers or "
            "transactions by spending pattern or preference to find natural segments."
        )
        st.markdown(
            "- *Limitation:* without a persistent customer ID across visits, segments describe "
            "**transactions**, not necessarily the **same customer over time**."
        )
    elif rating_col:
        other_cols = [c for c in columns if c != rating_col][:5]
        st.markdown(
            f"**A task this data actually supports — classification/regression** predicting "
            f"`{rating_col}` from `{', '.join(other_cols)}`."
        )
    else:
        st.markdown(
            f"**A task this data actually supports — descriptive/summary analysis** "
            f"over the available columns (`{', '.join(columns[:6])}`) — e.g. trends by category "
            "or time period, since no clear transaction, target, or customer structure was detected."
        )


# --------------------------------------------------------------------------------------
# Overview / About page
# --------------------------------------------------------------------------------------
def render_overview_page():
    st.title("🧹 Data Cleaning Lab")
    st.caption("A Streamlit app that catches cleaning mistakes")

    st.markdown(
        """
### What this app does, in plain words

Think of this app as a small **data doctor**. You give it messy data, and it:

1. **Looks at the data first** (Inspection) — how big is it, what types are the
   columns, and what's actually missing?
2. **Checks the rules are followed** (Validation) — an age of `180` is
   impossible, so it gets pulled out and treated as missing *before* we do any
   math with it. A port code like `" S "` (with stray spaces) is really just
   `"S"`, so we clean the formatting first.
3. **Fills in the gaps** (Interactive cleaning) — you choose whether missing
   ages get filled with the **mean** or the **median**, and missing embarkation
   codes get filled with the **most common value (mode)**.
4. **Groups ages into bands** (Age bands) — so every passenger, including a
   newborn (age 0), falls into a clear age group.
5. **Double-checks its own work** (Verification & export) — makes sure nothing
   is *quietly* still broken, and lets you download the cleaned file.

The **Walmart tab** does something different: instead of following fixed rules
(there is no "valid age range" for a supermarket dataset), it actually
**looks at whatever columns the real file has** and reasons about what kind of
analysis they support — for example, whether the data has enough detail to do
market-basket analysis ("customers who bought X also bought Y") or whether a
different task like customer segmentation is a better fit.

### Why this order matters (short version)
Validate → Impute → Band. You can't compute a trustworthy average from data
that still contains impossible values, and you can't put a "missing" age into
an age band. Each step needs the one before it to have already happened.
        """
    )

    st.markdown(
        '<div class="lab-good">Use the sidebar to switch between the Controlled Test Data and '
        'the Walmart Dataset pages.</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧹 Data Cleaning Lab")
    st.caption("Inspect, validate, clean, and export — the whole pipeline")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🧪 Controlled Test Data", "🛒 Walmart Dataset"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Built with Streamlit")

if page == "🏠 Overview":
    render_overview_page()
elif page == "🧪 Controlled Test Data":
    render_controlled_page()
else:
    render_walmart_page()
