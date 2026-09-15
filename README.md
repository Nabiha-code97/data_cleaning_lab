# Data Cleaning Lab

A Streamlit app that inspects, validates, cleans, and bands a small controlled
test dataset — and separately inspects a real Walmart supermarket dataset
downloaded live from Kaggle, reasoning about which data-mining task it
actually supports.

## What's in this folder

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app (everything: inspection, validation, cleaning, bands, export, Walmart tab) |
| `requirements.txt` | Python packages needed to run it |
| `cleaned_controlled_test_mean.csv` | Pre-generated cleaned output of the controlled dataset (mean imputation), so a cleaned CSV is included even without running the app |
| `walmart_archive.zip` | The Walmart dataset zip, ready to upload straight into the app's Walmart tab |

## How to run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

### Walmart tab — three ways to load the data
The Walmart tab lets you choose how to get the dataset onto disk:

1. **Upload the zip (recommended, no setup needed)** — pick
   `walmart_archive.zip` (included in this folder) in the file uploader.
   The app extracts it and inspects whatever files are inside.
2. **kagglehub download** — this is the standard `kagglehub` integration
   (`kagglehub.dataset_download(...)`). It requires your own Kaggle API
   credentials on the machine running the app:
   - Go to your Kaggle account settings → "Create New API Token" → this
     downloads a `kaggle.json` file.
   - Put it at `~/.kaggle/kaggle.json` (Linux/Mac) or
     `C:\Users\<you>\.kaggle\kaggle.json` (Windows).
   - Click **"📥 Download with kagglehub"** in the app.
3. **Local folder path** — if you've already extracted the zip somewhere,
   just paste the folder path.

If credentials aren't set up for option 2, the app shows a clear error
instead of crashing — it won't silently pretend the download worked.

### What's actually inside this Walmart dataset
The zip you provided (`walmart_archive.zip`) turned out to be the **Walmart
Recruiting — Store Sales Forecasting** style dataset, not a per-product
transaction file. It contains 5 linked CSVs inside a `Master Data/` folder:

| File | Columns | Role |
|---|---|---|
| `train.csv` | `Store, Dept, Date, Weekly_Sales, IsHoliday` | historical weekly sales per store/department |
| `test.csv` | `Store, Dept, Date, IsHoliday` | same structure, `Weekly_Sales` withheld (to be predicted) |
| `features.csv` | `Store, Date, Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment, IsHoliday` | external conditions per store/week |
| `stores.csv` | `Store, Type, Size` | static info about each store |
| `sampleSubmission.csv` | `Id, Weekly_Sales` | submission format for the original Kaggle competition |

**Reasoning-challenge answer:** no, a supermarket dataset does not
necessarily contain product baskets — this one doesn't. There's no
invoice/transaction ID and no product/item column anywhere in these five
files, only a store, a department, a date, and an aggregate sales figure.
Market-basket / association-rule mining is therefore **not supported** — you
can't reconstruct "what was bought together" when the data never recorded
individual items in the first place.

**Task this data actually supports:** **sales forecasting (regression /
time-series)** — predicting `Weekly_Sales` from `Store`, `Dept`, `Date`, and
`IsHoliday`, optionally enriched by joining in `features.csv` (weather,
fuel price, CPI, unemployment, markdowns) and `stores.csv` (store type and
size) on the shared `Store` (and `Date`) key. This matches the file names
themselves — `train`/`test`/`sampleSubmission` is the standard shape of a
predictive-modeling competition, not a clustering or basket-analysis dataset.

**Limitation:** `train.csv` alone only has the target and its keys — none of
the demand drivers (promotions, weather, local economy) live in that file.
A forecasting model built on `train.csv` in isolation would be missing
exactly the context most likely to explain *why* sales spike or dip, so
`features.csv` and `stores.csv` need to be joined in first.

The app's Walmart tab detects all of this automatically from whichever file
you select — it never assumes the dataset has product-level detail just
because it's a "supermarket" dataset.

## What the app actually does (plain-words walkthrough)

**Controlled test dataset (PassengerID, Age, Embarked):**

1. **Inspection** — shows the raw 10-row table, its shape, column types, and
   how many values are missing in each column.
2. **Validation** — two independent fixes:
   - `Embarked` gets whitespace trimmed (`" S "` → `"S"`), and anything left
     that isn't `S`/`C`/`Q` is treated as missing.
   - `Age` values outside `0–100` (e.g. `180`) are flagged and converted to
     missing **before** any average is calculated, so a bad value can't
     distort the statistic used to fix other missing values.
   - The app reports exactly how many values were changed by this step —
     tracked separately from "originally missing," so a `0` never hides an
     invalid value that got quietly dropped.
3. **Interactive cleaning** — you pick **mean** or **median** for `Age`
   imputation; `Embarked` is always filled with its **mode** (most common
   valid code) computed *after* standardization. The actual numbers used are
   displayed, not just "filled."
4. **Age bands** — every age (including a newborn, age `0`) is placed into
   one of four bands: `[0,18]`, `(18,40]`, `(40,65]`, `(65,100]`.
5. **Verification & export** — shows the cleaned table, confirms `0`
   remaining missing values and `0` unassigned bands, and lets you download
   the cleaned CSV.

Changing the mean/median choice re-runs **all** of the above from the
original raw data — Streamlit re-executes the whole script on every
interaction, and the pipeline always starts from the untouched original
dataframe, so there's no way for a stale, partially-cleaned version to leak
through.

**Walmart dataset:**

Nothing about its structure is assumed ahead of time — the Age/Embarked
rules above are specific to the controlled dataset and are **not** applied
here. The app:
- Downloads the dataset and walks the returned folder to find the actual
  data file(s) (it never assumes the `kagglehub` path *is* a CSV filename).
- Lets you pick a file if more than one is found.
- Runs the same inspection view (preview, shape, dtypes, missing counts).
- Looks at the columns that are actually present to decide whether
  **market-basket / association-rule mining** is realistic: that needs a
  transaction ID *and* multiple product rows per transaction. A supermarket
  dataset does not automatically have that — many public "supermarket
  sales" files are already aggregated to one row per transaction, in which
  case basket analysis isn't supported and the app proposes an alternative
  task (e.g. customer segmentation / clustering) grounded in whatever
  columns actually exist, along with a stated limitation.

## Why validation → imputation → banding must happen in that order

1. **Validation must come before imputation.** The mean/median for `Age` is
   only meaningful if it's computed from genuinely valid ages. Leaving the
   out-of-range value (`180`) in would pull the average upward before we've
   even started "cleaning." The same applies to `Embarked`: the mode has to
   be computed from *standardized* codes — if `" S "` were left untrimmed,
   it would be counted as a different category from `"S"`, which could
   dilute or even change which code the mode picks.
2. **Imputation must come before banding.** Age bands can only be assigned
   to a value that exists. If banding ran before missing ages were filled,
   every originally-missing row would end up with no band at all — silently
   dropping rows from any later group-based analysis instead of correctly
   classifying their (now-imputed) age.

In short: you can't trust statistics computed from dirty data, and you can't
categorize a value that isn't there yet — each step depends on the one
before it having already run.
