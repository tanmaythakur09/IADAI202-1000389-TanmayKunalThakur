# 🏥 Hospital Inpatient Discharges Dashboard

**MediScope Health Analytics · SA-1 Summative Assessment, Scenario 2**
CRS: Artificial Intelligence · Course: Mathematics for AI-II

| | |
|---|---|
| **Student** | Tanmay Kunal Thakur |
| **Candidate registration number** | 1000389 |
| **School** | Viraj International School |

> ### 🔗 Live app: **https://YOUR-APP-NAME.streamlit.app**
> *(replace this line with your real Streamlit Cloud link after you deploy)*

---

## 1. Project overview

An interactive Streamlit dashboard that helps **hospital administrators, clinicians and operations managers**
understand inpatient discharges: which diagnoses keep patients longest, what drives the bill, and where
length of stay differs by county, facility and payer.

**Data:** New York State SPARCS de-identified *Hospital Inpatient Discharges, 2021* - 2,101,588 stays, 33 columns
(demographics, diagnosis, severity of illness, payer, length of stay, charges and costs).
The original file is about 794 MB, which is too large for GitHub and Streamlit Cloud, so this repository contains a
**reproducible 5 % random sample** (~105,000 rows, fixed seed). See `scripts/make_sample.py`.

### The five research questions

| # | Question | Answered by |
|---|---|---|
| 1 | Which diagnosis codes are linked to long stays? | Chart 1 (bar) |
| 2 | Which age groups incur the highest charges? | EDA tab: age × gender chart |
| 3 | How does patient severity affect total cost? | Chart 2 (box plot) |
| 4 | Are there counties with longer average stays? | Chart 3 (heatmap) |
| 5 | How do different payment types affect the final billing? | Chart 4 (pie + bar) |

The *EDA & Insights* tab writes a plain-English answer to each question that **updates with the sidebar filters**.

## 2. Key features

- **Sidebar filters** - region, county, age group, gender, severity, payer, admission type, length-of-stay range,
  and an outlier switch. Every KPI, chart and insight recalculates.
- **Five required interactive Plotly charts** (hover, zoom, pan, download as PNG):
  1. **Bar chart** - average stay per diagnosis code (choose how many codes and the minimum case count)
  2. **Box plot** - total charges by severity (log-scale and outlier-point switches; dashed line = mean)
  3. **Heatmap** - average stay by facility × county, or county × severity
  4. **Pie chart** - patients by payment type, with average charges per payer beside it
  5. **Histogram** - distribution of length of stay (bin width, log scale, mean and median markers)
- **EDA & Insights tab** - statistical summary (mean, 95 % CI, median, SD, IQR, skewness), Spearman correlation
  of stay vs charges, `groupby` / `pivot_table` analyses, charges by age group and gender, high-utilisation facilities.
- **Data Cleaning tab** - a live log of every cleaning step with row counts, missing-value chart and engineered columns.
- **Problem & Research tab** - users, decisions, research questions, background research and limits.
- Every chart has a **"View chart data"** table, and colours come from a colour-blind-checked palette.

## 3. Data preprocessing (done in `load_and_clean()` in `app.py`)

| Step | What was done |
|---|---|
| Missing values | Dropped columns with more than 50 % empty values (Birth Weight, Payment Typology 2 and 3) |
| Column names | Renamed unclear headers (e.g. `APR Severity of Illness Description` → `Severity`); dropped ID, zip and constant columns |
| Length of stay | Converted text to numbers (`"120 +"` → 120); removed zero, negative or missing stays |
| Money | Converted `"38,439.59"` strings to floats; removed non-positive charges / costs |
| Categories | Standardised gender, race, ethnicity, age group and severity (ordered categories); fixed the truncated region name `Capital/Adirond` |
| Diagnosis codes | Mapped the first three letters of each CCSR code to a body-system category (e.g. `CIR` → Circulatory system) |
| New features | Stay Band, Charge per Day, Cost-to-Charge Ratio, Payer Group, Dx Category |
| Outliers | Flagged charges outside 1.5 × IQR fences (log scale) and stays above the 99th percentile; a sidebar switch removes them |

## 4. Integration details

```
data/hospital_sample.csv.gz
        │  pandas.read_csv  (everything read as text so nothing is mangled)
        ▼
load_and_clean()  ──  cached with @st.cache_data, runs once per session
        │  returns: clean DataFrame, cleaning log, missing-value table, summary
        ▼
apply_filters()   ──  sidebar choices → boolean mask on the DataFrame
        ▼
chart_*() functions (Plotly Express)  +  research_answers() / describe_numbers()
        ▼
Streamlit layout: KPI row + 4 tabs
```

| Library | Used for |
|---|---|
| `streamlit` | App layout, filters, tabs, deployment |
| `pandas` | Loading, cleaning, `groupby`, `pivot_table` |
| `numpy` | Log transforms, confidence intervals, masks |
| `plotly` | The interactive charts |

## 5. Repository structure

```
.
├── app.py                  # the whole dashboard
├── requirements.txt        # Python packages for Streamlit Cloud
├── README.md
├── .streamlit/config.toml  # theme colour
├── data/
│   └── hospital_sample.csv.gz   # 5 % sample of the SPARCS 2021 file
├── scripts/
│   └── make_sample.py      # re-creates the sample from the full 794 MB file
└── screenshots/            # images used below
```

## 6. Run it on your own computer

```bash
git clone https://github.com/<your-username>/<this-repo>.git
cd <this-repo>
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501.

## 7. Deployment instructions (Streamlit Community Cloud)

1. Push this repository to GitHub (make sure `data/hospital_sample.csv.gz` is included - it is about 6 MB).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **Create app** (or **Deploy an app**) → **Deploy a public app from GitHub**.
4. Choose this repository, branch `main`, and main file path `app.py`.
5. Click **Deploy**. The first build takes a few minutes while it installs `requirements.txt`.
6. Copy the `https://….streamlit.app` address into the **Live app** line at the top of this README.

## 8. Screenshots

| Dashboard | EDA & Insights |
|---|---|
| ![Dashboard](screenshots/dashboard.png) | ![EDA](screenshots/eda.png) |

*(add your own screenshots of the live app to the `screenshots/` folder with these file names)*

## 9. Limitations

- The data has **no patient ID or readmission flag**, so readmission rates (mentioned in the scenario) cannot be
  calculated; length of stay, charges and payer mix are analysed instead.
- Results use a **5 % random sample**, so very small groups (one county or a rare diagnosis) can be noisy -
  use the minimum-cases slider and the blank-cell rule in the heatmaps.
- Charges are billed amounts, not what insurers actually paid.

## 10. Sources

- [NY SPARCS - Statewide Planning and Research Cooperative System](https://www.health.ny.gov/statistics/sparcs/)
- [AHRQ HCUP - Clinical Classifications Software Refined (CCSR)](https://hcup-us.ahrq.gov/toolssoftware/ccsr/ccs_refined.jsp)
- [Streamlit blog - Improving healthcare management with Streamlit](https://blog.streamlit.io/improving-healthcare-management-with-streamlit/)
- [Plotly Express documentation](https://plotly.com/python/plotly-express/)
