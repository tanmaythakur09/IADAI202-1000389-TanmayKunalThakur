"""
MediScope Health Analytics - Hospital Inpatient Discharges Dashboard
=====================================================================
Course : Mathematics for AI-II (CRS: Artificial Intelligence)
Task   : SA-1, Scenario 2 - Hospital Inpatient Discharges Dashboard

Data   : New York State SPARCS de-identified inpatient discharges, 2021.
         The full file has 2,101,588 rows (about 794 MB), which is too big for
         GitHub and Streamlit Cloud, so this app ships with a reproducible
         5 % random sample (see scripts/make_sample.py).

The file is organised in the same order the assignment asks for:
    1. Settings and lookup tables
    2. Data loading and cleaning          (Step 2)
    3. Helper functions
    4. Chart functions                    (Step 4 - the 5 required visuals)
    5. Insight / EDA functions            (Step 3)
    6. Page layout: sidebar filters + four tabs
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =============================================================================
# 1. SETTINGS AND LOOKUP TABLES
# =============================================================================
st.set_page_config(
    page_title="MediScope | Hospital Discharges",
    page_icon="🏥",
    layout="wide",
)

DATA_FILE = Path(__file__).parent / "data" / "hospital_sample.csv.gz"
FULL_FILE_ROWS = 2_101_588   # rows in the original 794 MB file
NULL_LIMIT = 0.50            # drop a column if more than 50 % of it is empty
MIN_DX_CASES = 30            # minimum discharges before a diagnosis is ranked

# Ordered categories (so charts and tables sort Minor -> Extreme, young -> old)
SEVERITY_ORDER = ["Minor", "Moderate", "Major", "Extreme"]
AGE_ORDER = ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"]
STAY_BANDS = ["1-2 days", "3-7 days", "8-14 days", "15+ days"]

# One colour per entity, taken from a colour-blind-checked palette.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
VIOLET, RED, GREEN, GREY = "#6250d6", "#e34948", "#008300", "#8a8985"
SEVERITY_COLORS = {"Minor": "#86b6ef", "Moderate": "#3987e5", "Major": "#1c5cab", "Extreme": "#0d366b"}
HEAT_SCALE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # one-hue, light -> dark
PAYER_COLORS = {
    "Medicare": BLUE, "Medicaid": ORANGE, "Private Health Insurance": AQUA,
    "Blue Cross/Blue Shield": YELLOW, "Managed Care, Unspecified": MAGENTA,
    "Self-Pay": VIOLET, "Federal/State/Local/VA": RED,
    "Miscellaneous/Other": GREEN, "Department of Corrections": GREY, "Other payers": GREY,
}

# CCSR diagnosis codes start with a 3-letter body-system prefix defined by AHRQ.
# This table turns "CIR012" into a readable category such as "Circulatory system".
CCSR_BODY_SYSTEMS = {
    "BLD": "Blood & immune disorders", "CIR": "Circulatory system", "DEN": "Dental",
    "DIG": "Digestive system", "EAR": "Ear & mastoid", "END": "Endocrine & metabolic",
    "EXT": "External causes", "EYE": "Eye & adnexa", "FAC": "Health-status factors",
    "GEN": "Genitourinary system", "INF": "Infectious & parasitic", "INJ": "Injury & poisoning",
    "MAL": "Congenital conditions", "MBD": "Mental & behavioural", "MUS": "Musculoskeletal system",
    "NEO": "Neoplasms (tumours)", "NVS": "Nervous system", "PNL": "Perinatal conditions",
    "PRG": "Pregnancy & childbirth", "RSP": "Respiratory system", "SKN": "Skin disorders",
    "SYM": "Symptoms & signs (not elsewhere classified)",
}

# Raw (unclear) column name -> readable name. Columns not listed here are dropped.
COLUMN_RENAMES = {
    "Hospital Service Area": "Region",
    "Hospital County": "County",
    "Facility Name": "Facility",
    "Age Group": "Age Group",
    "Gender": "Gender",
    "Race": "Race",
    "Ethnicity": "Ethnicity",
    "Length of Stay": "Stay Days",
    "Type of Admission": "Admission Type",
    "Patient Disposition": "Disposition",
    "CCSR Diagnosis Code": "Dx Code",
    "CCSR Diagnosis Description": "Dx Description",
    "APR DRG Description": "DRG Description",
    "APR MDC Description": "MDC Description",
    "APR Severity of Illness Description": "Severity",
    "APR Risk of Mortality": "Mortality Risk",
    "APR Medical Surgical Description": "Case Type",
    "Payment Typology 1": "Primary Payer",
    "Emergency Department Indicator": "Emergency Dept",
    "Total Charges": "Total Charges",
    "Total Costs": "Total Costs",
}

PAYER_GROUPS = {
    "Medicare": "Government", "Medicaid": "Government", "Federal/State/Local/VA": "Government",
    "Department of Corrections": "Government",
    "Private Health Insurance": "Commercial", "Blue Cross/Blue Shield": "Commercial",
    "Managed Care, Unspecified": "Commercial", "Self-Pay": "Self-pay",
}


# =============================================================================
# 2. DATA LOADING AND CLEANING  (assignment Step 2)
# =============================================================================
@st.cache_data(show_spinner="Loading and cleaning the discharge data...")
def load_and_clean(path: str):
    """Read the CSV and return (clean_df, cleaning_log, missing_table, summary).

    Every step records what it changed so the 'Data Cleaning' tab can show it.
    """
    raw = pd.read_csv(path, dtype=str)           # read as text so nothing is mangled
    log = []                                     # one dict per cleaning step
    summary = {"raw_rows": len(raw), "raw_cols": raw.shape[1]}

    def note(step, action, affected):
        log.append({"Step": step, "What was done": action, "Rows / columns affected": affected})

    # --- 1. Missing values: measure, then drop null-heavy columns ------------
    missing = raw.isna().mean().sort_values(ascending=False)
    missing_table = (missing[missing > 0].mul(100).round(1)
                     .rename("Missing %").rename_axis("Column").reset_index())
    heavy = missing[missing > NULL_LIMIT].index.tolist()
    raw = raw.drop(columns=heavy)
    note("Drop null-heavy columns", f"Removed columns with more than {NULL_LIMIT:.0%} empty values: "
         + (", ".join(heavy) if heavy else "none"), f"{len(heavy)} columns")

    # --- 2. Keep useful columns, give them readable names --------------------
    ignored = [c for c in raw.columns if c not in COLUMN_RENAMES]
    df = raw[[c for c in COLUMN_RENAMES if c in raw.columns]].rename(columns=COLUMN_RENAMES).copy()
    note("Rename and select columns", "Renamed unclear headers (e.g. 'APR Severity of Illness Description' -> "
         "'Severity') and dropped ID / constant / zip columns: " + ", ".join(ignored), f"{len(ignored)} columns")

    # --- 3. Length of stay: text -> number, remove invalid values ------------
    topcoded = (df["Stay Days"].str.strip() == "120 +").sum()
    df["Stay Days"] = pd.to_numeric(df["Stay Days"].str.replace("+", "", regex=False).str.strip(), errors="coerce")
    note("Length of stay", "Source writes very long stays as '120 +'; converted to the number 120",
         f"{topcoded:,} rows")
    bad_stay = df["Stay Days"].isna() | (df["Stay Days"] <= 0)
    df = df[~bad_stay].copy()
    note("Invalid stay", "Removed rows with missing, zero or negative length of stay", f"{bad_stay.sum():,} rows")

    # --- 4. Money: '38,439.59' (text) -> float -------------------------------
    for col in ["Total Charges", "Total Costs"]:
        df[col] = pd.to_numeric(df[col].str.replace(r"[$,\s]", "", regex=True), errors="coerce")
    bad_money = df["Total Charges"].isna() | df["Total Costs"].isna() | (df["Total Charges"] <= 0) | (df["Total Costs"] <= 0)
    df = df[~bad_money].copy()
    note("Currency to float", "Stripped '$' and ',' from Total Charges / Total Costs and converted to numbers; "
         "removed rows that were missing or not positive", f"{bad_money.sum():,} rows removed")

    # --- 5. Standardise categorical fields ----------------------------------
    df["Gender"] = df["Gender"].str.strip().map({"M": "Male", "F": "Female"}).fillna("Unknown")
    df["Race"] = df["Race"].str.strip().replace({"Other Race": "Other", "Multi-racial": "Multiracial"}).fillna("Unknown")
    df["Ethnicity"] = (df["Ethnicity"].str.strip()
                       .replace({"Not Span/Hispanic": "Not Hispanic", "Spanish/Hispanic": "Hispanic"}).fillna("Unknown"))
    df["Admission Type"] = df["Admission Type"].str.strip().replace({"Not Available": "Unknown"}).fillna("Unknown")
    df["Emergency Dept"] = df["Emergency Dept"].map({"Y": "Yes", "N": "No"}).fillna("Unknown")
    df["Primary Payer"] = df["Primary Payer"].str.strip().fillna("Unknown")
    df["Disposition"] = df["Disposition"].str.strip().fillna("Unknown")
    df["Case Type"] = df["Case Type"].str.strip()

    # Geography: fix the truncated region name and label redacted hospitals
    df["Region"] = df["Region"].replace({"Capital/Adirond": "Capital/Adirondack"}).fillna("Not Disclosed")
    df["County"] = df["County"].fillna("Not Disclosed")
    df["Facility"] = df["Facility"].str.strip().replace({"Redacted for Confidentiality": "Not Disclosed"})
    redacted = (df["Facility"] == "Not Disclosed").sum()
    note("Standardise categories", "Gender M/F -> Male/Female, Race and Ethnicity shortened, truncated region name "
         "'Capital/Adirond' fixed, redacted facilities labelled 'Not Disclosed' (kept, but left out of county / "
         "facility charts)", f"{redacted:,} redacted rows")

    # Severity: keep only valid levels, store as an ordered category
    df["Severity"] = df["Severity"].str.strip().str.title()
    bad_sev = ~df["Severity"].isin(SEVERITY_ORDER)
    df = df[~bad_sev].copy()
    note("Severity", "Removed rows with no severity of illness (code 0 / 'Not Applicable')", f"{bad_sev.sum():,} rows")
    df["Severity"] = pd.Categorical(df["Severity"], categories=SEVERITY_ORDER, ordered=True)
    df["Mortality Risk"] = pd.Categorical(df["Mortality Risk"].str.strip().str.title(),
                                          categories=SEVERITY_ORDER, ordered=True)
    df["Age Group"] = pd.Categorical(df["Age Group"].str.strip(), categories=AGE_ORDER, ordered=True)

    # --- 6. Diagnosis codes -> readable medical categories -------------------
    df["Dx Category"] = df["Dx Code"].str[:3].map(CCSR_BODY_SYSTEMS).fillna("Unclassified")
    df["Dx Description"] = df["Dx Description"].str.title().fillna("Not recorded")
    df["Dx Code"] = df["Dx Code"].fillna("N/A")
    note("Diagnosis mapping", "Mapped the 3-letter CCSR prefix of every diagnosis code to a body-system category "
         "(e.g. CIR -> Circulatory system) and tidied the descriptions",
         f"{df['Dx Code'].nunique()} codes -> {df['Dx Category'].nunique()} categories")

    # --- 7. Feature engineering ---------------------------------------------
    df["Stay Band"] = pd.cut(df["Stay Days"], bins=[0, 2, 7, 14, np.inf], labels=STAY_BANDS)
    df["Charge per Day"] = df["Total Charges"] / df["Stay Days"]
    df["Cost-to-Charge Ratio"] = df["Total Costs"] / df["Total Charges"]
    df["Payer Group"] = df["Primary Payer"].map(PAYER_GROUPS).fillna("Other")
    note("New columns", "Stay Band, Charge per Day, Cost-to-Charge Ratio, Payer Group, Dx Category", "5 new columns")

    # --- 8. Outliers: flag (not delete) so the user can switch them on / off --
    log_charge = np.log10(df["Total Charges"])
    q1, q3 = log_charge.quantile([0.25, 0.75])
    fence = 1.5 * (q3 - q1)                       # Tukey fence on the log scale (charges are very skewed)
    df["Charge Outlier"] = (log_charge < q1 - fence) | (log_charge > q3 + fence)
    stay_cap = df["Stay Days"].quantile(0.99)     # stays are right-skewed: flag the longest 1 %
    df["Stay Outlier"] = df["Stay Days"] > stay_cap
    df["Outlier"] = df["Charge Outlier"] | df["Stay Outlier"]
    note("Outliers", f"Flagged charges outside the 1.5 x IQR fences (log scale) and stays longer than the 99th "
         f"percentile ({stay_cap:.0f} days). The sidebar switch removes them from every chart",
         f"{df['Outlier'].sum():,} rows flagged ({df['Outlier'].mean():.1%})")

    # --- 9. Memory: text columns -> category ---------------------------------
    for col in ["Region", "County", "Facility", "Gender", "Race", "Ethnicity", "Admission Type", "Disposition",
                "Dx Code", "Dx Description", "Dx Category", "DRG Description", "MDC Description", "Case Type",
                "Primary Payer", "Payer Group", "Emergency Dept"]:
        df[col] = df[col].astype("category")
    df["Stay Days"] = df["Stay Days"].astype("int32")
    df = df.reset_index(drop=True)

    summary.update(clean_rows=len(df), clean_cols=df.shape[1], dropped_heavy=heavy, stay_cap=float(stay_cap))
    return df, pd.DataFrame(log), missing_table, summary


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================
def usd(x) -> str:
    """Short dollar format: 1,234,567 -> '$1.2M'."""
    if pd.isna(x):
        return "n/a"
    for limit, suffix in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(x) >= limit:
            return f"${x / limit:.1f}{suffix}"
    return f"${x:,.0f}"


def show_chart(fig, key):
    """Draw a Plotly figure full-width (works on old and new Streamlit versions)."""
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_table(data, **kwargs):
    """Draw a dataframe full-width (works on old and new Streamlit versions)."""
    try:
        st.dataframe(data, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, hide_index=True, **kwargs)


def show_text_table(data):
    """Table whose cells wrap onto several lines (st.dataframe cuts long text off)."""
    st.table(data.set_index(data.columns[0]))


# Readable tick labels for log-scale axes (Plotly's default log ticks are cluttered)
MONEY_TICKS = dict(tickvals=[1e3, 1e4, 1e5, 1e6, 1e7], ticktext=["$1K", "$10K", "$100K", "$1M", "$10M"])
STAY_TICKS = dict(tickvals=[1, 2, 5, 10, 20, 50, 100], ticktext=["1", "2", "5", "10", "20", "50", "100"])
MIN_CELL = 5  # heatmap cells built from fewer discharges than this are left blank


def mean_pivot(data, index, columns, min_n=MIN_CELL):
    """Average stay for every index x column pair; pairs with fewer than min_n discharges are blanked,
    because an average of one or two stays is not reliable."""
    mean = data.pivot_table(index=index, columns=columns, values="Stay Days", aggfunc="mean", observed=True)
    n = data.pivot_table(index=index, columns=columns, values="Stay Days", aggfunc="count", observed=True)
    return mean.where(n >= min_n).dropna(how="all").dropna(axis=1, how="all")


def tidy(fig, height=430, title=None):
    """Common layout for every chart."""
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      title=dict(text=title, font=dict(size=15)) if title else None)
    return fig


def counts(series: pd.Series) -> pd.Series:
    """value_counts that ignores unused categories left over after filtering."""
    vc = series.value_counts()
    return vc[vc > 0]


def geo_only(f: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose hospital / county is redacted (used by county and facility views)."""
    return f[(f["County"] != "Not Disclosed") & (f["Facility"] != "Not Disclosed")]


def shorten(text: str, n: int = 42) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


# =============================================================================
# 4. CHART FUNCTIONS  (assignment Step 4 - five required visuals)
# =============================================================================
def dx_stay_table(f, min_cases):
    """Average stay per diagnosis code (data behind chart 1)."""
    g = (f.groupby(["Dx Code", "Dx Description", "Dx Category"], observed=True)["Stay Days"]
         .agg(**{"Avg stay": "mean", "Median stay": "median", "Discharges": "size"}).reset_index())
    return g[(g["Discharges"] >= min_cases) & (g["Dx Code"] != "N/A")]


def chart_dx_bar(g, top_n):
    """CHART 1 - Bar chart: average stay per diagnosis code."""
    top = g.nlargest(top_n, "Avg stay").sort_values("Avg stay").copy()
    top["Diagnosis"] = top["Dx Code"].astype(str) + " · " + top["Dx Description"].astype(str).map(lambda t: shorten(t, 44))
    fig = px.bar(top, x="Avg stay", y="Diagnosis", orientation="h", color_discrete_sequence=[BLUE],
                 custom_data=["Dx Description", "Dx Category", "Median stay", "Discharges"])
    fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>Body system: %{customdata[1]}<br>"
                      "Average stay: %{x:.1f} days<br>Median stay: %{customdata[2]:.0f} days<br>"
                      "Discharges: %{customdata[3]:,}<extra></extra>")
    fig.update_layout(xaxis_title="Average stay (days)", yaxis_title=None)
    return tidy(fig, height=max(380, 24 * top_n + 110))


def chart_charges_box(f, log_scale, show_points):
    """CHART 2 - Box plot: total charges segmented by severity."""
    fig = px.box(f, x="Severity", y="Total Charges", color="Severity", log_y=log_scale,
                 category_orders={"Severity": SEVERITY_ORDER}, color_discrete_map=SEVERITY_COLORS,
                 points="outliers" if show_points else False)
    fig.update_traces(boxmean=True)  # dashed line = mean (sits far above the median -> skewed charges)
    fig.update_layout(showlegend=False, xaxis_title="Severity of illness (APR)", yaxis_title="Total charges (US$)")
    if log_scale:
        fig.update_yaxes(**MONEY_TICKS)
    else:
        fig.update_yaxes(tickprefix="$")
    return tidy(fig, height=470)


def facility_county_matrix(f, n_facilities):
    """Pivot for chart 3 (view A): average stay for the busiest facilities by county."""
    g = geo_only(f)
    top = counts(g["Facility"]).head(n_facilities).index
    sub = g[g["Facility"].isin(top)]
    piv = mean_pivot(sub, "Facility", "County")
    piv.index, piv.columns = piv.index.astype(str), piv.columns.astype(str)
    # order: counties by volume, facilities grouped under their county
    county_order = [c for c in counts(sub["County"].astype(str)).index if c in piv.columns]
    fac_county = sub.groupby("Facility", observed=True)["County"].agg(lambda s: s.value_counts().index[0])
    fac_n = counts(sub["Facility"])
    rows = sorted(piv.index, key=lambda x: (county_order.index(str(fac_county[x])), -fac_n[x]))
    return piv.loc[rows, county_order]


def county_severity_matrix(f, n_counties):
    """Pivot for chart 3 (view B): average stay for each county and severity level."""
    g = geo_only(f)
    top = counts(g["County"]).head(n_counties).index
    sub = g[g["County"].isin(top)]
    piv = mean_pivot(sub, "County", "Severity")
    piv.index, piv.columns = piv.index.astype(str), piv.columns.astype(str)
    return piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]


def chart_heatmap(piv, label_cols):
    """CHART 3 - Heatmap: average stay duration by facility and county."""
    piv = piv.copy()
    piv.index = [shorten(i, 44) for i in piv.index]
    fig = px.imshow(piv, aspect="auto", color_continuous_scale=HEAT_SCALE, text_auto=".1f",
                    labels=dict(color="Avg stay (days)", x="", y=""))
    fig.update_xaxes(side="top", tickangle=-35 if label_cols else 0)
    fig.update_traces(hovertemplate="%{y}<br>%{x}<br>Average stay: %{z:.1f} days<extra></extra>")
    return tidy(fig, height=max(380, 26 * len(piv) + 140))


def payer_pie_table(f, min_share=0.03):
    """Counts per payer; small payers are folded into 'Other payers' so the pie stays readable."""
    vc = counts(f["Primary Payer"])
    share = vc / vc.sum()
    names = [p if share[p] >= min_share else "Other payers" for p in vc.index]
    out = pd.DataFrame({"Payer": names, "Discharges": vc.values}).groupby("Payer", as_index=False).sum()
    return out.sort_values("Discharges", ascending=False)


def chart_payer_pie(table):
    """CHART 4 - Pie chart: distribution of patients by payment type."""
    fig = px.pie(table, names="Payer", values="Discharges", color="Payer", color_discrete_map=PAYER_COLORS)
    fig.update_traces(textinfo="percent", textposition="inside", sort=False, insidetextorientation="horizontal",
                      marker=dict(line=dict(color="rgba(0,0,0,0)", width=2)),
                      hovertemplate="<b>%{label}</b><br>%{value:,} discharges (%{percent})<extra></extra>")
    fig.update_layout(legend=dict(orientation="h", y=-0.12))
    return tidy(fig, height=400)


def payer_charges_table(f, min_cases=30):
    g = (f.groupby("Primary Payer", observed=True)["Total Charges"]
         .agg(**{"Avg charges": "mean", "Median charges": "median", "Discharges": "size"}).reset_index())
    return g[g["Discharges"] >= min_cases].sort_values("Avg charges")


def chart_payer_bar(g):
    """Companion to chart 4: what each payer's patients are billed on average."""
    fig = px.bar(g, x="Avg charges", y="Primary Payer", orientation="h", color="Primary Payer",
                 color_discrete_map=PAYER_COLORS, custom_data=["Median charges", "Discharges"])
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Average charges: $%{x:,.0f}<br>Median charges: $%{customdata[0]:,.0f}"
                      "<br>Discharges: %{customdata[1]:,}<extra></extra>")
    fig.update_layout(showlegend=False, xaxis_title="Average total charges (US$)", yaxis_title=None, xaxis_tickprefix="$")
    return tidy(fig, height=400)


def chart_stay_hist(f, bin_width, log_y):
    """CHART 5 - Histogram: overall distribution of length of stay."""
    fig = px.histogram(f, x="Stay Days", color_discrete_sequence=[BLUE], log_y=log_y)
    fig.update_traces(xbins=dict(start=0.5, size=bin_width),
                      hovertemplate="Stay: %{x} days<br>Discharges: %{y:,}<extra></extra>")
    mean, median = f["Stay Days"].mean(), f["Stay Days"].median()
    fig.add_vline(x=mean, line_dash="dash", line_color=ORANGE, annotation_text=f"Mean {mean:.1f} d",
                  annotation_position="top right")
    fig.add_vline(x=median, line_dash="dot", line_color=AQUA, annotation_text=f"Median {median:.0f} d",
                  annotation_position="top left")
    fig.update_layout(xaxis_title="Length of stay (days)", yaxis_title="Number of discharges", bargap=0.05)
    return tidy(fig, height=420)


# =============================================================================
# 5. EDA AND INSIGHT FUNCTIONS  (assignment Step 3)
# =============================================================================
def research_answers(f):
    """Answer the five research questions in plain English, using the filtered data."""
    out = []

    # Q1 - diagnoses linked to long stays
    g = dx_stay_table(f, MIN_DX_CASES).nlargest(3, "Avg stay")
    if len(g):
        items = "; ".join(f"**{r['Dx Description']}** ({r['Dx Code']}) - {r['Avg stay']:.1f} days" for _, r in g.iterrows())
        out.append(("Q1 · Which diagnosis codes are linked to long stays?",
                    f"The longest average stays (diagnoses with at least {MIN_DX_CASES} cases) are: {items}. "
                    f"The overall average is {f['Stay Days'].mean():.1f} days."))

    # Q2 - age groups and charges
    g = f.groupby("Age Group", observed=True)["Total Charges"].agg(total="sum", avg="mean")
    g = g[g["total"] > 0]
    if len(g):
        top_total, top_avg = g["total"].idxmax(), g["avg"].idxmax()
        share = f"{usd(g.loc[top_total, 'total'])}, {g.loc[top_total, 'total'] / g['total'].sum():.0%} of the bill"
        if top_total == top_avg:
            text = (f"**{top_total}** lead on both measures: the most charges in total ({share}) and the highest "
                    f"average bill per discharge ({usd(g.loc[top_avg, 'avg'])}).")
        else:
            text = (f"**{top_total}** generate the most charges in total ({share}), while **{top_avg}** have the "
                    f"highest average bill per discharge ({usd(g.loc[top_avg, 'avg'])}).")
        out.append(("Q2 · Which age groups incur the highest charges?", text))

    # Q3 - severity and cost
    g = f.groupby("Severity", observed=True)["Total Charges"].median().dropna()
    if len(g) >= 2:
        lo, hi = g.index[0], g.index[-1]
        corr = f["Stay Days"].corr(f["Total Charges"], method="spearman")
        out.append(("Q3 · How does patient severity affect total cost?",
                    f"Median charges rise from {usd(g.iloc[0])} for **{lo}** cases to {usd(g.iloc[-1])} for **{hi}** "
                    f"cases - about **{g.iloc[-1] / g.iloc[0]:.1f}x** higher. Length of stay and charges move together "
                    f"(Spearman rank correlation ρ = {corr:.2f})."))

    # Q4 - counties with longer stays
    geo = geo_only(f)
    g = (geo.groupby("County", observed=True)["Stay Days"].agg(avg="mean", n="size").query("n >= 50")
         .nlargest(3, "avg"))
    if len(g):
        items = "; ".join(f"**{c}** ({r['avg']:.1f} days)" for c, r in g.iterrows())
        out.append(("Q4 · Are there counties with longer average stays?",
                    f"Yes - among counties with at least 50 discharges the longest averages are {items}, "
                    f"compared with {geo['Stay Days'].mean():.1f} days across all counties shown."))

    # Q5 - payment type and billing
    g = payer_charges_table(f)
    if len(g) >= 2:
        hi, lo = g.iloc[-1], g.iloc[0]
        out.append(("Q5 · How do different payment types affect the final billing?",
                    f"Average charges range from {usd(lo['Avg charges'])} (**{lo['Primary Payer']}**) to "
                    f"{usd(hi['Avg charges'])} (**{hi['Primary Payer']}**). Payers with sicker or longer-stay "
                    f"patients are billed more, so payer mix is a cost driver as well as a revenue factor."))
    return out


def describe_numbers(f):
    """Statistical summary table: centre, spread, shape and a 95 % confidence interval of the mean."""
    rows = []
    for col in ["Stay Days", "Total Charges", "Total Costs", "Charge per Day"]:
        s = f[col].astype(float)
        n, sd = len(s), s.std()
        rows.append({
            "Measure": col, "Mean": s.mean(), "95% CI of mean (±)": 1.96 * sd / np.sqrt(n) if n > 1 else np.nan,
            "Median": s.median(), "Std. dev.": sd, "IQR": s.quantile(0.75) - s.quantile(0.25),
            "Skewness": s.skew(), "Max": s.max(),
        })
    return pd.DataFrame(rows).round(2)


def apply_filters(df, sel):
    """Return the rows that match the sidebar choices (an empty choice means 'all')."""
    mask = np.ones(len(df), dtype=bool)
    for col in ["Region", "County", "Age Group", "Gender", "Severity", "Primary Payer", "Admission Type"]:
        if sel[col]:
            mask &= df[col].isin(sel[col]).to_numpy()
    mask &= df["Stay Days"].between(*sel["stay"]).to_numpy()
    if sel["no_outliers"]:
        mask &= ~df["Outlier"].to_numpy()
    return df[mask]


# =============================================================================
# 6. PAGE LAYOUT
# =============================================================================
if not DATA_FILE.exists():
    st.error(f"Data file not found: {DATA_FILE}. Put hospital_sample.csv.gz inside the 'data' folder.")
    st.stop()

df, cleaning_log, missing_table, summary = load_and_clean(str(DATA_FILE))

# ---------------------------------------------------------------- sidebar ---
st.sidebar.title("🏥 MediScope")
st.sidebar.caption("Leave a filter empty to include everything.")

regions = st.sidebar.multiselect("Region", sorted(df["Region"].cat.categories), placeholder="All regions")
county_pool = df if not regions else df[df["Region"].isin(regions)]
counties = st.sidebar.multiselect("County", sorted(counts(county_pool["County"]).index), placeholder="All counties")
ages = st.sidebar.multiselect("Age group", AGE_ORDER, placeholder="All ages")
genders = st.sidebar.multiselect("Gender", sorted(df["Gender"].cat.categories), placeholder="All")
severities = st.sidebar.multiselect("Severity of illness", SEVERITY_ORDER, placeholder="All severities")
payers = st.sidebar.multiselect("Primary payer", sorted(df["Primary Payer"].cat.categories), placeholder="All payers")
admissions = st.sidebar.multiselect("Admission type", sorted(df["Admission Type"].cat.categories), placeholder="All types")
stay_max = int(df["Stay Days"].max())
stay_range = st.sidebar.slider("Length of stay (days)", 1, stay_max, (1, stay_max))
no_outliers = st.sidebar.checkbox("Exclude statistical outliers", value=True,
                                  help="Removes the longest 1% of stays and extreme charges (see the Data Cleaning tab). "
                                       "Untick to see every record.")

filters = {"Region": regions, "County": counties, "Age Group": ages, "Gender": genders, "Severity": severities,
           "Primary Payer": payers, "Admission Type": admissions, "stay": stay_range, "no_outliers": no_outliers}
f = apply_filters(df, filters)

st.sidebar.divider()
st.sidebar.metric("Discharges selected", f"{len(f):,}", f"{len(f) / len(df):.0%} of the sample", delta_color="off")
st.sidebar.caption(f"Data: NY SPARCS 2021 inpatient discharges - a random 5% sample ({len(df):,} cleaned rows) of the "
                   f"{FULL_FILE_ROWS:,}-row original.")

# ----------------------------------------------------------------- header ---
st.title("🏥 Hospital Inpatient Discharges Dashboard")
st.markdown("**For hospital administrators, clinicians and operations managers** - see which diagnoses keep patients "
            "longest, what drives the bill, and where length of stay differs by county, facility and payer.")

if len(f) < 30:
    st.warning("Fewer than 30 discharges match these filters - widen the filters to see reliable charts.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Discharges", f"{len(f):,}")
k2.metric("Average stay", f"{f['Stay Days'].mean():.1f} days", help="Mean length of stay. The median is lower because a few very long stays pull the mean up.")
k3.metric("Median stay", f"{f['Stay Days'].median():.0f} days")
k4.metric("Average charges", f"${f['Total Charges'].mean():,.0f}")
k5.metric("Median charges", f"${f['Total Charges'].median():,.0f}")

tab_dash, tab_eda, tab_clean, tab_problem = st.tabs(
    ["📊 Dashboard", "🔍 EDA & Insights", "🧹 Data Cleaning", "📖 Problem & Research"])

# ============================================================== DASHBOARD ===
with tab_dash:
    left, right = st.columns(2)

    # ---- Chart 1: bar chart -------------------------------------------------
    with left, st.container(border=True):
        st.subheader("1 · Which diagnoses keep patients longest?")
        st.caption("Average length of stay per diagnosis code (CCSR). Hover for the median and number of cases.")
        c1, c2 = st.columns(2)
        top_n = c1.slider("Diagnoses shown", 5, 30, 12, key="top_n")
        min_cases = c2.slider("Minimum cases per diagnosis", 5, 200, MIN_DX_CASES, step=5, key="min_cases",
                              help="Hides rare diagnoses whose average would be unreliable.")
        dx_table = dx_stay_table(f, min_cases)
        if dx_table.empty:
            st.info("No diagnosis has that many cases under the current filters. Lower the minimum.")
        else:
            show_chart(chart_dx_bar(dx_table, top_n), "chart_dx")
            with st.expander("View chart data"):
                show_table(dx_table.nlargest(top_n, "Avg stay").round(2))

    # ---- Chart 2: box plot --------------------------------------------------
    with right, st.container(border=True):
        st.subheader("2 · How does severity change the bill?")
        st.caption("Total charges for each severity level. Box = middle 50% of bills, solid line = median, "
                   "dashed line = mean.")
        c1, c2 = st.columns(2)
        log_scale = c1.checkbox("Log scale (recommended)", value=True, key="box_log",
                                help="Charges are heavily skewed; a log axis keeps every box readable.")
        show_pts = c2.checkbox("Show outlier points", value=False, key="box_pts")
        show_chart(chart_charges_box(f, log_scale, show_pts), "chart_box")
        with st.expander("View chart data"):
            sev = (f.groupby("Severity", observed=True)["Total Charges"]
                   .describe(percentiles=[0.25, 0.5, 0.75])[["count", "mean", "25%", "50%", "75%", "max"]].round(0))
            show_table(sev.reset_index())

    # ---- Chart 3: heatmap ---------------------------------------------------
    with st.container(border=True):
        st.subheader("3 · Where are stays longest? (facility and county)")
        st.caption("Average length of stay in days. Darker = longer stays. Each hospital belongs to one county, so "
                   "the facility view fills one block per county; the county × severity view is a full grid. "
                   f"Cells built from fewer than {MIN_CELL} discharges are left blank.")
        c1, c2 = st.columns([2, 1])
        view = c1.radio("View", ["Facility × County", "County × Severity"], horizontal=True, key="heat_view")
        if view == "Facility × County":
            n_items = c2.slider("Busiest facilities shown", 10, 40, 20, key="heat_n")
            matrix = facility_county_matrix(f, n_items)
            label_cols = True
        else:
            n_items = c2.slider("Busiest counties shown", 8, 40, 20, key="heat_n2")
            matrix = county_severity_matrix(f, n_items)
            label_cols = False
        if matrix.empty:
            st.info("No facility / county data under the current filters.")
        else:
            show_chart(chart_heatmap(matrix, label_cols), "chart_heat")
            with st.expander("View chart data"):
                show_table(matrix.round(2).reset_index())

    # ---- Chart 4: pie chart -------------------------------------------------
    with st.container(border=True):
        st.subheader("4 · Who pays? Distribution of patients by payment type")
        st.caption("Left: share of discharges by primary payer (payers under 3% are grouped as 'Other payers'). "
                   "Right: what each payer's patients are billed on average.")
        pie_table = payer_pie_table(f)
        c1, c2 = st.columns(2)
        with c1:
            show_chart(chart_payer_pie(pie_table), "chart_pie")
        with c2:
            pay_table = payer_charges_table(f)
            if pay_table.empty:
                st.info("Not enough cases per payer under the current filters.")
            else:
                show_chart(chart_payer_bar(pay_table), "chart_paybar")
        with st.expander("View chart data"):
            show_table(pie_table.assign(Share=lambda d: (d["Discharges"] / d["Discharges"].sum() * 100).round(1)))

    # ---- Chart 5: histogram -------------------------------------------------
    with st.container(border=True):
        st.subheader("5 · How long do patients stay overall?")
        st.caption("Number of discharges at each length of stay. The right-hand tail is why the mean (orange) "
                   "sits above the median (green).")
        c1, c2 = st.columns(2)
        bin_width = c1.slider("Bin width (days)", 1, 10, 1, key="bin_w")
        log_y = c2.checkbox("Log scale on the count axis", value=False, key="hist_log")
        show_chart(chart_stay_hist(f, bin_width, log_y), "chart_hist")
        with st.expander("View chart data"):
            show_table(f["Stay Band"].value_counts().rename_axis("Stay band").reset_index(name="Discharges")
                       .sort_values("Stay band"))

# ================================================================== EDA ======
with tab_eda:
    st.header("Exploratory data analysis")
    st.caption("Everything on this tab recalculates when you change the sidebar filters.")

    st.subheader("Answers to the five research questions")
    with st.container(border=True):
        for question, answer in research_answers(f):
            st.markdown(f"**{question}**")
            st.markdown(answer)

    st.subheader("Statistical summary")
    stats = describe_numbers(f)
    show_table(stats.style.format({c: "{:,.2f}" for c in stats.columns if c != "Measure"}))
    st.caption("Skewness well above 0 means a long right tail (a few very expensive or very long stays), which is why "
               "medians describe a 'typical' patient better than means. The 95% CI is mean ± 1.96 × SD / √n.")

    st.subheader("Diagnosis category × severity: average stay (groupby + pivot_table)")
    st.caption(f"Cells built from fewer than {MIN_CELL} discharges are left blank.")
    piv = mean_pivot(f, "Dx Category", "Severity")
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    piv.columns = piv.columns.astype(str)
    fig = px.imshow(piv, aspect="auto", color_continuous_scale=HEAT_SCALE, text_auto=".1f",
                    labels=dict(color="Avg stay (days)", x="", y=""))
    fig.update_traces(hovertemplate="%{y}<br>Severity: %{x}<br>Average stay: %{z:.1f} days<extra></extra>")
    show_chart(tidy(fig, height=max(380, 26 * len(piv) + 100)), "eda_dx_sev")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Charges by age group and gender")
        metric = st.radio("Measure", ["Average charges per discharge", "Total charges (all discharges)"],
                          horizontal=True, key="age_metric")
        agg = "mean" if metric.startswith("Average") else "sum"
        ag = (f[f["Gender"].isin(["Female", "Male"])].groupby(["Age Group", "Gender"], observed=True)["Total Charges"]
              .agg(agg).reset_index())
        fig = px.bar(ag, x="Age Group", y="Total Charges", color="Gender", barmode="group",
                     color_discrete_map={"Female": BLUE, "Male": ORANGE}, category_orders={"Age Group": AGE_ORDER})
        fig.update_layout(yaxis_title=metric + " (US$)", xaxis_title=None, yaxis_tickprefix="$",
                          legend=dict(orientation="h", y=1.1, title=None))
        fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}: $%{y:,.0f}<extra></extra>")
        show_chart(tidy(fig, height=400), "eda_age_gender")
    with col_b:
        st.subheader("Stay vs charges (sampled points)")
        pts = f.sample(min(4000, len(f)), random_state=1)
        fig = px.scatter(pts, x="Stay Days", y="Total Charges", color="Severity", log_x=True, log_y=True, opacity=0.45,
                         category_orders={"Severity": SEVERITY_ORDER}, color_discrete_map=SEVERITY_COLORS)
        fig.update_layout(xaxis_title="Length of stay (days, log scale)", yaxis_title="Total charges (US$, log scale)",
                          legend=dict(orientation="h", y=1.1, title=None))
        fig.update_xaxes(**STAY_TICKS)
        fig.update_yaxes(**MONEY_TICKS)
        show_chart(tidy(fig, height=400), "eda_scatter")

    st.subheader("High-utilisation facilities")
    fac = (geo_only(f).groupby(["Facility", "County"], observed=True)
           .agg(Discharges=("Stay Days", "size"), **{"Avg stay (days)": ("Stay Days", "mean")},
                **{"Avg charges ($)": ("Total Charges", "mean")}, **{"Total charges ($)": ("Total Charges", "sum")})
           .reset_index().query("Discharges > 0").nlargest(10, "Discharges"))
    fac["Facility"], fac["County"] = fac["Facility"].astype(str), fac["County"].astype(str)
    show_table(fac.round(1))

    st.subheader("Region × payer: average charges (pivot_table)")
    rp = (geo_only(f).pivot_table(index="Region", columns="Primary Payer", values="Total Charges", aggfunc="mean",
                                  observed=True).round(0))
    rp.index, rp.columns = rp.index.astype(str), rp.columns.astype(str)
    show_table(rp.reset_index())

# ============================================================ DATA CLEANING ==
with tab_clean:
    st.header("Data preprocessing and cleaning")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows loaded", f"{summary['raw_rows']:,}")
    m2.metric("Rows after cleaning", f"{summary['clean_rows']:,}")
    m3.metric("Columns loaded", summary["raw_cols"])
    m4.metric("Columns after cleaning", summary["clean_cols"])
    st.caption(f"The repository sample is a random 5% of the original {FULL_FILE_ROWS:,}-row file "
               "(created with scripts/make_sample.py).")

    st.subheader("What was done, step by step")
    show_text_table(cleaning_log)

    st.subheader("Missing values in the raw data")
    fig = px.bar(missing_table.sort_values("Missing %"), x="Missing %", y="Column", orientation="h",
                 color_discrete_sequence=[BLUE])
    fig.add_vline(x=NULL_LIMIT * 100, line_dash="dash", line_color=ORANGE,
                  annotation_text="50% drop limit", annotation_position="bottom right")
    fig.update_traces(hovertemplate="%{y}: %{x:.1f}% missing<extra></extra>")
    fig.update_layout(yaxis_title=None, xaxis_title="Missing values (%)")
    show_chart(tidy(fig, height=max(300, 26 * len(missing_table) + 80)), "clean_missing")

    st.subheader("Engineered columns")
    show_text_table(pd.DataFrame([
        {"Column": "Dx Category", "Meaning": "Body-system group from the first 3 letters of the CCSR diagnosis code"},
        {"Column": "Stay Band", "Meaning": "1-2, 3-7, 8-14 or 15+ days"},
        {"Column": "Charge per Day", "Meaning": "Total charges ÷ length of stay"},
        {"Column": "Cost-to-Charge Ratio", "Meaning": "Estimated cost ÷ billed charges (how much of the bill is real cost)"},
        {"Column": "Payer Group", "Meaning": "Government, Commercial, Self-pay or Other"},
        {"Column": "Outlier", "Meaning": "True if charges or stay are extreme (see the outlier step above)"},
    ]))

    st.subheader("Preview of the cleaned data")
    show_table(df.drop(columns=["Charge Outlier", "Stay Outlier"]).head(25))

# ======================================================== PROBLEM & RESEARCH =
with tab_problem:
    st.header("Understanding the problem")
    st.markdown(
        "**Client:** MediScope Health Analytics, working with hospitals to improve care and run operations more "
        "efficiently.  \n**Data:** 2021 inpatient discharges reported to New York State's SPARCS system - one row per "
        "hospital stay, with patient demographics, diagnosis, severity, payer, length of stay and billed charges.")

    st.subheader("Who uses this dashboard, and for what?")
    show_text_table(pd.DataFrame([
        {"User": "Hospital administrators", "Decision": "Bed capacity, staffing and budgets",
         "What they look at": "Length of stay by diagnosis, county and facility; total charges by age group"},
        {"User": "Clinicians", "Decision": "Which cases need early discharge planning",
         "What they look at": "Stay and cost by severity of illness and diagnosis"},
        {"User": "Operations / finance managers", "Decision": "Payer mix and billing",
         "What they look at": "Distribution and average charges by payment type"},
    ]))

    st.subheader("Five research questions")
    show_text_table(pd.DataFrame([
        {"#": 1, "Question": "Which diagnosis codes are linked to long stays?",
         "Why it matters": "Long stays use the most bed-days; these diagnoses are where discharge planning pays off most.",
         "Answered by": "Chart 1 (bar), EDA pivot"},
        {"#": 2, "Question": "Which age groups incur the highest charges?",
         "Why it matters": "An ageing population shifts demand; budgets and wards should follow.",
         "Answered by": "EDA age × gender chart"},
        {"#": 3, "Question": "How does patient severity affect total cost?",
         "Why it matters": "Severity-adjusted cost is the fair way to compare hospitals and plan resources.",
         "Answered by": "Chart 2 (box plot), scatter"},
        {"#": 4, "Question": "Are there counties with longer average stays?",
         "Why it matters": "Regional gaps can point to capacity, access or after-care problems.",
         "Answered by": "Chart 3 (heatmap), EDA facility table"},
        {"#": 5, "Question": "How do payment types affect the final billing?",
         "Why it matters": "Payer mix shapes revenue; billed charges differ widely between payers.",
         "Answered by": "Chart 4 (pie + bar), region × payer pivot"},
    ]))

    st.subheader("Background research")
    st.markdown(
        "- **Length of stay** is a standard efficiency measure in hospital operations: every extra day occupies a bed "
        "and staff, so administrators watch which diagnoses and facilities run long.\n"
        "- **Severity of illness** in this data comes from the APR-DRG system, which grades each stay as Minor, "
        "Moderate, Major or Extreme - so costs can be compared fairly between sicker and healthier patient mixes.\n"
        "- **Diagnoses** are coded with AHRQ's Clinical Classifications Software Refined (CCSR); the first three "
        "letters of each code give the body system (CIR = circulatory, RSP = respiratory ...), which this app uses "
        "to group 391 codes into readable categories.\n"
        "- **Charges are not costs.** Charges are the amount a hospital bills; most payers pay negotiated or fixed "
        "rates, so billed charges are far higher than the estimated cost of care. The app shows both.\n"
        "- **Skewed money data:** hospital bills have a long right tail, so this dashboard reports medians next to "
        "means and offers log-scale axes.")
    st.markdown("**Sources:** [NY SPARCS](https://www.health.ny.gov/statistics/sparcs/) (data), "
                "[AHRQ HCUP CCSR](https://hcup-us.ahrq.gov/toolssoftware/ccsr/ccs_refined.jsp) (diagnosis categories), "
                "[Streamlit blog: improving healthcare management](https://blog.streamlit.io/improving-healthcare-management-with-streamlit/) "
                "(dashboard design reference from the brief).")

    st.subheader("Limits of this analysis")
    st.markdown(
        "- The data has **no patient ID or readmission flag**, so readmission rates cannot be calculated; length of "
        "stay, charges and payer mix are used instead.\n"
        "- Figures come from a **5% random sample**, so small groups (a single county or rare diagnosis) can be noisy - "
        "use the minimum-cases slider.\n"
        "- Charges are billed amounts, not what was actually paid.")
