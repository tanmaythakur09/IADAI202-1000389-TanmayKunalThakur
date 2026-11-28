"""
MediScope Health Analytics - Hospital Inpatient Discharges Dashboard
=====================================================================
Course : Mathematics for AI-II (CRS: Artificial Intelligence)
Task   : SA-1, Scenario 2 - Hospital Inpatient Discharges Dashboard

Data   : New York State SPARCS de-identified inpatient discharges, 2021.
         The full file has 2,101,588 rows (about 794 MB), which is too big for
         GitHub and Streamlit Cloud, so this app ships with a reproducible
         5 % random sample (see scripts/make_sample.py).

The file is organised into clear modular sections:
    1. Settings and lookup tables
    2. Authentication & dark-tech UI styling
    3. Data loading and cleaning          (Step 2)
    4. Helper functions
    5. Chart functions                    (Step 4 - the 5 required visuals)
    6. Insight / EDA functions            (Step 3)
    7. Page layout: login gate, sidebar filters, KPI strip & four tabs
"""
import base64
import hmac
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =============================================================================
# 1. SETTINGS AND LOOKUP TABLES
# =============================================================================
# st.set_page_config must remain the very first Streamlit command.
_is_authed = bool(st.session_state.get("authenticated", False))
st.set_page_config(
    page_title="MediScope | Hospital Discharges",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded" if _is_authed else "collapsed",
)

DATA_FILE = Path(__file__).parent / "data" / "hospital_sample.csv.gz"
DASHBOARD_BG_FILE = Path(__file__).parent / "assets" / "dashboard_bg.jpg"
LOGIN_BG_FILE = Path(__file__).parent / "assets" / "login_bg.jpg"
LOGIN_BG_LIGHT_FILE = Path(__file__).parent / "assets" / "login_bg_light.jpg"
LOGIN_CORNER_LOGO_FILE = Path(__file__).parent / "assets" / "login_corner_logo.png"
LOGIN_CORNER_LOGO_LIGHT_FILE = Path(__file__).parent / "assets" / "login_corner_logo_light.png"
FULL_FILE_ROWS = 2_101_588   # rows in the original 794 MB file
NULL_LIMIT = 0.50            # drop a column if more than 50 % of it is empty
MIN_DX_CASES = 30            # minimum discharges before a diagnosis is ranked

# Ordered categories (so charts and tables sort Minor -> Extreme, young -> old)
SEVERITY_ORDER = ["Minor", "Moderate", "Major", "Extreme"]
AGE_ORDER = ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"]
STAY_BANDS = ["1-2 days", "3-7 days", "8-14 days", "15+ days"]

# Luminous colour-blind-safe dark-tech palette (Okabe-Ito / Tol inspired).
# Avoids red/green clashes and keeps strong luminance contrast on dark canvas.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#38BDF8", "#FB923C", "#2DD4BF", "#FACC15", "#F472B6"
VIOLET, RED, GREEN, GREY = "#A78BFA", "#F87171", "#34D399", "#94A3B8"
SEVERITY_COLORS = {
    "Minor": "#93C5FD",
    "Moderate": "#38BDF8",
    "Major": "#A78BFA",
    "Extreme": "#FB923C",
}
HEAT_SCALE = ["#10243E", "#1D4ED8", "#0EA5E9", "#2DD4BF", "#FDE047"]
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

DEMO_USER = "demo"
DEMO_PASS = "demo123"

# Shared parameterized inline SVG icon builder for crisp HUD badges across the app.
def _hud_svg(paths: str, size: int = 15, viewbox: str = "0 0 24 24", color: str = "currentColor") -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="{viewbox}" fill="none" '
        f'stroke="{color}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" '
        f'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">{paths}</svg>'
    )


def _medical_cross_svg(size: int, grad_id: str) -> str:
    """Neon medical cross + ECG pulse mark used on login and telemetry headers."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 48 48" fill="none" '
        f'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        f'<defs><linearGradient id="{grad_id}" x1="6" y1="6" x2="42" y2="42" gradientUnits="userSpaceOnUse">'
        f'<stop offset="0%" stop-color="#A5F3FC"/><stop offset="50%" stop-color="#38BDF8"/>'
        f'<stop offset="100%" stop-color="#0EA5E9"/></linearGradient></defs>'
        f'<path d="M18 6H30V18H42V30H30V42H18V30H6V18H18V6Z" '
        f'fill="rgba(14,165,233,0.20)" stroke="url(#{grad_id})" stroke-width="2.3" stroke-linejoin="round"/>'
        f'<path d="M20.2 9.5H27.8V20.2H38.5V27.8H27.8V38.5H20.2V27.8H9.5V20.2H20.2V9.5Z" '
        f'stroke="rgba(224,242,254,0.58)" stroke-width="1.1" stroke-linejoin="round"/>'
        f'<path d="M10 24.5H16.2L19.4 17L23.5 31.2L27.1 20L29.6 24.5H38" '
        f'stroke="#E0F2FE" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )


MEDISCOPE_LOGO_SVG = """
<svg width="34" height="34" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <defs>
    <linearGradient id="msGrad" x1="2" y1="2" x2="34" y2="34" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="100%" stop-color="#818CF8"/>
    </linearGradient>
  </defs>
  <rect x="2" y="2" width="32" height="32" rx="9" fill="rgba(56,189,248,0.12)" stroke="url(#msGrad)" stroke-width="1.6"/>
  <path d="M7.5 18.5H12.2L15.1 11.5L19.2 24.5L22.5 16L24.7 18.5H28.5" stroke="#38BDF8" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""
MEDISCOPE_SIDEBAR_LOGO_SVG = _medical_cross_svg(38, "msCrossSidebar")
MEDISCOPE_HERO_PULSE_SVG = (
    '<svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M3.5 18.5H10.2L14 7.5L19.4 28.5L23.4 14.5L26.2 18.5H32.5" '
    'stroke="#38BDF8" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'
    '<circle cx="14" cy="7.5" r="2" fill="#A5F3FC"/><circle cx="19.4" cy="28.5" r="2" fill="#2DD4BF"/>'
    '</svg>'
)

KPI_ICON_PATHS = {
    "discharges": '<path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2"/><rect x="9" y="3" width="6" height="4" rx="1"/><path d="M9 14l2 2 4-4"/>',
    "average stay": '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18M12 14v4M10 16h4"/>',
    "median stay": '<path d="M2 17v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5"/><path d="M2 14h20M6 10V7a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v3"/>',
    "average charges": '<circle cx="12" cy="12" r="9"/><path d="M14.8 9a2.8 2.8 0 0 0-5.6 0c0 1.8 2.8 2.2 2.8 4a2.8 2.8 0 0 1-5.6 0M12 5.5v13"/>',
    "median charges": '<path d="M4 3h16v18l-3-2-2.5 2L12 19l-2.5 2L7 19l-3 2V3z"/><path d="M9 9h6M9 13h4"/>',
    "rows loaded": '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v6c0 1.66 3.58 3 8 3s8-1.34 8-3V6"/><path d="M4 12v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6"/>',
    "rows after cleaning": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "columns loaded": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18M15 3v18M3 9h18"/>',
    "columns after cleaning": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
}

SIDEBAR_GROUP_ICONS = {
    "Geography": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
    "Patient Profile": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/>',
    "Clinical & Payer": '<path d="M12 21s-7-4.5-9-9.5C1.5 7.5 4.5 4 8.2 4c2.1 0 3.4 1.2 3.8 2 .4-.8 1.7-2 3.8-2C19.5 4 22.5 7.5 21 11.5 19 16.5 12 21 12 21z"/>',
    "Stay & Outliers": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
}

PANEL_TAG_ICONS = {
    "bar": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "box": '<rect x="8" y="6" width="8" height="12" rx="1"/><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="8" y1="12" x2="16" y2="12"/>',
    "heatmap": '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/>',
    "pie": '<path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/>',
    "histogram": '<rect x="3" y="11" width="4" height="9" rx="1"/><rect x="10" y="5" width="4" height="15" rx="1"/><rect x="17" y="8" width="4" height="12" rx="1"/>',
}

USER_AVATAR_SVG = _hud_svg('<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>', size=14, color="#38BDF8")
CHIP_TERMINAL_SVG = _hud_svg('<rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>', size=12, color="#38BDF8")
CHIP_SHIELD_SVG = _hud_svg('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/>', size=12, color="#7DD3FC")

LOGIN_CORNER_LOGO_SVG = _medical_cross_svg(40, "msCrossCorner")
LOGIN_MEDALLION_SVG = _medical_cross_svg(48, "msCrossMedallion")
LOGIN_BADGE_WAVE_SVG = """
<svg width="16" height="16" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <circle cx="9" cy="9" r="7.5" fill="rgba(56,189,248,0.22)" stroke="#38BDF8" stroke-width="1.2"/>
  <path d="M4.2 9.2H6.8L8.1 6.1L10 11.9L11.4 8.2L12.3 9.2H13.8" stroke="#E0F2FE" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""
LOGIN_INFO_SVG = """
<svg width="13" height="13" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <circle cx="8" cy="8" r="6.8" fill="rgba(56,189,248,0.2)" stroke="#38BDF8" stroke-width="1.3"/>
  <path d="M8 7.2V11.2M8 4.8H8.01" stroke="#7DD3FC" stroke-width="1.6" stroke-linecap="round"/>
</svg>
"""
LOGIN_KEY_SVG = """
<svg width="12" height="12" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <circle cx="5.5" cy="7.5" r="2.8" stroke="#94A3B8" stroke-width="1.3"/>
  <path d="M8 7.5H14M12 7.5V9.8" stroke="#94A3B8" stroke-width="1.3" stroke-linecap="round"/>
</svg>
"""
LOGIN_SPARKLE_SVG = """
<svg width="13" height="13" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M8 1.5L9.6 6.4L14.5 8L9.6 9.6L8 14.5L6.4 9.6L1.5 8L6.4 6.4L8 1.5Z" fill="#38BDF8" fill-opacity="0.35" stroke="#7DD3FC" stroke-width="1.2" stroke-linejoin="round"/>
</svg>
"""


@st.cache_data(show_spinner=False)
def load_asset_b64(path: str, mtime_ns: int = 0) -> str:
    """Read and base64-encode a local static asset for inline HTML/CSS rendering."""
    _ = mtime_ns
    asset_path = Path(path)
    if not asset_path.is_file():
        return ""
    return base64.b64encode(asset_path.read_bytes()).decode("ascii")


# =============================================================================
# 2. AUTHENTICATION & DARK-TECH UI STYLING
# =============================================================================
def render_raw_html(markup: str) -> None:
    """Render HTML cleanly via st.markdown without 4-space code-block indentation issues."""
    compact = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(compact, unsafe_allow_html=True)


def get_auth_credentials() -> tuple[dict[str, str], bool]:
    """Load login credentials from st.secrets inside try/except with a safe demo fallback.

    Returns (credentials_dict, using_demo_fallback). No real passwords are stored in code.
    """
    creds: dict[str, str] = {}
    try:
        secrets = st.secrets
        if "auth" in secrets:
            auth_cfg = secrets["auth"]
            if "users" in auth_cfg:
                for user, pwd in dict(auth_cfg["users"]).items():
                    if user and pwd:
                        creds[str(user)] = str(pwd)
            elif "username" in auth_cfg and "password" in auth_cfg:
                creds[str(auth_cfg["username"])] = str(auth_cfg["password"])
        if not creds and "users" in secrets:
            for user, pwd in dict(secrets["users"]).items():
                if user and pwd:
                    creds[str(user)] = str(pwd)
        if not creds and "username" in secrets and "password" in secrets:
            creds[str(secrets["username"])] = str(secrets["password"])
    except Exception:
        creds = {}

    if not creds:
        return {DEMO_USER: DEMO_PASS}, True
    return creds, False


def verify_credentials(username: str, password: str) -> bool:
    """Validate username and password using constant-time comparison."""
    cleaned_user = username.strip()
    if not cleaned_user or not password:
        return False
    creds, _ = get_auth_credentials()
    expected = creds.get(cleaned_user)
    if expected is None:
        return False
    return hmac.compare_digest(password, expected)


def init_auth_state() -> None:
    """Ensure authentication keys exist in st.session_state."""
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("auth_user", "")
    st.session_state.setdefault("auth_mode", "")
    st.session_state.setdefault("remember_me", False)
    st.session_state.setdefault("remembered_user", "")
    st.session_state.setdefault("login_dark_mode", True)


def sign_in_user(username: str, mode: str) -> None:
    """Persist authenticated session state and trigger an immediate app rerun."""
    if mode != "Guest":
        remember = bool(st.session_state.get("login_remember", False))
        st.session_state["remember_me"] = remember
        st.session_state["remembered_user"] = username if remember else ""
    st.session_state["authenticated"] = True
    st.session_state["auth_user"] = username
    st.session_state["auth_mode"] = mode
    st.rerun()


def logout_user() -> None:
    """Clear login state and return the user to the full-screen login page."""
    st.session_state["authenticated"] = False
    st.session_state["auth_user"] = ""
    st.session_state["auth_mode"] = ""
    st.rerun()


def _fixed_bg_css(bg_file: Path, *, app_bg_color: str, overlay: str = "", bg_filter: str = "brightness(1.0)", opacity: str = "1") -> str:
    """Build fixed full-viewport background CSS for either login or dashboard views."""
    mtime = bg_file.stat().st_mtime_ns if bg_file.is_file() else 0
    b64 = load_asset_b64(str(bg_file), mtime)
    if not b64:
        return ""
    img_layer = f'url("data:image/jpeg;base64,{b64}")'
    layers = f"{overlay}, {img_layer}" if overlay else img_layer
    return f"""
        [data-testid="stApp"] {{
            background: transparent !important;
        }}
        [data-testid="stAppViewContainer"] {{
            position: relative !important;
            background-color: {app_bg_color} !important;
            background-image: none !important;
        }}
        [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            background-image: {layers};
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            filter: {bg_filter};
            opacity: {opacity};
        }}
        [data-testid="stMain"] {{
            position: relative;
            z-index: 1;
        }}
        [data-testid="stSidebar"] {{
            z-index: 2;
        }}
    """


def inject_theme_css(authenticated: bool) -> None:
    """Inject the dark-tech design system CSS; hides the sidebar until logged in."""
    login_only_css = ""
    dash_bg_css = ""
    if authenticated:
        dash_bg_css = _fixed_bg_css(
            DASHBOARD_BG_FILE,
            app_bg_color="#02091E",
            overlay="linear-gradient(180deg, rgba(2, 10, 26, 0.24) 0%, rgba(2, 10, 26, 0.14) 50%, rgba(2, 10, 26, 0.28) 100%)",
            bg_filter="brightness(1.06) saturate(1.06)",
            opacity="1",
        )
        hosp_mtime = LOGIN_BG_FILE.stat().st_mtime_ns if LOGIN_BG_FILE.is_file() else 0
        hosp_b64 = load_asset_b64(str(LOGIN_BG_FILE), hosp_mtime)
        if hosp_b64:
            dash_bg_css += f"""
        .ms-hero::after {{
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            z-index: 1;
            background-image: url("data:image/jpeg;base64,{hosp_b64}");
            background-size: 920px auto;
            background-position: right -360px top 62%;
            background-repeat: no-repeat;
            opacity: 0.56;
            -webkit-mask-image: linear-gradient(96deg, transparent 0%, transparent 46%, rgba(0, 0, 0, 0.35) 66%, rgba(0, 0, 0, 0.92) 100%);
            mask-image: linear-gradient(96deg, transparent 0%, transparent 46%, rgba(0, 0, 0, 0.35) 66%, rgba(0, 0, 0, 0.92) 100%);
        }}
            """
        dash_bg_css += """
        [data-testid="stAppViewContainer"] {
            height: 100vh !important;
            height: 100dvh !important;
            max-height: 100vh !important;
            max-height: 100dvh !important;
            overflow: hidden !important;
        }
        [data-testid="stMain"] {
            height: 100vh !important;
            height: 100dvh !important;
            max-height: 100vh !important;
            max-height: 100dvh !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }
        [data-testid="stSidebar"] {
            height: 100vh !important;
            height: 100dvh !important;
            max-height: 100vh !important;
            max-height: 100dvh !important;
            overflow: hidden !important;
        }
        [data-testid="stSidebarContent"] {
            height: 100% !important;
            max-height: 100vh !important;
            max-height: 100dvh !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding-bottom: 1.5rem !important;
        }

        /* Sidebar section headers & status pods matching HUD pill aesthetic */
        [data-testid="stSidebar"] {
            box-shadow: 6px 0 36px rgba(2, 132, 199, 0.14), inset -1px 0 0 rgba(56, 189, 248, 0.32) !important;
        }
        [data-testid="stSidebar"] .ms-sidebar-brand {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.24), rgba(37, 99, 235, 0.14)) !important;
            border: 1px solid rgba(56, 189, 248, 0.48) !important;
            border-radius: 14px !important;
            box-shadow:
                0 0 22px rgba(56, 189, 248, 0.16),
                inset 0 1px 0 rgba(224, 242, 254, 0.22),
                inset 0 -2px 0 rgba(56, 189, 248, 0.55) !important;
        }
        [data-testid="stSidebar"] .ms-user-pill {
            background: linear-gradient(180deg, rgba(8, 25, 62, 0.84), rgba(4, 14, 36, 0.90)) !important;
            border: 1px solid rgba(56, 189, 248, 0.34) !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(186, 230, 253, 0.14) !important;
        }
        .ms-avatar-ring {
            width: 24px;
            height: 24px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: rgba(56, 189, 248, 0.16);
            border: 1px solid rgba(56, 189, 248, 0.45);
            flex-shrink: 0;
        }
        [data-testid="stSidebar"] .ms-filter-group {
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
            margin: 1rem 0 0.45rem 0 !important;
            padding: 0.30rem 0.65rem !important;
            border-radius: 9px !important;
            background: linear-gradient(90deg, rgba(14, 165, 233, 0.18), rgba(37, 99, 235, 0.04) 78%, transparent) !important;
            border: none !important;
            border-left: 2.5px solid #38BDF8 !important;
            color: #BAE6FD !important;
            letter-spacing: 0.10em !important;
        }
        [data-testid="stSidebar"] .ms-filter-group svg {
            color: #38BDF8;
            filter: drop-shadow(0 0 6px rgba(56, 189, 248, 0.65));
            flex-shrink: 0;
        }
        [data-testid="stSidebar"] .react-aria-ComboBox > div:first-of-type:not(:has([data-tag])) {
            position: relative !important;
            padding-left: 34px !important;
        }
        [data-testid="stSidebar"] .react-aria-ComboBox > div:first-of-type:not(:has([data-tag]))::before {
            content: "";
            position: absolute;
            left: 11px;
            top: 50%;
            transform: translateY(-50%);
            width: 14px;
            height: 14px;
            pointer-events: none;
            background: #38BDF8;
            opacity: 0.82;
            -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.1' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 21s-6-5.33-6-10a6 6 0 1 1 12 0c0 4.67-6 10-6 10z'/%3E%3Ccircle cx='12' cy='11' r='2.3'/%3E%3C/svg%3E") center / contain no-repeat;
            mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.1' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 21s-6-5.33-6-10a6 6 0 1 1 12 0c0 4.67-6 10-6 10z'/%3E%3Ccircle cx='12' cy='11' r='2.3'/%3E%3C/svg%3E") center / contain no-repeat;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(56, 189, 248, 0.22) !important;
            margin: 0.85rem 0 !important;
        }
        [data-testid="stSidebar"] .ms-sidebar-telemetry {
            background: linear-gradient(180deg, rgba(8, 25, 62, 0.84), rgba(4, 14, 36, 0.92)) !important;
            border: 1px solid rgba(56, 189, 248, 0.38) !important;
            border-radius: 13px !important;
            box-shadow:
                0 6px 22px rgba(0, 0, 0, 0.32),
                0 0 18px rgba(56, 189, 248, 0.12),
                inset 0 1px 0 rgba(224, 242, 254, 0.16),
                inset 0 -2px 0 rgba(56, 189, 248, 0.65) !important;
        }

        /* Header Deploy action pill & menus */
        [data-testid="stHeader"] button {
            border-radius: 999px !important;
            transition: background 0.16s ease, border-color 0.16s ease, color 0.16s ease !important;
        }
        [data-testid="stAppDeployButton"] button {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.22), rgba(37, 99, 235, 0.16)) !important;
            border: 1px solid rgba(56, 189, 248, 0.45) !important;
            color: #E0F2FE !important;
            padding: 0.22rem 0.85rem !important;
            box-shadow: 0 0 14px rgba(56, 189, 248, 0.18) !important;
        }
        [data-testid="stAppDeployButton"] button:hover {
            border-color: #38BDF8 !important;
            box-shadow: 0 0 18px rgba(56, 189, 248, 0.34) !important;
        }
        [data-testid="stMainMenuPopover"] [role="menu"],
        [data-baseweb="popover"] > div,
        [role="tooltip"] {
            background: linear-gradient(180deg, rgba(6, 22, 56, 0.96), rgba(3, 12, 34, 0.98)) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border: 1px solid rgba(56, 189, 248, 0.42) !important;
            border-radius: 12px !important;
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.56), 0 0 22px rgba(56, 189, 248, 0.18) !important;
            color: #F1F5F9 !important;
        }

        /* Hero card, KPI cards & panel HUD polish across all tabs */
        .ms-hero > * {
            position: relative;
            z-index: 2;
        }
        .ms-kpi {
            border-radius: 16px !important;
            background: linear-gradient(172deg, rgba(8, 26, 62, 0.84) 0%, rgba(3, 12, 32, 0.92) 100%) !important;
            border: 1px solid rgba(56, 189, 248, 0.26) !important;
            box-shadow:
                0 12px 28px rgba(0, 0, 0, 0.34),
                inset 0 1px 0 rgba(224, 242, 254, 0.12) !important;
            min-height: 128px !important;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .ms-kpi::before {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(240px 100px at 85% 95%, color-mix(in srgb, var(--kpi-accent, #38BDF8) 18%, transparent), transparent 72%);
            pointer-events: none;
            z-index: 0;
        }
        .ms-kpi:hover {
            transform: translateY(-3px) !important;
            border-color: color-mix(in srgb, var(--kpi-accent, #38BDF8) 72%, rgba(255, 255, 255, 0.2)) !important;
            box-shadow:
                0 16px 36px rgba(0, 0, 0, 0.46),
                0 0 24px color-mix(in srgb, var(--kpi-accent, #38BDF8) 26%, transparent),
                inset 0 1px 0 rgba(224, 242, 254, 0.22) !important;
        }
        .ms-kpi > *:not(.ms-kpi-wave):not(.ms-kpi-bar) {
            position: relative;
            z-index: 2;
        }
        .ms-kpi-head {
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 0.35rem;
        }
        .ms-kpi-icon {
            width: 21px;
            height: 21px;
            border-radius: 6px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: color-mix(in srgb, var(--kpi-accent, #38BDF8) 18%, rgba(6, 20, 50, 0.85));
            border: 1px solid color-mix(in srgb, var(--kpi-accent, #38BDF8) 45%, transparent);
            color: var(--kpi-accent, #38BDF8);
            flex-shrink: 0;
        }
        .ms-kpi-head .ms-kpi-label {
            margin-bottom: 0 !important;
            font-size: 0.64rem !important;
            letter-spacing: 0.045em !important;
            white-space: nowrap;
        }
        .ms-kpi-sub {
            font-size: 0.74rem !important;
            line-height: 1.28 !important;
            margin-top: 0.28rem !important;
        }
        .ms-kpi-wave {
            position: absolute;
            right: 0;
            bottom: 0;
            width: 140px;
            height: 44px;
            pointer-events: none;
            opacity: 0.32;
            z-index: 1;
        }
        .ms-panel-tag {
            border-radius: 999px !important;
            padding: 3px 10px !important;
            background: rgba(14, 165, 233, 0.14) !important;
            border: 1px solid rgba(56, 189, 248, 0.36) !important;
            box-shadow: inset 0 1px 0 rgba(224, 242, 254, 0.12) !important;
        }
        div[class*="st-key-card_"] {
            position: relative !important;
            overflow: hidden !important;
            border-radius: 18px !important;
            border: 1px solid rgba(56, 189, 248, 0.28) !important;
            box-shadow:
                0 14px 36px rgba(0, 0, 0, 0.38),
                inset 0 1px 0 rgba(186, 230, 253, 0.12) !important;
        }
        div[class*="st-key-card_"]::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, rgba(56, 189, 248, 0.78), rgba(45, 212, 191, 0.42), transparent 88%);
            pointer-events: none;
        }
        div[class*="st-key-card_"]:hover {
            border-color: rgba(56, 189, 248, 0.48) !important;
            box-shadow:
                0 16px 40px rgba(0, 0, 0, 0.48),
                0 0 24px rgba(56, 189, 248, 0.12),
                inset 0 1px 0 rgba(224, 242, 254, 0.18) !important;
        }

        /* Headings, DataFrames, Tables, Alerts & Links across all 4 tabs */
        [data-testid="stTabs"] h2 {
            font-size: clamp(1.32rem, 2vw, 1.65rem) !important;
            font-weight: 700 !important;
            padding-left: 0.78rem !important;
            border-left: 3px solid #38BDF8 !important;
            background: linear-gradient(95deg, #F8FAFC 18%, #BAE6FD 92%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            margin-top: 0.25rem !important;
            margin-bottom: 0.35rem !important;
        }
        [data-testid="stTabs"] h3 {
            color: #F1F5F9 !important;
            font-weight: 600 !important;
            letter-spacing: -0.015em !important;
        }
        [data-testid="stTabs"] a {
            color: #38BDF8 !important;
            text-decoration-color: rgba(56, 189, 248, 0.45) !important;
        }
        [data-testid="stTabs"] a:hover {
            color: #7DD3FC !important;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(56, 189, 248, 0.30) !important;
            border-radius: 13px !important;
            overflow: hidden !important;
            background: rgba(4, 14, 36, 0.78) !important;
            box-shadow: 0 8px 26px rgba(0, 0, 0, 0.30), inset 0 1px 0 rgba(186, 230, 253, 0.08) !important;
        }
        [data-testid="stTable"] {
            border: 1px solid rgba(56, 189, 248, 0.28) !important;
            border-radius: 13px !important;
            overflow: hidden !important;
            background: rgba(4, 14, 36, 0.74) !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28) !important;
        }
        [data-testid="stTable"] th {
            background: linear-gradient(180deg, rgba(10, 29, 72, 0.95), rgba(7, 20, 50, 0.95)) !important;
            color: #7DD3FC !important;
            padding: 0.65rem 0.85rem !important;
        }
        [data-testid="stTable"] td {
            padding: 0.60rem 0.85rem !important;
            border-bottom: 1px solid rgba(56, 189, 248, 0.12) !important;
        }
        [data-testid="stTable"] tbody tr:nth-child(even) {
            background: rgba(14, 165, 233, 0.04) !important;
        }
        [data-testid="stTable"] tbody tr:hover {
            background: rgba(56, 189, 248, 0.11) !important;
        }
        [data-testid="stAlert"] {
            border-radius: 13px !important;
            border: 1px solid rgba(56, 189, 248, 0.36) !important;
            background: linear-gradient(135deg, rgba(8, 26, 62, 0.86), rgba(4, 14, 36, 0.92)) !important;
            backdrop-filter: blur(12px) !important;
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.28) !important;
        }
        [data-testid="stPlotlyChart"] .modebar-btn path {
            fill: #7DD3FC !important;
            opacity: 0.72;
        }
        [data-testid="stPlotlyChart"] .modebar-btn:hover path {
            fill: #E0F2FE !important;
            opacity: 1;
        }

        /* Sidebar & main multiselect pills + frosted dropdown popover */
        [data-testid="stMultiSelect"] [data-testid="stWidgetLabel"] p {
            color: #E2E8F0 !important;
            font-weight: 500 !important;
            transition: color 0.18s ease !important;
        }
        [data-testid="stMultiSelect"]:hover [data-testid="stWidgetLabel"] p,
        [data-testid="stMultiSelect"]:focus-within [data-testid="stWidgetLabel"] p {
            color: #7DD3FC !important;
        }
        [data-testid="stMultiSelect"] .react-aria-ComboBox > div:first-of-type,
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
            min-height: 42px !important;
            padding: 4px 8px !important;
            border-radius: 12px !important;
            background: linear-gradient(180deg, rgba(7, 23, 56, 0.80), rgba(4, 14, 36, 0.88)) !important;
            border: 1px solid rgba(56, 189, 248, 0.28) !important;
            box-shadow:
                0 4px 14px rgba(0, 0, 0, 0.26),
                inset 0 1px 0 rgba(186, 230, 253, 0.10) !important;
            transition: background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease !important;
            cursor: pointer !important;
        }
        [data-testid="stMultiSelect"] .react-aria-ComboBox > div:first-of-type:hover,
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div:hover {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.20), rgba(37, 99, 235, 0.15)) !important;
            border-color: rgba(56, 189, 248, 0.56) !important;
            box-shadow:
                0 0 18px rgba(56, 189, 248, 0.20),
                inset 0 1px 0 rgba(224, 242, 254, 0.18) !important;
        }
        [data-testid="stMultiSelect"] .react-aria-ComboBox:focus-within > div:first-of-type,
        [data-testid="stMultiSelect"]:has(input[aria-expanded="true"]) .react-aria-ComboBox > div:first-of-type,
        [data-testid="stMultiSelect"]:has([data-tag]) .react-aria-ComboBox > div:first-of-type,
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.30), rgba(37, 99, 235, 0.22)) !important;
            border-color: rgba(56, 189, 248, 0.65) !important;
            box-shadow:
                0 0 20px rgba(56, 189, 248, 0.25),
                inset 0 1px 0 rgba(224, 242, 254, 0.22),
                inset 0 -2px 0 rgba(56, 189, 248, 0.85) !important;
        }
        [data-testid="stMultiSelect"] span[data-tag],
        [data-baseweb="tag"] {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.36), rgba(37, 99, 235, 0.26)) !important;
            border: 1px solid rgba(56, 189, 248, 0.58) !important;
            border-radius: 999px !important;
            color: #F8FAFC !important;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.16) !important;
        }
        [data-testid="stMultiSelect"] span[data-tag] span {
            color: #F8FAFC !important;
        }
        [data-testid="stMultiSelect"] button[aria-label="Open"] svg,
        [data-testid="stMultiSelect"] button[aria-label="Clear all"] svg {
            color: #7DD3FC !important;
            transition: transform 0.18s ease, color 0.18s ease !important;
        }
        [data-testid="stMultiSelect"]:has(input[aria-expanded="true"]) button[aria-label="Open"] svg {
            color: #BAE6FD !important;
            transform: rotate(180deg) !important;
        }
        [data-testid="stMultiSelectDropdown"] {
            background: linear-gradient(180deg, rgba(6, 22, 56, 0.96), rgba(3, 12, 34, 0.98)) !important;
            backdrop-filter: blur(18px) !important;
            -webkit-backdrop-filter: blur(18px) !important;
            border: 1px solid rgba(56, 189, 248, 0.48) !important;
            border-radius: 13px !important;
            box-shadow:
                0 18px 44px rgba(0, 0, 0, 0.62),
                0 0 26px rgba(56, 189, 248, 0.20),
                inset 0 1px 0 rgba(224, 242, 254, 0.16) !important;
            padding: 5px !important;
        }
        [data-testid="stMultiSelectDropdown"] [role="listbox"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }
        [data-testid="stMultiSelectDropdown"] [role="option"] {
            border-radius: 9px !important;
            border: 1px solid transparent !important;
            color: #E2E8F0 !important;
            transition: background 0.14s ease, border-color 0.14s ease, color 0.14s ease !important;
        }
        [data-testid="stMultiSelectDropdown"] [role="option"]:hover,
        [data-testid="stMultiSelectDropdown"] [role="option"][data-focused="true"] {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.30), rgba(37, 99, 235, 0.22)) !important;
            border-color: rgba(56, 189, 248, 0.56) !important;
            color: #F8FAFC !important;
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.18) !important;
        }

        /* Sidebar slider card pod */
        [data-testid="stSidebar"] [data-testid="stSlider"] {
            background: linear-gradient(180deg, rgba(7, 22, 54, 0.74), rgba(4, 14, 34, 0.86)) !important;
            border: 1px solid rgba(56, 189, 248, 0.26) !important;
            border-radius: 13px !important;
            padding: 0.72rem 0.85rem 0.58rem !important;
            margin-bottom: 0.55rem !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.24), inset 0 1px 0 rgba(186, 230, 253, 0.10) !important;
            transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
        }
        [data-testid="stSidebar"] [data-testid="stSlider"]:hover,
        [data-testid="stSidebar"] [data-testid="stSlider"]:has([data-dragging="true"]) {
            border-color: rgba(56, 189, 248, 0.58) !important;
            box-shadow: 0 0 18px rgba(56, 189, 248, 0.20), inset 0 1px 0 rgba(224, 242, 254, 0.18) !important;
        }

        /* Tactile HUD pill checkboxes (sidebar & main charts) */
        [data-testid="stCheckbox"] label {
            width: 100% !important;
            min-height: 40px !important;
            padding: 0.45rem 0.78rem !important;
            border-radius: 12px !important;
            background: linear-gradient(180deg, rgba(7, 22, 54, 0.74), rgba(4, 14, 34, 0.86)) !important;
            border: 1px solid rgba(56, 189, 248, 0.24) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.22), inset 0 1px 0 rgba(186, 230, 253, 0.08) !important;
            transition: background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease !important;
            cursor: pointer !important;
        }
        [data-testid="stCheckbox"] label:hover {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.20), rgba(37, 99, 235, 0.15)) !important;
            border-color: rgba(56, 189, 248, 0.54) !important;
            box-shadow: 0 0 16px rgba(56, 189, 248, 0.18), inset 0 1px 0 rgba(224, 242, 254, 0.16) !important;
        }
        [data-testid="stCheckbox"] label[data-selected="true"],
        [data-testid="stCheckbox"] label:has(input:checked) {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.30), rgba(37, 99, 235, 0.22)) !important;
            border: 1px solid rgba(56, 189, 248, 0.62) !important;
            box-shadow:
                0 0 18px rgba(56, 189, 248, 0.22),
                inset 0 1px 0 rgba(224, 242, 254, 0.20),
                inset 0 -2px 0 rgba(56, 189, 248, 0.82) !important;
        }
        [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] {
            flex: 1 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            gap: 6px !important;
        }
        [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p {
            color: #E2E8F0 !important;
            font-weight: 500 !important;
            font-size: 0.86rem !important;
        }
        [data-testid="stCheckbox"] label[data-selected="true"] [data-testid="stWidgetLabel"] p,
        [data-testid="stCheckbox"] label:has(input:checked) [data-testid="stWidgetLabel"] p {
            color: #F8FAFC !important;
            font-weight: 600 !important;
        }
        [data-testid="stCheckbox"] label > div:first-of-type {
            border-radius: 6px !important;
            border-color: rgba(56, 189, 248, 0.45) !important;
            transition: box-shadow 0.18s ease, background-color 0.18s ease !important;
        }
        [data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type,
        [data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type {
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.65) !important;
        }

        /* Segmented HUD radio switcher matching tab dock */
        [data-testid="stRadioGroup"] {
            display: inline-flex !important;
            flex-wrap: wrap !important;
            gap: 6px !important;
            padding: 5px !important;
            border-radius: 13px !important;
            background: linear-gradient(180deg, rgba(6, 20, 50, 0.80), rgba(3, 12, 32, 0.88)) !important;
            border: 1px solid rgba(56, 189, 248, 0.28) !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(186, 230, 253, 0.10) !important;
        }
        [data-testid="stRadioOption"] {
            padding: 0.42rem 0.92rem !important;
            border-radius: 9px !important;
            border: 1px solid transparent !important;
            color: #94A3B8 !important;
            transition: background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease, color 0.18s ease !important;
            cursor: pointer !important;
        }
        [data-testid="stRadioOption"]:hover {
            background: rgba(56, 189, 248, 0.10) !important;
            border-color: rgba(56, 189, 248, 0.28) !important;
            color: #E0F2FE !important;
        }
        [data-testid="stRadioOption"][data-selected="true"],
        [data-testid="stRadioOption"]:has(input:checked) {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.34), rgba(37, 99, 235, 0.24)) !important;
            border: 1px solid rgba(56, 189, 248, 0.62) !important;
            color: #F8FAFC !important;
            box-shadow:
                0 0 18px rgba(56, 189, 248, 0.24),
                inset 0 1px 0 rgba(224, 242, 254, 0.22),
                inset 0 -2px 0 rgba(56, 189, 248, 0.85) !important;
        }
        [data-testid="stRadioOption"] p {
            font-weight: 600 !important;
            font-size: 0.86rem !important;
            margin: 0 !important;
        }

        /* Glass pill expanders ("View chart data") & tooltips */
        [data-testid="stExpander"] {
            border-radius: 13px !important;
            background: linear-gradient(180deg, rgba(7, 22, 54, 0.74), rgba(4, 14, 34, 0.88)) !important;
            border: 1px solid rgba(56, 189, 248, 0.28) !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.24), inset 0 1px 0 rgba(186, 230, 253, 0.10) !important;
            overflow: hidden !important;
            transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
        }
        [data-testid="stExpander"]:hover {
            border-color: rgba(56, 189, 248, 0.56) !important;
            box-shadow: 0 0 18px rgba(56, 189, 248, 0.18), inset 0 1px 0 rgba(224, 242, 254, 0.16) !important;
        }
        [data-testid="stExpander"] summary {
            padding: 0.60rem 0.90rem !important;
            transition: background 0.18s ease !important;
        }
        [data-testid="stExpander"] summary:hover {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.16), rgba(37, 99, 235, 0.10)) !important;
        }
        [data-testid="stExpander"] details[open] > summary {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.26), rgba(37, 99, 235, 0.18)) !important;
            border-bottom: 1px solid rgba(56, 189, 248, 0.38) !important;
        }
        [data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
            color: #38BDF8 !important;
        }
        [data-testid="stTooltipIcon"] button {
            color: #7DD3FC !important;
            transition: color 0.16s ease, filter 0.16s ease !important;
        }
        [data-testid="stTooltipIcon"] button:hover {
            color: #BAE6FD !important;
            filter: drop-shadow(0 0 6px rgba(56, 189, 248, 0.72)) !important;
        }

        /* Bottom sidebar Log out HUD pill button */
        [data-testid="stSidebar"] .st-key-logout_btn button {
            min-height: 42px !important;
            border-radius: 12px !important;
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.20), rgba(37, 99, 235, 0.14)) !important;
            border: 1px solid rgba(56, 189, 248, 0.46) !important;
            color: #F0F9FF !important;
            box-shadow:
                0 0 18px rgba(56, 189, 248, 0.16),
                inset 0 1px 0 rgba(224, 242, 254, 0.18),
                inset 0 -2px 0 rgba(56, 189, 248, 0.65) !important;
            transition: all 0.18s ease !important;
        }
        [data-testid="stSidebar"] .st-key-logout_btn button:hover {
            transform: translateY(-1px) !important;
            border-color: rgba(248, 113, 113, 0.68) !important;
            background: linear-gradient(135deg, rgba(185, 28, 28, 0.34), rgba(127, 29, 29, 0.28)) !important;
            color: #FEE2E2 !important;
            box-shadow: 0 0 18px rgba(248, 113, 113, 0.28), inset 0 -2px 0 rgba(248, 113, 113, 0.85) !important;
        }
        """
    else:
        dark_login = bool(st.session_state.get("login_dark_mode", True))
        if dark_login:
            bg_override_css = _fixed_bg_css(
                LOGIN_BG_FILE,
                app_bg_color="#030A1C",
                bg_filter="brightness(1.10)",
                opacity="1",
            )
        else:
            light_bg = LOGIN_BG_LIGHT_FILE if LOGIN_BG_LIGHT_FILE.is_file() else LOGIN_BG_FILE
            bg_override_css = _fixed_bg_css(
                light_bg,
                app_bg_color="#EDF5FF",
                bg_filter="brightness(1.01) contrast(1.03)",
                opacity="1",
            )
        light_card_overrides = ""
        if not dark_login:
            light_card_overrides = """
        .st-key-login_card,
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ms-login-anchor) {
            background: rgba(255, 255, 255, 0.94) !important;
            border: 1px solid rgba(14, 165, 233, 0.28) !important;
            box-shadow:
                0 24px 60px rgba(15, 23, 42, 0.11),
                0 6px 20px rgba(2, 132, 199, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 1) !important;
        }
        .st-key-login_card::before {
            height: 2px !important;
            background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.55), transparent) !important;
            box-shadow: none !important;
        }
        .st-key-login_card::after {
            display: none !important;
        }
        .ms-login-medallion {
            background: linear-gradient(145deg, #38BDF8 0%, #0284C7 100%) !important;
            border: 3px solid #FFFFFF !important;
            box-shadow:
                0 10px 26px rgba(2, 132, 199, 0.34),
                0 0 0 1px rgba(14, 165, 233, 0.22) !important;
        }
        .ms-login-medallion svg {
            filter: drop-shadow(0 1px 4px rgba(3, 105, 161, 0.25)) !important;
        }
        .ms-login-medallion path:nth-of-type(1) {
            fill: rgba(255, 255, 255, 0.22) !important;
            stroke: #FFFFFF !important;
        }
        .ms-login-medallion path:nth-of-type(2) {
            stroke: rgba(255, 255, 255, 0.70) !important;
        }
        .ms-login-medallion path:nth-of-type(3) {
            stroke: #FFFFFF !important;
        }
        .ms-login-title,
        .ms-login-corner-title,
        .st-key-login_card [data-testid="stTextInput"] input,
        .st-key-login_card [data-testid="stCheckbox"] label p,
        .ms-demo-item,
        .ms-demo-right,
        .ms-demo-right strong,
        .st-key-login_dark_pill [data-testid="stCheckbox"] label p {
            color: #0F172A !important;
            text-shadow: none !important;
        }
        .ms-portal-accent {
            color: #0284C7 !important;
            text-shadow: none !important;
        }
        .ms-login-sub,
        .ms-login-divider,
        .ms-login-corner-sub,
        .ms-demo-head-note {
            color: #475569 !important;
            text-shadow: none !important;
        }
        .ms-login-foot {
            color: #475569 !important;
            font-weight: 500 !important;
            text-shadow: 0 1px 6px rgba(255, 255, 255, 0.85) !important;
        }
        .ms-login-divider::before,
        .ms-login-divider::after {
            background: rgba(148, 163, 184, 0.35) !important;
        }
        .ms-login-badge {
            background: #EFF6FF !important;
            border-color: #BAE6FD !important;
            color: #0369A1 !important;
            box-shadow: none !important;
        }
        .ms-login-badge svg circle {
            fill: rgba(2, 132, 199, 0.14) !important;
            stroke: #0284C7 !important;
        }
        .ms-login-badge svg path {
            stroke: #0284C7 !important;
        }
        .st-key-login_dark_pill {
            background: rgba(255, 255, 255, 0.90) !important;
            border: 1px solid rgba(148, 163, 184, 0.38) !important;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.95) !important;
        }
        .st-key-login_dark_pill [data-testid="stCheckbox"] label::after {
            background: #0284C7 !important;
        }
        .st-key-login_dark_pill [data-testid="stCheckbox"] label > div:first-of-type {
            background-color: #CBD5E1 !important;
            border: 1px solid #94A3B8 !important;
        }
        .st-key-login_card [data-testid="stTextInput"] [data-testid="stTextInputRootElement"],
        .st-key-login_card [data-testid="stTextInput"] [data-baseweb="input"] {
            background: #F4F7FB !important;
            border: 1.5px solid #D6E1EE !important;
            color: #0F172A !important;
        }
        .st-key-login_card [data-testid="stTextInput"] [data-testid="stTextInputRootElement"]:focus-within,
        .st-key-login_card [data-testid="stTextInput"] [data-baseweb="input"]:focus-within {
            background: #FFFFFF !important;
            border-color: #0EA5E9 !important;
            box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.16) !important;
        }
        .st-key-login_card [data-testid="stTextInput"] input::placeholder,
        .st-key-login_card [data-testid="stTextInput"] svg,
        .st-key-login_card [data-testid="stTextInput"] span[data-testid="stIconMaterial"] {
            color: #64748B !important;
        }
        .st-key-login_card [data-testid="stTextInput"] [data-testid="stTextInputRootElement"]:focus-within svg,
        .st-key-login_card [data-testid="stTextInput"] [data-testid="stTextInputRootElement"]:focus-within span[data-testid="stIconMaterial"] {
            color: #0284C7 !important;
        }
        .st-key-login_card .st-key-login_submit_btn button,
        .st-key-login_card .st-key-login_submit_btn button[kind="primary"] {
            background: linear-gradient(95deg, #0EA5E9 0%, #2563EB 100%) !important;
            border: 1px solid rgba(2, 132, 199, 0.40) !important;
            box-shadow: 0 8px 22px rgba(2, 132, 199, 0.28) !important;
        }
        .st-key-login_card .st-key-login_submit_btn button:hover,
        .st-key-login_card .st-key-login_submit_btn button[kind="primary"]:hover {
            box-shadow: 0 10px 26px rgba(2, 132, 199, 0.38) !important;
        }
        .st-key-login_card .st-key-guest_access_btn button,
        .st-key-login_card .st-key-guest_access_btn button[kind="secondary"] {
            background: #FFFFFF !important;
            border: 1.5px solid #CBD5E1 !important;
            color: #0F172A !important;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05) !important;
        }
        .st-key-login_card .st-key-guest_access_btn button p {
            color: #0F172A !important;
        }
        .st-key-login_card .st-key-guest_access_btn span[data-testid="stIconMaterial"] {
            color: #0284C7 !important;
        }
        .st-key-login_card .st-key-guest_access_btn button:hover,
        .st-key-login_card .st-key-guest_access_btn button[kind="secondary"]:hover {
            background: #F0F9FF !important;
            border-color: #38BDF8 !important;
            color: #0369A1 !important;
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.14) !important;
        }
        .st-key-login_card [data-testid="stCheckbox"] label > div:first-of-type {
            border-color: #94A3B8 !important;
            background-color: #FFFFFF !important;
        }
        .st-key-login_card [data-testid="stCheckbox"]:has(input:checked) label > div:first-of-type {
            border-color: #0284C7 !important;
            background-color: #0284C7 !important;
        }
        .st-key-forgot_pwd_btn button,
        .ms-demo-head-title {
            color: #0284C7 !important;
        }
        .st-key-forgot_pwd_btn button:hover {
            color: #0369A1 !important;
        }
        .ms-demo-box {
            background: linear-gradient(180deg, #F4F8FD 0%, #EAF2FB 100%) !important;
            border: 1px solid #C8DCF4 !important;
        }
        .ms-demo-creds {
            border-right-color: #CBDDF2 !important;
        }
        .ms-demo-code {
            background: #FFFFFF !important;
            border: 1px solid #BAE6FD !important;
            color: #0369A1 !important;
            box-shadow: 0 1px 2px rgba(2, 132, 199, 0.06) !important;
        }
        .ms-login-corner-brand {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            box-shadow: none !important;
        }
        .ms-login-corner-img {
            filter: drop-shadow(0 2px 6px rgba(14, 165, 233, 0.14)) !important;
        }
            """
        login_only_css = f"""
        [data-testid="stSidebar"],
        [data-testid="stExpandSidebarButton"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"],
        button[kind="headerNoPadding"],
        [data-testid="stHeader"] {{
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }}
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {{
            display: none !important;
        }}
        [data-testid="stMain"] {{
            overflow-x: hidden !important;
        }}
        [data-testid="stMainBlockContainer"] {{
            max-width: 100% !important;
            width: 100% !important;
            min-height: 100vh !important;
            margin: 0 auto !important;
            padding: 36px 16px 18px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: safe center !important;
            align-items: center !important;
        }}
        [data-testid="stMainBlockContainer"] > div,
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {{
            width: 100% !important;
            min-height: calc(100vh - 54px) !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: safe center !important;
            align-items: center !important;
            gap: 0 !important;
        }}
        [data-testid="stMainBlockContainer"] [data-testid="stLayoutWrapper"]:has(.st-key-login_card) {{
            width: min(472px, calc(100vw - 32px)) !important;
            max-width: 472px !important;
            margin: 0 auto !important;
            align-self: center !important;
        }}
        [data-testid="stMainBlockContainer"] [data-testid="stElementContainer"]:has(.ms-login-foot) {{
            width: min(472px, calc(100vw - 32px)) !important;
            max-width: 472px !important;
            margin: 0.65rem auto 0 !important;
            align-self: center !important;
        }}
        [data-testid="stMainBlockContainer"] [data-testid="stLayoutWrapper"]:has(.st-key-login_dark_pill),
        [data-testid="stElementContainer"]:has(.ms-login-corner-brand) {{
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 0 !important;
            height: 0 !important;
            overflow: visible !important;
            z-index: 1000000 !important;
        }}
        {bg_override_css}
        {light_card_overrides}
        """

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {{
        --ms-bg: #070B14;
        --ms-surface: #0D1526;
        --ms-surface-2: #121D34;
        --ms-border: rgba(148, 163, 184, 0.16);
        --ms-border-cyan: rgba(56, 189, 248, 0.30);
        --ms-cyan: #38BDF8;
        --ms-teal: #2DD4BF;
        --ms-indigo: #818CF8;
        --ms-amber: #FB923C;
        --ms-text: #F1F5F9;
        --ms-muted: #94A3B8;
    }}

    @keyframes msFadeUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes msPulseDot {{
        0%, 100% {{ opacity: 1; box-shadow: 0 0 0 0 rgba(45, 212, 191, 0.55); }}
        50% {{ opacity: 0.75; box-shadow: 0 0 0 6px rgba(45, 212, 191, 0); }}
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: var(--ms-text);
    }}

    h1, h2, h3, h4, .ms-display {{
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
        letter-spacing: -0.02em;
    }}

    code, pre, .ms-mono {{
        font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace !important;
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: var(--ms-bg);
        background-image:
            radial-gradient(900px 440px at 12% -5%, rgba(56, 189, 248, 0.11), transparent 65%),
            radial-gradient(820px 420px at 92% 0%, rgba(129, 140, 248, 0.10), transparent 60%),
            linear-gradient(rgba(148, 163, 184, 0.032) 1px, transparent 1px),
            linear-gradient(90deg, rgba(148, 163, 184, 0.032) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 34px 34px, 34px 34px;
        background-attachment: fixed;
    }}

    [data-testid="stMain"] {{
        scroll-padding-top: 5rem;
    }}

    [data-testid="stHeader"] {{
        background: rgba(3, 11, 30, 0.72) !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-bottom: 1px solid rgba(56, 189, 248, 0.24) !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.38);
    }}

    [data-testid="stMainBlockContainer"] {{
        padding-top: 4.85rem !important;
        padding-bottom: 3rem !important;
        animation: msFadeUp 0.42s ease-out;
    }}

    /* Sidebar dark-tech telemetry navigation styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(4, 15, 42, 0.80) 0%, rgba(2, 9, 24, 0.88) 100%) !important;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-right: 1px solid rgba(56, 189, 248, 0.28) !important;
        box-shadow: 4px 0 32px rgba(0, 0, 0, 0.45);
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: #CBD5E1;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] .react-aria-ComboBox > div:first-of-type {{
        background: rgba(5, 16, 40, 0.78) !important;
        border: 1px solid rgba(56, 189, 248, 0.26) !important;
        border-radius: 10px !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] > div:hover,
    [data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within,
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] .react-aria-ComboBox > div:first-of-type:hover,
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] .react-aria-ComboBox > div:first-of-type:focus-within {{
        border-color: rgba(56, 189, 248, 0.60) !important;
        box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.22) !important;
    }}
    .ms-sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 0.78rem 0.9rem;
        margin-bottom: 0.7rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.18), rgba(37, 99, 235, 0.10));
        border: 1px solid rgba(56, 189, 248, 0.36);
        box-shadow: 0 0 24px rgba(14, 165, 233, 0.14), inset 0 1px 0 rgba(186, 230, 253, 0.16);
    }}
    .ms-sidebar-brand-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.15;
    }}
    .ms-sidebar-brand-sub {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #38BDF8;
        margin-top: 2px;
    }}
    .ms-user-pill {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.55rem 0.8rem;
        border-radius: 10px;
        background: rgba(6, 20, 48, 0.76);
        border: 1px solid rgba(56, 189, 248, 0.24);
        margin-bottom: 0.55rem;
    }}
    .ms-user-name {{
        font-size: 0.84rem;
        font-weight: 600;
        color: #F1F5F9;
        display: flex;
        align-items: center;
        gap: 7px;
    }}
    .ms-status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: #2DD4BF;
        display: inline-block;
        animation: msPulseDot 2.4s infinite;
    }}
    .ms-role-tag {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.66rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 2px 7px;
        border-radius: 999px;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.14);
        border: 1px solid rgba(56, 189, 248, 0.30);
    }}
    .ms-filter-group {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #7DD3FC;
        margin: 0.9rem 0 0.35rem 0;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid rgba(56, 189, 248, 0.20);
    }}
    .ms-sidebar-telemetry {{
        padding: 0.85rem 0.95rem;
        border-radius: 12px;
        background: rgba(6, 22, 54, 0.80);
        border: 1px solid rgba(56, 189, 248, 0.32);
        box-shadow: 0 0 20px rgba(14, 165, 233, 0.10);
        margin-top: 0.5rem;
    }}
    .ms-sidebar-telemetry-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.67rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8;
    }}
    .ms-sidebar-telemetry-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.45rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 2px;
    }}
    .ms-sidebar-telemetry-sub {{
        font-size: 0.76rem;
        color: #2DD4BF;
        margin-top: 2px;
    }}

    /* Multiselect tags */
    [data-baseweb="tag"] {{
        background: rgba(56, 189, 248, 0.18) !important;
        border: 1px solid rgba(56, 189, 248, 0.40) !important;
        border-radius: 7px !important;
    }}
    [data-baseweb="tag"] span {{
        color: #E0F2FE !important;
    }}

    /* Hero header */
    .ms-hero {{
        position: relative;
        overflow: hidden;
        padding: 1.35rem 1.65rem;
        margin-bottom: 1.15rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(6, 21, 52, 0.78) 0%, rgba(3, 12, 32, 0.86) 100%);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(56, 189, 248, 0.34);
        box-shadow: 0 18px 46px rgba(0, 0, 0, 0.44), 0 0 28px rgba(14, 165, 233, 0.12), inset 0 1px 0 rgba(186, 230, 253, 0.12);
    }}
    .ms-hero::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #38BDF8, #0EA5E9, #2DD4BF, rgba(56, 189, 248, 0.15));
    }}
    .ms-hero-top {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 0.75rem;
    }}
    .ms-hero-badges {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
    }}
    .ms-chip {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(14, 165, 233, 0.14);
        border: 1px solid rgba(56, 189, 248, 0.34);
        color: #7DD3FC;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }}
    .ms-chip-teal {{
        background: rgba(45, 212, 191, 0.14);
        border-color: rgba(45, 212, 191, 0.34);
        color: #5EEAD4;
    }}
    .ms-hero-title-row {{
        display: flex;
        align-items: center;
        gap: 14px;
    }}
    [data-testid="stMarkdownContainer"] h1.ms-hero-title,
    .ms-hero-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.45rem, 2.6vw, 2.05rem);
        font-weight: 700;
        line-height: 1.15 !important;
        padding: 0 !important;
        margin: 0 !important;
        background: linear-gradient(95deg, #F8FAFC 15%, #BAE6FD 65%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .ms-hero-sub {{
        margin: 0.55rem 0 0 0;
        color: #CBD5E1;
        font-size: 0.94rem;
        line-height: 1.5;
        max-width: 72ch;
    }}

    /* KPI telemetry cards */
    .ms-kpi {{
        position: relative;
        overflow: hidden;
        padding: 0.95rem 1.05rem 0.9rem;
        border-radius: 14px;
        background: linear-gradient(170deg, rgba(7, 22, 54, 0.78), rgba(4, 13, 34, 0.86));
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.22);
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.32);
        transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        margin-bottom: 0.75rem;
    }}
    .ms-kpi:hover {{
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.46);
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.44), 0 0 18px rgba(56, 189, 248, 0.14);
    }}
    .ms-kpi-bar {{
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--kpi-accent, #38BDF8);
        opacity: 0.92;
    }}
    .ms-kpi-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 0.35rem;
    }}
    .ms-kpi-value {{
        font-family: 'JetBrains Mono', 'Space Grotesk', monospace;
        font-size: clamp(1.32rem, 2vw, 1.65rem);
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.15;
        font-variant-numeric: tabular-nums;
    }}
    .ms-kpi-sub {{
        font-size: 0.76rem;
        color: #94A3B8;
        margin-top: 0.36rem;
        line-height: 1.3;
    }}

    /* Segmented HUD navigation tabs (Streamlit 1.66 React Aria + BaseWeb fallback) */
    [data-testid="stTabs"] [role="tablist"],
    [data-baseweb="tab-list"] {{
        width: 100% !important;
        gap: 8px !important;
        padding: 6px !important;
        border-radius: 14px !important;
        background: linear-gradient(180deg, rgba(6, 20, 50, 0.80), rgba(3, 12, 32, 0.88)) !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(56, 189, 248, 0.30) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.38), inset 0 1px 0 rgba(186, 230, 253, 0.12) !important;
        margin-bottom: 0.95rem !important;
    }}
    [data-testid="stTabs"] [data-testid="stTab"],
    [data-testid="stTabs"] [role="tab"],
    [data-baseweb="tab"] {{
        height: auto !important;
        min-height: 38px !important;
        border-radius: 10px !important;
        padding: 0.48rem 1.05rem !important;
        font-weight: 600 !important;
        font-size: 0.89rem !important;
        color: #94A3B8 !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        border-bottom: none !important;
        transition: all 0.18s ease !important;
    }}
    [data-testid="stTabs"] [data-testid="stTab"] p,
    [data-testid="stTabs"] [role="tab"] p {{
        color: inherit !important;
        font-weight: inherit !important;
        margin: 0 !important;
    }}
    [data-testid="stTabs"] [data-testid="stTab"]:hover,
    [data-testid="stTabs"] [role="tab"]:hover,
    [data-baseweb="tab"]:hover {{
        color: #E0F2FE !important;
        background: rgba(56, 189, 248, 0.10) !important;
        border-color: rgba(56, 189, 248, 0.25) !important;
    }}
    [data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"],
    [data-testid="stTabs"] [role="tab"][aria-selected="true"],
    [data-baseweb="tab"][aria-selected="true"] {{
        color: #F8FAFC !important;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.34), rgba(37, 99, 235, 0.24)) !important;
        border: 1px solid rgba(56, 189, 248, 0.62) !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.25), inset 0 1px 0 rgba(224, 242, 254, 0.22) !important;
    }}
    [data-testid="stTabs"] [role="tablist"] > div:not([role="tab"]),
    [data-baseweb="tab-highlight"],
    [data-baseweb="tab-border"] {{
        display: none !important;
    }}

    /* Glassmorphic bordered containers (chart panels & insight blocks) */
    div[class*="st-key-card_"],
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: linear-gradient(175deg, rgba(6, 20, 48, 0.78), rgba(3, 12, 30, 0.86)) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.22) !important;
        border-radius: 16px !important;
        box-shadow: 0 12px 34px rgba(0, 0, 0, 0.36) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }}
    div[class*="st-key-card_"]:hover,
    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        border-color: rgba(56, 189, 248, 0.42) !important;
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.44), 0 0 18px rgba(56, 189, 248, 0.10) !important;
    }}
    div[class*="st-key-card_"] {{
        padding: 1.05rem 1.15rem !important;
    }}
    /* Crisp dark input fields on login and filters */
    [data-testid="stTextInput"] input {{
        background: rgba(7, 11, 20, 0.78) !important;
        border: 1px solid rgba(148, 163, 184, 0.24) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
    }}
    [data-testid="stTextInput"] input:focus {{
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.22) !important;
    }}
    .ms-panel-tag {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.67rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #38BDF8;
        padding: 2px 8px;
        border-radius: 6px;
        background: rgba(56, 189, 248, 0.10);
        border: 1px solid rgba(56, 189, 248, 0.22);
        margin-bottom: 0.3rem;
    }}

    /* Buttons & inputs */
    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {{
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.18s ease !important;
    }}
    .stButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] > button[kind="primary"] {{
        background: linear-gradient(135deg, #0284C7 0%, #2563EB 55%, #4F46E5 100%) !important;
        border: 1px solid rgba(125, 211, 252, 0.45) !important;
        color: #FFFFFF !important;
        box-shadow: 0 8px 24px rgba(2, 132, 199, 0.34) !important;
    }}
    .stButton > button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {{
        transform: translateY(-1px);
        box-shadow: 0 10px 28px rgba(56, 189, 248, 0.42) !important;
    }}
    .stButton > button[kind="secondary"] {{
        background: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        color: #E2E8F0 !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        border-color: rgba(56, 189, 248, 0.45) !important;
        color: #38BDF8 !important;
    }}
    [data-testid="stSidebar"] .st-key-logout_btn {{
        margin-top: 0.45rem !important;
    }}
    [data-testid="stSidebar"] .st-key-logout_btn button {{
        background: rgba(6, 20, 48, 0.84) !important;
        border: 1px solid rgba(56, 189, 248, 0.30) !important;
        color: #E0F2FE !important;
    }}
    [data-testid="stSidebar"] .st-key-logout_btn button:hover {{
        border-color: rgba(248, 113, 113, 0.55) !important;
        background: rgba(127, 29, 29, 0.28) !important;
        color: #FCA5A5 !important;
    }}
    [data-testid="stSidebarContent"]::-webkit-scrollbar,
    [data-testid="stMain"]::-webkit-scrollbar {{
        width: 7px;
    }}
    [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb,
    [data-testid="stMain"]::-webkit-scrollbar-thumb {{
        background: rgba(56, 189, 248, 0.28);
        border-radius: 999px;
    }}
    [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb:hover,
    [data-testid="stMain"]::-webkit-scrollbar-thumb:hover {{
        background: rgba(56, 189, 248, 0.52);
    }}

    /* Tactile precision-instrument sliders across all charts and sidebar */
    [data-testid="stSlider"] {{
        padding-top: 0.15rem;
        padding-bottom: 0.2rem;
    }}
    [data-testid="stSlider"] [data-testid="stWidgetLabel"] {{
        margin-bottom: 1.65rem !important;
    }}
    [data-testid="stSlider"] [data-testid="stWidgetLabel"] p {{
        font-weight: 500 !important;
        color: #E2E8F0 !important;
        transition: color 0.18s ease !important;
    }}
    [data-testid="stSlider"]:hover [data-testid="stWidgetLabel"] p,
    [data-testid="stSlider"]:focus-within [data-testid="stWidgetLabel"] p {{
        color: #F8FAFC !important;
    }}
    [data-testid="stSlider"] [role="group"] {{
        padding: 0 10px !important;
        cursor: pointer !important;
    }}
    [data-testid="stSlider"] [role="group"]:active {{
        cursor: grabbing !important;
    }}
    [data-testid="stSlider"] [role="group"] > div[data-orientation="horizontal"] {{
        height: 22px !important;
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
    }}
    /* Preserve Streamlit's inline linear-gradient track progress while refining geometry & glow */
    [data-testid="stSlider"] [role="group"] > div[data-orientation="horizontal"] > div:first-child {{
        height: 8px !important;
        border-radius: 999px !important;
        box-shadow:
            inset 0 1px 2px rgba(2, 6, 23, 0.65),
            0 0 10px rgba(56, 189, 248, 0.16) !important;
        transition: height 0.16s ease, box-shadow 0.2s ease !important;
    }}
    [data-testid="stSlider"]:hover [role="group"] > div[data-orientation="horizontal"] > div:first-child {{
        box-shadow:
            inset 0 1px 2px rgba(2, 6, 23, 0.55),
            0 0 16px rgba(56, 189, 248, 0.34) !important;
    }}
    [data-testid="stSlider"]:active [role="group"] > div[data-orientation="horizontal"] > div:first-child,
    [data-testid="stSlider"]:has([data-dragging="true"]) [role="group"] > div[data-orientation="horizontal"] > div:first-child {{
        height: 10px !important;
        box-shadow:
            inset 0 1px 2px rgba(2, 6, 23, 0.5),
            0 0 22px rgba(45, 212, 191, 0.52) !important;
    }}
    /* Precision luminous orb thumb — keep position instant (60fps) while smoothly animating size and halo */
    [data-testid="stSlider"] [role="group"] > div[data-orientation="horizontal"] > div[data-rac] {{
        width: 20px !important;
        height: 20px !important;
        border-radius: 999px !important;
        background: radial-gradient(circle at 36% 32%, #FFFFFF 0%, #7DD3FC 38%, #0EA5E9 75%, #0284C7 100%) !important;
        border: 2px solid rgba(248, 250, 252, 0.95) !important;
        box-shadow:
            0 0 0 3px rgba(56, 189, 248, 0.18),
            0 3px 10px rgba(0, 0, 0, 0.45),
            0 0 14px rgba(56, 189, 248, 0.55) !important;
        cursor: grab !important;
        transition: width 0.16s cubic-bezier(0.34, 1.56, 0.64, 1),
                    height 0.16s cubic-bezier(0.34, 1.56, 0.64, 1),
                    box-shadow 0.18s ease,
                    background 0.18s ease,
                    border-color 0.18s ease !important;
    }}
    [data-testid="stSlider"] [role="group"] > div[data-orientation="horizontal"] > div[data-rac]:hover {{
        cursor: grab !important;
        width: 23px !important;
        height: 23px !important;
        box-shadow:
            0 0 0 6px rgba(56, 189, 248, 0.26),
            0 0 22px rgba(56, 189, 248, 0.85),
            0 6px 16px rgba(0, 0, 0, 0.50) !important;
    }}
    [data-testid="stSlider"] [role="group"] > div[data-orientation="horizontal"] > div[data-rac]:active,
    [data-testid="stSlider"] [role="group"] > div[data-orientation="horizontal"] > div[data-rac]:focus-within,
    [data-testid="stSlider"] [role="group"] > div[data-orientation="horizontal"] > div[data-rac][data-dragging="true"] {{
        cursor: grabbing !important;
        width: 25px !important;
        height: 25px !important;
        background: radial-gradient(circle at 36% 32%, #FFFFFF 0%, #A5F3FC 38%, #2DD4BF 75%, #0EA5E9 100%) !important;
        border-color: #FFFFFF !important;
        box-shadow:
            0 0 0 9px rgba(45, 212, 191, 0.30),
            0 0 28px rgba(45, 212, 191, 0.95),
            0 8px 20px rgba(0, 0, 0, 0.55) !important;
    }}
    /* Floating telemetry value pill above the thumb */
    [data-testid="stSliderThumbValue"] {{
        top: auto !important;
        bottom: calc(100% + 7px) !important;
        padding: 2px 9px !important;
        border-radius: 999px !important;
        background: linear-gradient(135deg, rgba(6, 22, 54, 0.94), rgba(14, 165, 233, 0.24)) !important;
        border: 1px solid rgba(56, 189, 248, 0.52) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.38), 0 0 12px rgba(56, 189, 248, 0.22) !important;
        line-height: 1.2 !important;
        pointer-events: none !important;
        transition: bottom 0.16s cubic-bezier(0.34, 1.56, 0.64, 1),
                    background 0.18s ease,
                    border-color 0.18s ease,
                    box-shadow 0.18s ease !important;
    }}
    [data-testid="stSliderThumbValue"] p {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        color: #E0F2FE !important;
        margin: 0 !important;
        line-height: 1.25 !important;
        transition: color 0.18s ease !important;
    }}
    [data-testid="stSlider"] [role="group"] > div[data-orientation="horizontal"] > div[data-rac]:hover [data-testid="stSliderThumbValue"] {{
        bottom: calc(100% + 9px) !important;
        border-color: rgba(103, 232, 249, 0.85) !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.45), 0 0 16px rgba(56, 189, 248, 0.48) !important;
    }}
    [data-testid="stSlider"] [role="group"] > div[data-orientation="horizontal"] > div[data-rac]:active [data-testid="stSliderThumbValue"],
    [data-testid="stSlider"] [role="group"] > div[data-orientation="horizontal"] > div[data-rac]:focus-within [data-testid="stSliderThumbValue"],
    [data-testid="stSlider"] [role="group"] > div[data-orientation="horizontal"] > div[data-rac][data-dragging="true"] [data-testid="stSliderThumbValue"] {{
        bottom: calc(100% + 10px) !important;
        background: linear-gradient(135deg, #38BDF8 0%, #2DD4BF 100%) !important;
        border-color: #FFFFFF !important;
        box-shadow: 0 6px 18px rgba(45, 212, 191, 0.55), 0 0 22px rgba(56, 189, 248, 0.72) !important;
    }}
    [data-testid="stSlider"] [role="group"] > div[data-orientation="horizontal"] > div[data-rac]:active [data-testid="stSliderThumbValue"] p,
    [data-testid="stSlider"] [role="group"] > div[data-orientation="horizontal"] > div[data-rac]:focus-within [data-testid="stSliderThumbValue"] p,
    [data-testid="stSlider"] [role="group"] > div[data-orientation="horizontal"] > div[data-rac][data-dragging="true"] [data-testid="stSliderThumbValue"] p {{
        color: #041024 !important;
    }}
    [data-testid="stSliderTickBar"] {{
        margin-top: 2px !important;
    }}
    [data-testid="stSliderTickBar"] p {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.71rem !important;
        color: rgba(148, 163, 184, 0.85) !important;
        transition: color 0.18s ease !important;
    }}
    [data-testid="stSlider"]:hover [data-testid="stSliderTickBar"] p {{
        color: #7DD3FC !important;
    }}

    /* Tables & Expanders */
    [data-testid="stExpander"] {{
        border: 1px solid rgba(148, 163, 184, 0.16) !important;
        border-radius: 11px !important;
        background: rgba(8, 13, 25, 0.55) !important;
    }}
    [data-testid="stTable"] table {{
        border-collapse: collapse;
        width: 100%;
    }}
    [data-testid="stTable"] th {{
        background: rgba(15, 23, 42, 0.92) !important;
        color: #7DD3FC !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border-bottom: 1px solid rgba(56, 189, 248, 0.24) !important;
    }}
    [data-testid="stTable"] td {{
        border-bottom: 1px solid rgba(148, 163, 184, 0.11) !important;
        color: #E2E8F0 !important;
        font-size: 0.88rem;
    }}

    /* Login card styling */
    .ms-login-corner-brand {{
        position: fixed;
        top: 12px;
        left: 18px;
        z-index: 30;
        display: flex;
        align-items: center;
        gap: 11px;
        pointer-events: none;
    }}
    .ms-login-corner-img {{
        display: block;
        height: clamp(76px, 6.8vw, 96px);
        width: auto;
        filter: drop-shadow(0 0 12px rgba(56, 189, 248, 0.35));
    }}
    .ms-login-corner-icon {{
        display: flex;
        align-items: center;
        filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.72));
    }}
    .ms-login-corner-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.02rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.15;
        text-shadow: 0 2px 12px rgba(0, 0, 0, 0.78);
    }}
    .ms-login-corner-sub {{
        font-family: 'Inter', sans-serif;
        font-size: 0.74rem;
        color: #94A3B8;
        margin-top: 1px;
    }}
    .st-key-login_dark_pill {{
        position: fixed !important;
        top: clamp(26px, 2.4vw, 35px) !important;
        right: 24px !important;
        z-index: 1000001 !important;
        pointer-events: auto !important;
        width: auto !important;
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
        background: rgba(5, 16, 40, 0.42) !important;
        border: 1px solid rgba(56, 189, 248, 0.48) !important;
        border-radius: 999px !important;
        padding: 7px 18px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.32), 0 0 18px rgba(14, 165, 233, 0.14), inset 0 1px 0 rgba(186, 230, 253, 0.14) !important;
        backdrop-filter: blur(10px) !important;
    }}
    .st-key-login_dark_pill [data-testid="stVerticalBlock"] {{
        gap: 0 !important;
    }}
    .st-key-login_dark_pill [data-testid="stCheckbox"] {{
        margin: 0 !important;
        padding: 0 !important;
        min-height: 0 !important;
    }}
    .st-key-login_dark_pill [data-testid="stCheckbox"] label {{
        flex-direction: row-reverse !important;
        align-items: center !important;
        gap: 12px !important;
        margin: 0 !important;
        padding: 2px 0 !important;
        cursor: pointer !important;
    }}
    .st-key-login_dark_pill [data-testid="stCheckbox"] label p {{
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: #F1F5F9 !important;
        white-space: nowrap !important;
        letter-spacing: 0.01em !important;
    }}
    .st-key-login_dark_pill [data-testid="stCheckbox"] label > div:first-of-type {{
        transform: scale(1.16) !important;
        transform-origin: center right !important;
    }}
    .st-key-login_dark_pill [data-testid="stCheckbox"] label::after {{
        content: "";
        width: 16px;
        height: 16px;
        display: inline-block;
        flex-shrink: 0;
        background: #E0F2FE;
        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M21 12.79A9 9 0 1 1 11.21 3a7 7 0 0 0 9.79 9.79z'/%3E%3C/svg%3E") center / contain no-repeat;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M21 12.79A9 9 0 1 1 11.21 3a7 7 0 0 0 9.79 9.79z'/%3E%3C/svg%3E") center / contain no-repeat;
    }}
    @media (max-width: 720px) {{
        .ms-login-corner-brand {{
            display: none;
        }}
        .st-key-login_dark_pill {{
            top: 10px !important;
            right: 12px !important;
            min-height: 36px !important;
            padding: 4px 12px !important;
        }}
    }}
    .st-key-login_card,
    [data-testid="stVerticalBlockBorderWrapper"]:has(.ms-login-anchor) {{
        position: relative !important;
        overflow: visible !important;
        width: 100% !important;
        max-width: 472px !important;
        height: auto !important;
        margin: 0 auto !important;
        background:
            radial-gradient(320px 160px at 50% 0%, rgba(56, 189, 248, 0.22), transparent 70%),
            radial-gradient(320px 120px at 50% 100%, rgba(14, 165, 233, 0.16), transparent 72%),
            linear-gradient(175deg, rgba(6, 18, 44, 0.64) 0%, rgba(4, 12, 32, 0.78) 100%) !important;
        border: 1.5px solid rgba(76, 195, 255, 0.78) !important;
        border-radius: 22px !important;
        padding: 1.55rem 1.85rem 1.4rem !important;
        box-shadow:
            0 0 0 1px rgba(125, 211, 252, 0.22),
            0 24px 72px rgba(0, 0, 0, 0.70),
            0 0 40px rgba(14, 165, 233, 0.36),
            0 0 76px rgba(2, 132, 199, 0.18),
            inset 0 1px 0 rgba(186, 230, 253, 0.55),
            inset 0 -1px 0 rgba(56, 189, 248, 0.35) !important;
        backdrop-filter: blur(13px) !important;
        -webkit-backdrop-filter: blur(13px) !important;
    }}
    .st-key-login_card > div,
    .st-key-login_card [data-testid="stVerticalBlock"] {{
        overflow: visible !important;
        gap: 0.40rem !important;
        width: 100% !important;
        height: auto !important;
        padding-bottom: 0 !important;
    }}
    .st-key-login_card [data-testid="stElementContainer"]:has(.ms-login-header),
    .st-key-login_card [data-testid="stMarkdownContainer"]:has(.ms-login-header) {{
        width: 100% !important;
        text-align: center !important;
    }}
    .st-key-login_card [data-testid="stElementContainer"],
    .st-key-login_card [data-testid="stMarkdownContainer"] {{
        margin-bottom: 0 !important;
        height: auto !important;
    }}
    .st-key-login_card::before {{
        content: "";
        position: absolute;
        top: -1.5px;
        left: 14%;
        right: 14%;
        height: 2.5px;
        background: linear-gradient(90deg, transparent, #A5F3FC, #38BDF8, #A5F3FC, transparent);
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.95);
        pointer-events: none;
    }}
    .st-key-login_card::after {{
        content: "";
        position: absolute;
        bottom: -1.5px;
        left: 16%;
        right: 16%;
        height: 2.5px;
        background: linear-gradient(90deg, transparent, #7DD3FC, #0EA5E9, #7DD3FC, transparent);
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.90);
        pointer-events: none;
    }}
    .ms-login-medallion-wrap {{
        display: flex;
        justify-content: center;
        margin-top: -62px;
        margin-bottom: 6px;
        position: relative;
        z-index: 5;
    }}
    .ms-login-medallion {{
        width: 68px;
        height: 68px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: radial-gradient(circle at 50% 45%, #113063 0%, #06132C 76%);
        border: 1.5px solid rgba(103, 214, 255, 0.85);
        box-shadow:
            0 0 0 6px rgba(56, 189, 248, 0.13),
            0 0 0 13px rgba(14, 165, 233, 0.06),
            0 0 34px rgba(56, 189, 248, 0.65),
            inset 0 0 16px rgba(56, 189, 248, 0.28);
    }}
    .ms-login-medallion svg {{
        width: 40px;
        height: 40px;
        filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.85));
    }}
    .ms-login-header Log scale
        width: 100% !important;
        text-align: center !important;
        margin-bottom: 0.24rem;
    }}
    .ms-login-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 7px;
        font-family: 'Inter', sans-serif;
        font-size: 0.76rem;
        font-weight: 500;
        letter-spacing: 0.015em;
        padding: 4px 14px;
        border-radius: 999px;
        color: #E0F2FE;
        background: rgba(8, 22, 52, 0.80);
        border: 1px solid rgba(56, 189, 248, 0.42);
        box-shadow: inset 0 1px 0 rgba(186, 230, 253, 0.18), 0 0 12px rgba(14, 165, 233, 0.16);
        margin: 0 auto 0.34rem auto;
    }}
    .ms-login-title {{
        display: block !important;
        width: 100% !important;
        text-align: center !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.82rem;
        font-weight: 700;
        line-height: 1.08;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        padding: 0 !important;
        margin: 0.06rem auto 0.26rem auto !important;
    }}
    .ms-login-line2 {{
        display: inline-block;
        white-space: nowrap;
    }}
    .ms-portal-accent {{
        color: #25C2FF;
        text-shadow: 0 0 22px rgba(56, 189, 248, 0.55);
    }}
    .ms-login-sub {{
        display: block;
        width: 100%;
        text-align: center !important;
        color: #CBD5E1;
        font-size: 0.90rem;
        line-height: 1.32;
        margin: 0 auto 0.12rem auto;
    }}
    .st-key-login_card [data-testid="stTextInput"] [data-testid="stTextInputRootElement"],
    .st-key-login_card [data-testid="stTextInput"] [data-baseweb="input"] {{
        background: rgba(11, 23, 50, 0.58) !important;
        border: 1px solid rgba(148, 197, 255, 0.32) !important;
        border-radius: 11px !important;
        min-height: 43px !important;
        height: 43px !important;
        padding-left: 0.55rem !important;
        padding-right: 0.42rem !important;
        transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
    }}
    .st-key-login_card [data-testid="stTextInput"] [data-testid="stTextInputRootElement"]:focus-within,
    .st-key-login_card [data-testid="stTextInput"] [data-baseweb="input"]:focus-within {{
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.22), 0 0 16px rgba(56, 189, 248, 0.20) !important;
    }}
    .st-key-login_card [data-testid="stTextInput"] [data-baseweb="base-input"],
    .st-key-login_card [data-testid="stTextInput"] [data-testid="stTextInputField"],
    .st-key-login_card [data-testid="stTextInput"] input {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #F8FAFC !important;
        font-size: 0.92rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }}
    .st-key-login_card [data-testid="stTextInput"] input::placeholder {{
        color: #94A3B8 !important;
        opacity: 1 !important;
    }}
    .st-key-login_card [data-testid="stTextInput"] [data-testid="stTextInputIcon"],
    .st-key-login_card [data-testid="stTextInput"] svg,
    .st-key-login_card [data-testid="stTextInput"] span[data-testid="stIconMaterial"] {{
        color: #94A3B8 !important;
    }}
    .st-key-login_card [data-testid="stTextInput"] [data-testid="stTextInputRootElement"]:focus-within svg,
    .st-key-login_card [data-testid="stTextInput"] [data-testid="stTextInputRootElement"]:focus-within span[data-testid="stIconMaterial"],
    .st-key-login_card [data-testid="stTextInput"] [data-baseweb="input"]:focus-within svg {{
        color: #38BDF8 !important;
    }}
    .st-key-login_card [data-testid="stHorizontalBlock"] {{
        align-items: center !important;
        gap: 0.25rem !important;
        margin: 0.12rem 0 !important;
        padding: 0 !important;
    }}
    .st-key-login_card [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:last-child [data-testid="stVerticalBlock"] {{
        align-items: flex-end !important;
    }}
    .st-key-login_card [data-testid="stCheckbox"] {{
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }}
    .st-key-login_card [data-testid="stCheckbox"] label {{
        gap: 8px !important;
        padding: 0 !important;
        min-height: 22px !important;
    }}
    .st-key-login_card [data-testid="stCheckbox"] label p {{
        font-size: 0.84rem !important;
        color: #CBD5E1 !important;
    }}
    .st-key-forgot_pwd_btn {{
        width: auto !important;
        margin-left: auto !important;
    }}
    .st-key-forgot_pwd_btn button {{
        min-height: 0 !important;
        height: auto !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        color: #38BDF8 !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        text-decoration: underline !important;
        text-underline-offset: 2px !important;
    }}
    .st-key-forgot_pwd_btn button:hover {{
        color: #7DD3FC !important;
    }}
    .st-key-login_card .st-key-login_submit_btn button,
    .st-key-login_card .st-key-login_submit_btn button[kind="primary"] {{
        background: linear-gradient(90deg, #2AE2FF 0%, #0EA5E9 52%, #1C7BFF 100%) !important;
        border: 1px solid rgba(186, 230, 253, 0.68) !important;
        border-radius: 11px !important;
        min-height: 44px !important;
        height: 44px !important;
        font-size: 0.96rem !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 26px rgba(14, 165, 233, 0.58), 0 0 28px rgba(56, 189, 248, 0.42), inset 0 1px 0 rgba(255, 255, 255, 0.45) !important;
    }}
    .st-key-login_card .st-key-login_submit_btn button:hover,
    .st-key-login_card .st-key-login_submit_btn button[kind="primary"]:hover {{
        transform: translateY(-1px);
        box-shadow: 0 9px 30px rgba(56, 189, 248, 0.72), 0 0 32px rgba(28, 123, 255, 0.52), inset 0 1px 0 rgba(255, 255, 255, 0.52) !important;
    }}
    .st-key-login_card .st-key-guest_access_btn button,
    .st-key-login_card .st-key-guest_access_btn button[kind="secondary"] {{
        background: rgba(8, 20, 46, 0.58) !important;
        border: 1px solid rgba(96, 165, 250, 0.38) !important;
        border-radius: 11px !important;
        min-height: 42px !important;
        height: 42px !important;
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        color: #F1F5F9 !important;
    }}
    .st-key-login_card .st-key-guest_access_btn span[data-testid="stIconMaterial"] {{
        color: #7DD3FC !important;
    }}
    .st-key-login_card .st-key-guest_access_btn button:hover,
    .st-key-login_card .st-key-guest_access_btn button[kind="secondary"]:hover {{
        border-color: rgba(56, 189, 248, 0.72) !important;
        background: rgba(14, 29, 62, 0.82) !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.25) !important;
    }}
    .ms-login-divider {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 0.14rem 0;
        color: #94A3B8;
        font-family: 'Inter', sans-serif;
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.08em;
    }}
    .ms-login-divider::before,
    .ms-login-divider::after {{
        content: "";
        flex: 1;
        height: 1px;
        background: rgba(148, 163, 184, 0.22);
    }}
    .ms-demo-box {{
        margin-top: 0.20rem;
        margin-bottom: 0;
        padding: 0.65rem 0.92rem;
        border-radius: 12px;
        background: rgba(7, 18, 42, 0.64);
        border: 1px solid rgba(56, 189, 248, 0.34);
    }}
    .ms-demo-head {{
        display: flex;
        align-items: center;
        gap: 7px;
        font-size: 0.78rem;
        margin-bottom: 0.34rem;
    }}
    .ms-demo-head-title {{
        color: #38BDF8;
        font-weight: 600;
    }}
    .ms-demo-head-note {{
        color: #94A3B8;
        font-size: 0.72rem;
    }}
    .ms-demo-grid {{
        display: grid;
        grid-template-columns: 1.08fr 0.92fr;
        gap: 12px;
        align-items: center;
    }}
    .ms-demo-creds {{
        display: flex;
        flex-direction: column;
        gap: 5px;
        padding-right: 12px;
        border-right: 1px solid rgba(148, 163, 184, 0.24);
    }}
    .ms-demo-item {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        font-size: 0.79rem;
        color: #CBD5E1;
    }}
    .ms-demo-label {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }}
    .ms-demo-code {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        font-weight: 600;
        color: #F8FAFC;
        background: rgba(10, 26, 58, 0.88);
        border: 1px solid rgba(56, 189, 248, 0.36);
        padding: 2px 10px;
        border-radius: 999px;
        min-width: 66px;
        text-align: center;
    }}
    .ms-demo-right {{
        display: flex;
        align-items: flex-start;
        gap: 7px;
        font-size: 0.78rem;
        line-height: 1.32;
        color: #CBD5E1;
    }}
    .ms-demo-right strong {{
        color: #F8FAFC;
        font-weight: 600;
    }}
    .ms-login-foot {{
        width: 100%;
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: rgba(148, 163, 184, 0.90);
        letter-spacing: 0.015em;
        text-shadow: 0 1px 8px rgba(0, 0, 0, 0.90);
    }}
    @media (max-height: 700px) {{
        .st-key-login_card {{
            padding: 1.2rem 1.55rem 1.1rem !important;
        }}
        .st-key-login_card > div,
        .st-key-login_card [data-testid="stVerticalBlock"] {{
            gap: 0.26rem !important;
        }}
        .ms-login-medallion-wrap {{
            margin-top: -50px !important;
            margin-bottom: 4px !important;
        }}
        .ms-login-medallion {{
            width: 56px !important;
            height: 56px !important;
        }}
        .ms-login-medallion svg {{
            width: 32px !important;
            height: 32px !important;
        }}
        .ms-login-title {{
            font-size: 1.58rem !important;
            margin: 0.04rem auto 0.18rem auto !important;
        }}
        .ms-login-sub {{
            font-size: 0.82rem !important;
        }}
        .st-key-login_card [data-testid="stTextInput"] [data-testid="stTextInputRootElement"],
        .st-key-login_card [data-testid="stTextInput"] [data-baseweb="input"],
        .st-key-login_card .st-key-login_submit_btn button {{
            min-height: 37px !important;
            height: 37px !important;
        }}
        .st-key-login_card .st-key-guest_access_btn button {{
            min-height: 35px !important;
            height: 35px !important;
        }}
        .ms-demo-box {{
            padding: 0.48rem 0.76rem !important;
        }}
    }}
    @media (max-height: 580px) {{
        .st-key-login_card {{
            padding: 0.95rem 1.3rem 0.85rem !important;
        }}
        .st-key-login_card > div,
        .st-key-login_card [data-testid="stVerticalBlock"] {{
            gap: 0.18rem !important;
        }}
        .ms-login-medallion-wrap {{
            margin-top: -40px !important;
            margin-bottom: 2px !important;
        }}
        .ms-login-medallion {{
            width: 46px !important;
            height: 46px !important;
        }}
        .ms-login-medallion svg {{
            width: 26px !important;
            height: 26px !important;
        }}
        .ms-login-title {{
            font-size: 1.36rem !important;
        }}
        .st-key-login_card [data-testid="stTextInput"] [data-testid="stTextInputRootElement"],
        .st-key-login_card [data-testid="stTextInput"] [data-baseweb="input"],
        .st-key-login_card .st-key-login_submit_btn button {{
            min-height: 32px !important;
            height: 32px !important;
        }}
        .st-key-login_card .st-key-guest_access_btn button {{
            min-height: 31px !important;
            height: 31px !important;
        }}
        .ms-demo-box {{
            padding: 0.34rem 0.62rem !important;
        }}
    }}

    {dash_bg_css}
    {login_only_css}
    </style>
    """
    st.html(css)


def render_login_page() -> None:
    """Render the full-screen dark-tech login card with demo credentials and guest access."""
    _, using_fallback = get_auth_credentials()

    if "login_username" not in st.session_state and st.session_state.get("remember_me"):
        st.session_state["login_username"] = st.session_state.get("remembered_user", "")

    dark_login = bool(st.session_state.get("login_dark_mode", True))
    logo_file = LOGIN_CORNER_LOGO_FILE if (dark_login or not LOGIN_CORNER_LOGO_LIGHT_FILE.is_file()) else LOGIN_CORNER_LOGO_LIGHT_FILE
    corner_mtime = logo_file.stat().st_mtime_ns if logo_file.is_file() else 0
    corner_b64 = load_asset_b64(str(logo_file), corner_mtime)
    if corner_b64:
        corner_markup = (
            '<div class="ms-login-corner-brand">'
            f'<img class="ms-login-corner-img" src="data:image/png;base64,{corner_b64}" '
            'alt="Hospital Analytics — Powered by AI">'
            '</div>'
        )
    else:
        corner_markup = (
            f'<div class="ms-login-corner-brand">'
            f'<div class="ms-login-corner-icon">{LOGIN_CORNER_LOGO_SVG}</div>'
            f'<div><div class="ms-login-corner-title">Hospital Analytics</div>'
            f'<div class="ms-login-corner-sub">Powered by AI</div></div></div>'
        )
    render_raw_html(corner_markup)

    with st.container(key="login_dark_pill"):
        st.toggle("Dark Mode", key="login_dark_mode")

    with st.container(border=True, key="login_card"):
        render_raw_html(
            f"""
            <span class="ms-login-anchor" style="display:none;"></span>
            <div class="ms-login-medallion-wrap">
                <div class="ms-login-medallion">{LOGIN_MEDALLION_SVG}</div>
            </div>
            <div class="ms-login-header">
                <div class="ms-login-badge">
                    {LOGIN_BADGE_WAVE_SVG}
                    <span>MediScope Health Analytics</span>
                </div>
                <div class="ms-login-title" role="heading" aria-level="1">Hospital<br><span class="ms-login-line2">Discharges <span class="ms-portal-accent">Portal</span></span></div>
                <p class="ms-login-sub">Unlock insights. Improve care.</p>
            </div>
            """
        )

        username = login_text_input(
            "Username",
            placeholder="Enter username (demo)",
            key="login_username",
            icon=":material/person:",
        )
        password = login_text_input(
            "Password",
            type="password",
            placeholder="Enter password",
            key="login_password",
            icon=":material/lock:",
        )

        try:
            rem_col, fp_col = st.columns([1.1, 1], vertical_alignment="center")
        except TypeError:
            rem_col, fp_col = st.columns([1.1, 1])

        with rem_col:
            st.checkbox(
                "Remember me",
                value=bool(st.session_state.get("remember_me", False)),
                key="login_remember",
            )
        with fp_col:
            forgot_clicked = tertiary_button("Forgot password?", key="forgot_pwd_btn")

        if forgot_clicked:
            st.toast(
                f"Use demo credentials ({DEMO_USER} / {DEMO_PASS}) or click Continue as Guest.",
                icon=":material/info:",
            )

        submit = stretch_button(
            "Sign In to Dashboard →",
            key="login_submit_btn",
            type="primary",
        )

        if submit:
            if verify_credentials(username, password):
                sign_in_user(username.strip(), "Operator")
            else:
                st.error("Invalid username or password. Use the demo credentials below or continue as guest.")

        render_raw_html('<div class="ms-login-divider">OR</div>')

        if stretch_button(
            "Continue as Guest",
            key="guest_access_btn",
            type="secondary",
            icon=":material/shield:",
        ):
            sign_in_user("Guest Reviewer", "Guest")

        demo_note = "(for testing)" if using_fallback else "(secrets active)"
        render_raw_html(
            f"""
            <div class="ms-demo-box">
                <div class="ms-demo-head">
                    {LOGIN_INFO_SVG}
                    <span class="ms-demo-head-title">Demo Credentials</span>
                    <span class="ms-demo-head-note">{html.escape(demo_note)}</span>
                </div>
                <div class="ms-demo-grid">
                    <div class="ms-demo-creds">
                        <div class="ms-demo-item">
                            <span class="ms-demo-label">{LOGIN_KEY_SVG}<span>Username:</span></span>
                            <code class="ms-demo-code">{DEMO_USER}</code>
                        </div>
                        <div class="ms-demo-item">
                            <span class="ms-demo-label">{LOGIN_KEY_SVG}<span>Password:</span></span>
                            <code class="ms-demo-code">{DEMO_PASS}</code>
                        </div>
                    </div>
                    <div class="ms-demo-right">
                        <span>{LOGIN_SPARKLE_SVG}</span>
                        <span><strong>Explore</strong> the dashboard with demo access!</span>
                    </div>
                </div>
            </div>
            """
        )

    render_raw_html('<div class="ms-login-foot">MediScope Health Analytics • v1.0.0</div>')


# =============================================================================
# 3. DATA LOADING AND CLEANING  (assignment Step 2)
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
# 4. HELPER FUNCTIONS
# =============================================================================
def usd(x) -> str:
    """Short dollar format: 1,234,567 -> '$1.2M'."""
    if pd.isna(x):
        return "n/a"
    for limit, suffix in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(x) >= limit:
            return f"${x / limit:.1f}{suffix}"
    return f"${x:,.0f}"


def login_text_input(label: str, *, key: str, placeholder: str, type: str = "default", icon: str | None = None) -> str:
    """Render a collapsed-label text input with an inline icon on newer Streamlit versions."""
    kwargs = {"type": type, "placeholder": placeholder, "key": key, "label_visibility": "collapsed"}
    try:
        return st.text_input(label, icon=icon, **kwargs) or ""
    except TypeError:
        return st.text_input(label, **kwargs) or ""


def tertiary_button(label: str, key: str) -> bool:
    """Render a plain link-style button safely on old and new Streamlit versions."""
    try:
        return st.button(label, key=key, type="tertiary")
    except TypeError:
        return st.button(label, key=key, type="secondary")


def stretch_button(label: str, key: str, type: str = "secondary", container=None, icon: str | None = None) -> bool:
    """Render a full-width st.button safely on old and new Streamlit versions."""
    target = container or st
    kwargs = {"key": key, "type": type}
    if icon:
        kwargs["icon"] = icon
    try:
        return target.button(label, width="stretch", **kwargs)
    except TypeError:
        kwargs.pop("icon", None)
        try:
            return target.button(label, width="stretch", **kwargs)
        except TypeError:
            return target.button(label, use_container_width=True, **kwargs)


def stretch_form_submit_button(label: str, type: str = "primary") -> bool:
    """Render a full-width st.form_submit_button safely on old and new Streamlit versions."""
    try:
        return st.form_submit_button(label, type=type, width="stretch")
    except TypeError:
        return st.form_submit_button(label, type=type, use_container_width=True)


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


def _kpi_wave_svg(accent: str, idx: int) -> str:
    """Decorative subtle telemetry wave curve rendered inside the bottom corner of a KPI card."""
    curves = (
        "M0 36 C28 34, 42 22, 68 24 C92 26, 112 10, 140 6",
        "M0 34 C30 28, 48 34, 72 20 C96 8, 116 24, 140 12",
        "M0 24 C34 25, 58 21, 86 24 C108 26, 124 20, 140 21",
        "M0 38 C26 30, 50 32, 78 20 C102 10, 120 18, 140 8",
        "M0 36 C32 34, 52 24, 78 26 C104 28, 120 14, 140 10",
    )
    path = curves[idx % len(curves)]
    grad_id = f"kpiWaveGrad{idx}"
    return (
        f'<svg class="ms-kpi-wave" viewBox="0 0 140 44" preserveAspectRatio="none" fill="none" aria-hidden="true">'
        f'<defs><linearGradient id="{grad_id}" x1="0" y1="0" x2="0" y2="44" gradientUnits="userSpaceOnUse">'
        f'<stop offset="0%" stop-color="{accent}" stop-opacity="0.45"/>'
        f'<stop offset="100%" stop-color="{accent}" stop-opacity="0"/></linearGradient></defs>'
        f'<path d="{path} L140 44 L0 44 Z" fill="url(#{grad_id})"/>'
        f'<path d="{path}" stroke="{accent}" stroke-width="1.8" stroke-linecap="round"/>'
        f'</svg>'
    )


def render_kpi_cards(cards: list[dict[str, str]]) -> None:
    """Render a responsive row of glassmorphic KPI telemetry cards."""
    cols = st.columns(len(cards))
    for idx, (col, card) in enumerate(zip(cols, cards)):
        accent = card.get("accent", BLUE)
        sub_html = f'<div class="ms-kpi-sub">{html.escape(card["sub"])}</div>' if card.get("sub") else ""
        icon_paths = KPI_ICON_PATHS.get(card["label"].lower(), KPI_ICON_PATHS["discharges"])
        icon_html = _hud_svg(icon_paths, size=13, color=accent)
        wave_html = _kpi_wave_svg(accent, idx)
        with col:
            render_raw_html(
                f"""
                <div class="ms-kpi" style="--kpi-accent: {accent};">
                    <div class="ms-kpi-bar"></div>
                    <div class="ms-kpi-head">
                        <div class="ms-kpi-icon">{icon_html}</div>
                        <div class="ms-kpi-label">{html.escape(card["label"])}</div>
                    </div>
                    <div class="ms-kpi-value">{html.escape(card["value"])}</div>
                    {sub_html}
                    {wave_html}
                </div>
                """
            )


def render_panel_header(tag: str, title: str, caption: str) -> None:
    """Render a uniform card header with a monospace tag, title, and caption."""
    tag_low = tag.lower()
    icon_key = next((k for k in PANEL_TAG_ICONS if k in tag_low), "bar")
    tag_icon = _hud_svg(PANEL_TAG_ICONS[icon_key], size=12, color="#38BDF8")
    render_raw_html(f'<div class="ms-panel-tag">{tag_icon}<span>{html.escape(tag)}</span></div>')
    st.subheader(title)
    st.caption(caption)


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
    """Apply the cohesive dark-tech Plotly layout to every chart."""
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=14, t=50 if title else 22, b=14),
        title=dict(text=title, font=dict(size=15, color="#F8FAFC")) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(4, 13, 34, 0.56)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#E2E8F0", size=12),
        hoverlabel=dict(
            bgcolor="rgba(6, 19, 48, 0.96)",
            bordercolor="#38BDF8",
            font=dict(family="Inter, sans-serif", color="#F8FAFC", size=12),
        ),
        legend=dict(font=dict(color="#CBD5E1", size=11)),
    )
    fig.update_xaxes(
        gridcolor="rgba(56, 189, 248, 0.09)",
        zerolinecolor="rgba(56, 189, 248, 0.16)",
        linecolor="rgba(56, 189, 248, 0.18)",
        tickfont=dict(color="#CBD5E1", size=11),
        title_font=dict(color="#94A3B8", size=12),
    )
    fig.update_yaxes(
        gridcolor="rgba(56, 189, 248, 0.09)",
        zerolinecolor="rgba(56, 189, 248, 0.16)",
        linecolor="rgba(56, 189, 248, 0.18)",
        tickfont=dict(color="#CBD5E1", size=11),
        title_font=dict(color="#94A3B8", size=12),
    )
    return fig


def counts(series: pd.Series) -> pd.Series:
    """value_counts that ignores unused categories left over after filtering."""
    vc = series.value_counts()
    return vc[vc > 0]


def geo_only(f: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose hospital / county is redacted (used by county and facility views)."""
    return f[(f["County"] != "Not Disclosed") & (f["Facility"] != "Not Disclosed")]


def shorten(text: str, n: int = 42) -> str:
    """Truncate long labels cleanly with an ellipsis."""
    return text if len(text) <= n else text[: n - 1] + "…"


# =============================================================================
# 5. CHART FUNCTIONS  (assignment Step 4 - five required visuals)
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
    fig.update_traces(
        marker=dict(color=BLUE, opacity=0.92, line=dict(color="rgba(186, 230, 253, 0.45)", width=1)),
        hovertemplate="<b>%{customdata[0]}</b><br>Body system: %{customdata[1]}<br>"
                      "Average stay: %{x:.1f} days<br>Median stay: %{customdata[2]:.0f} days<br>"
                      "Discharges: %{customdata[3]:,}<extra></extra>",
    )
    fig.update_layout(xaxis_title="Average stay (days)", yaxis_title=None)
    return tidy(fig, height=max(380, 24 * top_n + 110))


def chart_charges_box(f, log_scale, show_points):
    """CHART 2 - Box plot: total charges segmented by severity."""
    fig = px.box(f, x="Severity", y="Total Charges", color="Severity", log_y=log_scale,
                 category_orders={"Severity": SEVERITY_ORDER}, color_discrete_map=SEVERITY_COLORS,
                 points="outliers" if show_points else False)
    fig.update_traces(boxmean=True, line=dict(width=1.8))  # dashed line = mean
    fig.update_layout(showlegend=False, xaxis_title="Severity of illness (APR)", yaxis_title="Total charges (US$)")
    fig = tidy(fig, height=470)
    if log_scale:
        fig.update_yaxes(**MONEY_TICKS)
    else:
        fig.update_yaxes(tickprefix="$")
    return fig


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
    fig.update_traces(
        xgap=2,
        ygap=2,
        hovertemplate="%{y}<br>%{x}<br>Average stay: %{z:.1f} days<extra></extra>",
    )
    fig.update_layout(coloraxis_colorbar=dict(tickfont=dict(color="#CBD5E1"), title_font=dict(color="#CBD5E1")))
    fig = tidy(fig, height=max(380, 26 * len(piv) + 140))
    fig.update_xaxes(side="top", tickangle=-35 if label_cols else 0, showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


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
    fig.update_traces(
        textinfo="percent",
        textposition="inside",
        sort=False,
        insidetextorientation="horizontal",
        textfont=dict(color="#070B14", size=12),
        marker=dict(line=dict(color="#0B1220", width=2)),
        hovertemplate="<b>%{label}</b><br>%{value:,} discharges (%{percent})<extra></extra>",
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.12))
    return tidy(fig, height=400)


def payer_charges_table(f, min_cases=30):
    """Average and median total charges per primary payer (companion to chart 4)."""
    g = (f.groupby("Primary Payer", observed=True)["Total Charges"]
         .agg(**{"Avg charges": "mean", "Median charges": "median", "Discharges": "size"}).reset_index())
    return g[g["Discharges"] >= min_cases].sort_values("Avg charges")


def chart_payer_bar(g):
    """Companion to chart 4: what each payer's patients are billed on average."""
    fig = px.bar(g, x="Avg charges", y="Primary Payer", orientation="h", color="Primary Payer",
                 color_discrete_map=PAYER_COLORS, custom_data=["Median charges", "Discharges"])
    fig.update_traces(
        marker=dict(opacity=0.92, line=dict(color="rgba(255, 255, 255, 0.18)", width=1)),
        hovertemplate="<b>%{y}</b><br>Average charges: $%{x:,.0f}<br>Median charges: $%{customdata[0]:,.0f}"
                      "<br>Discharges: %{customdata[1]:,}<extra></extra>",
    )
    fig.update_layout(showlegend=False, xaxis_title="Average total charges (US$)", yaxis_title=None, xaxis_tickprefix="$")
    return tidy(fig, height=400)


def chart_stay_hist(f, bin_width, log_y):
    """CHART 5 - Histogram: overall distribution of length of stay."""
    fig = px.histogram(f, x="Stay Days", color_discrete_sequence=[BLUE], log_y=log_y)
    fig.update_traces(
        xbins=dict(start=0.5, size=bin_width),
        marker=dict(color=BLUE, opacity=0.88, line=dict(color="#070B14", width=0.8)),
        hovertemplate="Stay: %{x} days<br>Discharges: %{y:,}<extra></extra>",
    )
    mean, median = f["Stay Days"].mean(), f["Stay Days"].median()
    fig.add_vline(
        x=mean, line_dash="dash", line_color=ORANGE, line_width=2,
        annotation_text=f"Mean {mean:.1f} d", annotation_position="top right",
        annotation_bgcolor="rgba(11, 18, 34, 0.88)", annotation_font_color=ORANGE,
    )
    fig.add_vline(
        x=median, line_dash="dot", line_color=AQUA, line_width=2,
        annotation_text=f"Median {median:.0f} d", annotation_position="top left",
        annotation_bgcolor="rgba(11, 18, 34, 0.88)", annotation_font_color=AQUA,
    )
    fig.update_layout(xaxis_title="Length of stay (days)", yaxis_title="Number of discharges", bargap=0.05)
    return tidy(fig, height=420)


# =============================================================================
# 6. EDA AND INSIGHT FUNCTIONS  (assignment Step 3)
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

    # Q3 - severity and cost (pure-pandas rank correlation avoids requiring scipy)
    g = f.groupby("Severity", observed=True)["Total Charges"].median().dropna()
    if len(g) >= 2:
        lo, hi = g.index[0], g.index[-1]
        corr = f["Stay Days"].rank().corr(f["Total Charges"].rank())
        corr_text = f"Spearman rank correlation ρ = {corr:.2f}" if pd.notna(corr) else "Spearman rank correlation unavailable"
        out.append(("Q3 · How does patient severity affect total cost?",
                    f"Median charges rise from {usd(g.iloc[0])} for **{lo}** cases to {usd(g.iloc[-1])} for **{hi}** "
                    f"cases - about **{g.iloc[-1] / g.iloc[0]:.1f}x** higher. Length of stay and charges move together "
                    f"({corr_text})."))

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
# 7. PAGE LAYOUT: LOGIN GATE, SIDEBAR FILTERS, HERO, KPIS & FOUR TABS
# =============================================================================
def _sidebar_group_html(title: str) -> str:
    """Return sidebar section heading markup paired with its matching HUD icon."""
    icon = _hud_svg(SIDEBAR_GROUP_ICONS.get(title, ""), size=14, color="#38BDF8")
    return f'<div class="ms-filter-group">{icon}<span>{html.escape(title)}</span></div>'


def render_sidebar(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Render the dark-tech filter sidebar, session badge, and logout button."""
    auth_user = st.session_state.get("auth_user") or "Operator"
    auth_mode = st.session_state.get("auth_mode") or "Operator"

    with st.sidebar:
        render_raw_html(
            f"""
            <div class="ms-sidebar-brand">
                {MEDISCOPE_SIDEBAR_LOGO_SVG}
                <div>
                    <div class="ms-sidebar-brand-title">MediScope</div>
                    <div class="ms-sidebar-brand-sub">Telemetry Console</div>
                </div>
            </div>
            <div class="ms-user-pill">
                <div class="ms-user-name">
                    <span class="ms-avatar-ring">{USER_AVATAR_SVG}</span>
                    <span class="ms-status-dot"></span>
                    <span>{html.escape(auth_user)}</span>
                </div>
                <span class="ms-role-tag">{html.escape(auth_mode)}</span>
            </div>
            """
        )

        st.caption("Leave a filter empty to include everything.")

        render_raw_html(_sidebar_group_html("Geography"))
        regions = st.multiselect("Region", sorted(df["Region"].cat.categories), placeholder="All regions")
        county_pool = df if not regions else df[df["Region"].isin(regions)]
        counties = st.multiselect("County", sorted(counts(county_pool["County"]).index), placeholder="All counties")

        render_raw_html(_sidebar_group_html("Patient Profile"))
        ages = st.multiselect("Age group", AGE_ORDER, placeholder="All ages")
        genders = st.multiselect("Gender", sorted(df["Gender"].cat.categories), placeholder="All")

        render_raw_html(_sidebar_group_html("Clinical & Payer"))
        severities = st.multiselect("Severity of illness", SEVERITY_ORDER, placeholder="All severities")
        payers = st.multiselect("Primary payer", sorted(df["Primary Payer"].cat.categories), placeholder="All payers")
        admissions = st.multiselect("Admission type", sorted(df["Admission Type"].cat.categories), placeholder="All types")

        render_raw_html(_sidebar_group_html("Stay & Outliers"))
        stay_max = int(df["Stay Days"].max())
        stay_range = st.slider("Length of stay (days)", 1, stay_max, (1, stay_max))
        no_outliers = st.checkbox(
            "Exclude statistical outliers",
            value=True,
            help="Removes the longest 1% of stays and extreme charges (see the Data Cleaning tab). "
                 "Untick to see every record.",
        )

        filters = {
            "Region": regions, "County": counties, "Age Group": ages, "Gender": genders,
            "Severity": severities, "Primary Payer": payers, "Admission Type": admissions,
            "stay": stay_range, "no_outliers": no_outliers,
        }
        f = apply_filters(df, filters)

        st.divider()
        render_raw_html(
            f"""
            <div class="ms-sidebar-telemetry">
                <div class="ms-sidebar-telemetry-label">Discharges Selected</div>
                <div class="ms-sidebar-telemetry-val">{len(f):,}</div>
                <div class="ms-sidebar-telemetry-sub">{len(f) / len(df):.0%} of cleaned sample</div>
            </div>
            """
        )
        st.caption(
            f"Data: NY SPARCS 2021 inpatient discharges - a random 5% sample ({len(df):,} cleaned rows) of the "
            f"{FULL_FILE_ROWS:,}-row original."
        )
        if stretch_button("Log out", key="logout_btn", type="secondary", container=st.sidebar, icon=":material/logout:"):
            logout_user()
    return filters, f


def render_hero_header(f: pd.DataFrame, df: pd.DataFrame, no_outliers: bool) -> None:
    """Render the modern tech hero banner with active cohort telemetry."""
    auth_user = st.session_state.get("auth_user") or "Operator"
    outlier_chip = "Outliers Excluded" if no_outliers else "All Records"
    render_raw_html(
        f"""
        <div class="ms-hero">
            <div class="ms-hero-top">
                <div class="ms-hero-badges">
                    <span class="ms-chip">{CHIP_TERMINAL_SVG}<span>MEDISCOPE // NY SPARCS 2021</span></span>
                    <span class="ms-chip ms-chip-teal"><span class="ms-status-dot"></span>{len(f):,} / {len(df):,} ACTIVE</span>
                    <span class="ms-chip">{CHIP_SHIELD_SVG}<span>{outlier_chip}</span></span>
                </div>
                <div class="ms-hero-badges">
                    <span class="ms-chip">{USER_AVATAR_SVG}<span>SESSION: {html.escape(auth_user.upper())}</span></span>
                </div>
            </div>
            <div class="ms-hero-title-row">
                {MEDISCOPE_HERO_PULSE_SVG}
                <h1 class="ms-hero-title">Hospital Inpatient Discharges Dashboard</h1>
            </div>
            <p class="ms-hero-sub">
                <strong>For hospital administrators, clinicians and operations managers</strong> - see which diagnoses keep
                patients longest, what drives the bill, and where length of stay differs by county, facility and payer.
            </p>
        </div>
        """
    )


def render_dashboard_tab(f: pd.DataFrame) -> None:
    """Render Tab 1 with the five required interactive Plotly charts and data expanders."""
    left, right = st.columns(2)

    # ---- Chart 1: bar chart -------------------------------------------------
    with left, st.container(border=True, key="card_chart_dx"):
        render_panel_header(
            "CHART 01 · BAR",
            "1 · Which diagnoses keep patients longest?",
            "Average length of stay per diagnosis code (CCSR). Hover for the median and number of cases.",
        )
        c1, c2 = st.columns(2)
        top_n = c1.slider("Diagnoses shown", 5, 30, 12, key="top_n")
        min_cases = c2.slider(
            "Minimum cases per diagnosis", 5, 200, MIN_DX_CASES, step=5, key="min_cases",
            help="Hides rare diagnoses whose average would be unreliable.",
        )
        dx_table = dx_stay_table(f, min_cases)
        if dx_table.empty:
            st.info("No diagnosis has that many cases under the current filters. Lower the minimum.")
        else:
            show_chart(chart_dx_bar(dx_table, top_n), "chart_dx")
            with st.expander("View chart data"):
                show_table(dx_table.nlargest(top_n, "Avg stay").round(2))

    # ---- Chart 2: box plot --------------------------------------------------
    with right, st.container(border=True, key="card_chart_box"):
        render_panel_header(
            "CHART 02 · BOX PLOT",
            "2 · How does severity change the bill?",
            "Total charges for each severity level. Box = middle 50% of bills, solid line = median, dashed line = mean.",
        )
        c1, c2 = st.columns(2)
        log_scale = c1.checkbox(
            "Log scale (recommended)", value=True, key="box_log",
            help="Charges are heavily skewed; a log axis keeps every box readable.",
        )
        show_pts = c2.checkbox("Show outlier points", value=False, key="box_pts")
        show_chart(chart_charges_box(f, log_scale, show_pts), "chart_box")
        with st.expander("View chart data"):
            sev = (f.groupby("Severity", observed=True)["Total Charges"]
                   .describe(percentiles=[0.25, 0.5, 0.75])[["count", "mean", "25%", "50%", "75%", "max"]].round(0))
            show_table(sev.reset_index())

    # ---- Chart 3: heatmap ---------------------------------------------------
    with st.container(border=True, key="card_chart_heat"):
        render_panel_header(
            "CHART 03 · HEATMAP",
            "3 · Where are stays longest? (facility and county)",
            "Average length of stay in days. Brighter / warmer = longer stays. Each hospital belongs to one county, so "
            "the facility view fills one block per county; the county × severity view is a full grid. "
            f"Cells built from fewer than {MIN_CELL} discharges are left blank.",
        )
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
    with st.container(border=True, key="card_chart_pie"):
        render_panel_header(
            "CHART 04 · PIE & PAYER BILLING",
            "4 · Who pays? Distribution of patients by payment type",
            "Left: share of discharges by primary payer (payers under 3% are grouped as 'Other payers'). "
            "Right: what each payer's patients are billed on average.",
        )
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
    with st.container(border=True, key="card_chart_hist"):
        render_panel_header(
            "CHART 05 · HISTOGRAM",
            "5 · How long do patients stay overall?",
            "Number of discharges at each length of stay. The right-hand tail is why the mean (orange) "
            "sits above the median (teal).",
        )
        c1, c2 = st.columns(2)
        bin_width = c1.slider("Bin width (days)", 1, 10, 1, key="bin_w")
        log_y = c2.checkbox("Log scale on the count axis", value=False, key="hist_log")
        show_chart(chart_stay_hist(f, bin_width, log_y), "chart_hist")
        with st.expander("View chart data"):
            show_table(f["Stay Band"].value_counts().rename_axis("Stay band").reset_index(name="Discharges")
                       .sort_values("Stay band"))


def render_eda_tab(f: pd.DataFrame) -> None:
    """Render Tab 2 with research answers, statistical summary, pivots, and scatter analysis."""
    st.header("Exploratory data analysis")
    st.caption("Everything on this tab recalculates when you change the sidebar filters.")

    st.subheader("Answers to the five research questions")
    with st.container(border=True, key="card_eda_answers"):
        for question, answer in research_answers(f):
            st.markdown(f"**{question}**")
            st.markdown(answer.replace("$", r"\$"))

    with st.container(border=True, key="card_eda_stats"):
        st.subheader("Statistical summary")
        stats = describe_numbers(f)
        show_table(stats.style.format({c: "{:,.2f}" for c in stats.columns if c != "Measure"}))
        st.caption("Skewness well above 0 means a long right tail (a few very expensive or very long stays), which is why "
                   "medians describe a 'typical' patient better than means. The 95% CI is mean ± 1.96 × SD / √n.")

    with st.container(border=True, key="card_eda_dx_sev"):
        st.subheader("Diagnosis category × severity: average stay (groupby + pivot_table)")
        st.caption(f"Cells built from fewer than {MIN_CELL} discharges are left blank.")
        piv = mean_pivot(f, "Dx Category", "Severity")
        piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
        piv.columns = piv.columns.astype(str)
        fig = px.imshow(piv, aspect="auto", color_continuous_scale=HEAT_SCALE, text_auto=".1f",
                        labels=dict(color="Avg stay (days)", x="", y=""))
        fig.update_traces(xgap=2, ygap=2, hovertemplate="%{y}<br>Severity: %{x}<br>Average stay: %{z:.1f} days<extra></extra>")
        fig.update_layout(coloraxis_colorbar=dict(tickfont=dict(color="#CBD5E1"), title_font=dict(color="#CBD5E1")))
        fig = tidy(fig, height=max(380, 26 * len(piv) + 100))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        show_chart(fig, "eda_dx_sev")

    col_a, col_b = st.columns(2)
    with col_a, st.container(border=True, key="card_eda_age_gender"):
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

    with col_b, st.container(border=True, key="card_eda_scatter"):
        st.subheader("Stay vs charges (sampled points)")
        pts = f.sample(min(4000, len(f)), random_state=1)
        fig = px.scatter(pts, x="Stay Days", y="Total Charges", color="Severity", log_x=True, log_y=True, opacity=0.52,
                         category_orders={"Severity": SEVERITY_ORDER}, color_discrete_map=SEVERITY_COLORS)
        fig.update_layout(xaxis_title="Length of stay (days, log scale)", yaxis_title="Total charges (US$, log scale)",
                          legend=dict(orientation="h", y=1.1, title=None))
        fig = tidy(fig, height=400)
        fig.update_xaxes(**STAY_TICKS)
        fig.update_yaxes(**MONEY_TICKS)
        show_chart(fig, "eda_scatter")

    with st.container(border=True, key="card_eda_facilities"):
        st.subheader("High-utilisation facilities")
        fac = (geo_only(f).groupby(["Facility", "County"], observed=True)
               .agg(Discharges=("Stay Days", "size"), **{"Avg stay (days)": ("Stay Days", "mean")},
                    **{"Avg charges ($)": ("Total Charges", "mean")}, **{"Total charges ($)": ("Total Charges", "sum")})
               .reset_index().query("Discharges > 0").nlargest(10, "Discharges"))
        fac["Facility"], fac["County"] = fac["Facility"].astype(str), fac["County"].astype(str)
        show_table(fac.round(1))

    with st.container(border=True, key="card_eda_region_payer"):
        st.subheader("Region × payer: average charges (pivot_table)")
        rp = (geo_only(f).pivot_table(index="Region", columns="Primary Payer", values="Total Charges", aggfunc="mean",
                                      observed=True).round(0))
        rp.index, rp.columns = rp.index.astype(str), rp.columns.astype(str)
        show_table(rp.reset_index())


def render_cleaning_tab(df: pd.DataFrame, cleaning_log: pd.DataFrame, missing_table: pd.DataFrame, summary: dict) -> None:
    """Render Tab 3 with cleaning metrics, step-by-step log, missing-value chart, and preview."""
    st.header("Data preprocessing and cleaning")
    render_kpi_cards([
        {"label": "Rows loaded", "value": f"{summary['raw_rows']:,}", "sub": "Raw 5% sample input", "accent": BLUE},
        {"label": "Rows after cleaning", "value": f"{summary['clean_rows']:,}",
         "sub": f"{summary['clean_rows'] / summary['raw_rows']:.1%} retained", "accent": AQUA},
        {"label": "Columns loaded", "value": str(summary["raw_cols"]), "sub": "Original SPARCS fields", "accent": VIOLET},
        {"label": "Columns after cleaning", "value": str(summary["clean_cols"]), "sub": "Includes 5 engineered", "accent": ORANGE},
    ])
    st.caption(f"The repository sample is a random 5% of the original {FULL_FILE_ROWS:,}-row file "
               "(created with scripts/make_sample.py).")

    with st.container(border=True, key="card_clean_log"):
        st.subheader("What was done, step by step")
        show_text_table(cleaning_log)

    with st.container(border=True, key="card_clean_missing"):
        st.subheader("Missing values in the raw data")
        fig = px.bar(missing_table.sort_values("Missing %"), x="Missing %", y="Column", orientation="h",
                     color_discrete_sequence=[BLUE])
        fig.add_vline(
            x=NULL_LIMIT * 100, line_dash="dash", line_color=ORANGE, line_width=2,
            annotation_text="50% drop limit", annotation_position="bottom right",
            annotation_bgcolor="rgba(11, 18, 34, 0.88)", annotation_font_color=ORANGE,
        )
        fig.update_traces(hovertemplate="%{y}: %{x:.1f}% missing<extra></extra>")
        fig.update_layout(yaxis_title=None, xaxis_title="Missing values (%)")
        show_chart(tidy(fig, height=max(300, 26 * len(missing_table) + 80)), "clean_missing")

    with st.container(border=True, key="card_clean_engineered"):
        st.subheader("Engineered columns")
        show_text_table(pd.DataFrame([
            {"Column": "Dx Category", "Meaning": "Body-system group from the first 3 letters of the CCSR diagnosis code"},
            {"Column": "Stay Band", "Meaning": "1-2, 3-7, 8-14 or 15+ days"},
            {"Column": "Charge per Day", "Meaning": "Total charges ÷ length of stay"},
            {"Column": "Cost-to-Charge Ratio", "Meaning": "Estimated cost ÷ billed charges (how much of the bill is real cost)"},
            {"Column": "Payer Group", "Meaning": "Government, Commercial, Self-pay or Other"},
            {"Column": "Outlier", "Meaning": "True if charges or stay are extreme (see the outlier step above)"},
        ]))

    with st.container(border=True, key="card_clean_preview"):
        st.subheader("Preview of the cleaned data")
        show_table(df.drop(columns=["Charge Outlier", "Stay Outlier"]).head(25))


def render_problem_tab() -> None:
    """Render Tab 4 with problem framing, stakeholders, research questions, background, and limits."""
    st.header("Understanding the problem")
    with st.container(border=True, key="card_prob_overview"):
        st.markdown(
            "**Client:** MediScope Health Analytics, working with hospitals to improve care and run operations more "
            "efficiently.  \n**Data:** 2021 inpatient discharges reported to New York State's SPARCS system - one row per "
            "hospital stay, with patient demographics, diagnosis, severity, payer, length of stay and billed charges."
        )

    with st.container(border=True, key="card_prob_users"):
        st.subheader("Who uses this dashboard, and for what?")
        show_text_table(pd.DataFrame([
            {"User": "Hospital administrators", "Decision": "Bed capacity, staffing and budgets",
             "What they look at": "Length of stay by diagnosis, county and facility; total charges by age group"},
            {"User": "Clinicians", "Decision": "Which cases need early discharge planning",
             "What they look at": "Stay and cost by severity of illness and diagnosis"},
            {"User": "Operations / finance managers", "Decision": "Payer mix and billing",
             "What they look at": "Distribution and average charges by payment type"},
        ]))

    with st.container(border=True, key="card_prob_questions"):
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

    left, right = st.columns([1.35, 1])
    with left, st.container(border=True, key="card_prob_research"):
        st.subheader("Background research")
        st.markdown(
            "- **Length of stay** is a standard efficiency measure in hospital operations: every extra day occupies a bed "
            "and staff, so administrators watch which diagnoses and facilities run long.\n"
            "- **Severity of illness** in this data comes from the APR-DRG system, which grades each stay as Minor, "
            "Moderate, Major or Extreme - so costs can be compared fairly between sicker and healthier patient mixes.\n"
            "- **Diagnoses** are coded with AHRQ's Clinical Classifications Software Refined (CCSR); the first three "
            "letters of each code give the body system (CIR = circulatory, RSP = respiratory ...), which this app uses "
            "to group the diagnosis codes into readable categories.\n"
            "- **Charges are not costs.** Charges are the amount a hospital bills; most payers pay negotiated or fixed "
            "rates, so billed charges are far higher than the estimated cost of care. The app shows both.\n"
            "- **Skewed money data:** hospital bills have a long right tail, so this dashboard reports medians next to "
            "means and offers log-scale axes."
        )
        st.markdown("**Sources:** [NY SPARCS](https://www.health.ny.gov/statistics/sparcs/) (data), "
                    "[AHRQ HCUP CCSR](https://hcup-us.ahrq.gov/toolssoftware/ccsr/ccs_refined.jsp) (diagnosis categories), "
                    "[Streamlit blog: improving healthcare management](https://blog.streamlit.io/improving-healthcare-management-with-streamlit/) "
                    "(dashboard design reference from the brief).")

    with right, st.container(border=True, key="card_prob_limits"):
        st.subheader("Limits of this analysis")
        st.markdown(
            "- The data has **no patient ID or readmission flag**, so readmission rates cannot be calculated; length of "
            "stay, charges and payer mix are used instead.\n"
            "- Figures come from a **5% random sample**, so small groups (a single county or rare diagnosis) can be noisy - "
            "use the minimum-cases slider.\n"
            "- Charges are billed amounts, not what was actually paid."
        )


def main() -> None:
    """Application entrypoint handling login gating and dashboard rendering."""
    init_auth_state()
    authenticated = bool(st.session_state.get("authenticated", False))
    inject_theme_css(authenticated)

    if not authenticated:
        render_login_page()
        st.stop()

    if not DATA_FILE.exists():
        st.error(f"Data file not found: {DATA_FILE}. Put hospital_sample.csv.gz inside the 'data' folder.")
        st.stop()

    df, cleaning_log, missing_table, summary = load_and_clean(str(DATA_FILE))
    filters, f = render_sidebar(df)
    render_hero_header(f, df, filters["no_outliers"])

    if len(f) < 30:
        st.warning("Fewer than 30 discharges match these filters - widen the filters to see reliable charts.")
        st.stop()

    render_kpi_cards([
        {"label": "Discharges", "value": f"{len(f):,}", "sub": f"{len(f) / len(df):.1%} of cleaned sample", "accent": BLUE},
        {"label": "Average stay", "value": f"{f['Stay Days'].mean():.1f} days",
         "sub": "Mean pulled up by long stays", "accent": AQUA},
        {"label": "Median stay", "value": f"{f['Stay Days'].median():.0f} days",
         "sub": "50th percentile inpatient stay", "accent": "#60A5FA"},
        {"label": "Average charges", "value": f"${f['Total Charges'].mean():,.0f}",
         "sub": f"Mean cost {usd(f['Total Costs'].mean())}", "accent": VIOLET},
        {"label": "Median charges", "value": f"${f['Total Charges'].median():,.0f}",
         "sub": f"Median cost {usd(f['Total Costs'].median())}", "accent": ORANGE},
    ])

    tab_dash, tab_eda, tab_clean, tab_problem = st.tabs(
        ["📊 Dashboard", "🔍 EDA & Insights", "🧹 Data Cleaning", "📖 Problem & Research"]
    )

    with tab_dash:
        render_dashboard_tab(f)
    with tab_eda:
        render_eda_tab(f)
    with tab_clean:
        render_cleaning_tab(df, cleaning_log, missing_table, summary)
    with tab_problem:
        render_problem_tab()


main()
