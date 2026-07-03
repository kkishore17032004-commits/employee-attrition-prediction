"""
Employee Attrition Prediction System
Production-quality Streamlit dashboard for HR analytics and attrition prediction.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = BASE_DIR / "attrition_model.pkl"
LOGO_PATH = BASE_DIR / "assets" / "company_logo.png"
REPORTS_DIR = BASE_DIR / "reports"

# Corporate color palette (dark-mode compatible)
COLORS = {
    "primary": "#1e40af",
    "secondary": "#3b82f6",
    "accent": "#06b6d4",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "dark": "#0f172a",
    "light": "#f8fafc",
    "card_bg": "rgba(255, 255, 255, 0.92)",
    "gradient_start": "#1e3a8a",
    "gradient_end": "#0891b2",
}

# Feature tooltips for the prediction form
FEATURE_TOOLTIPS: Dict[str, str] = {
    "Age": "Employee age in years.",
    "BusinessTravel": "Frequency of business travel for the role.",
    "DailyRate": "Daily rate of pay (USD).",
    "Department": "Organizational department assignment.",
    "DistanceFromHome": "Distance from home to office in miles.",
    "Education": "Education level (1=Below College to 5=Doctor).",
    "EducationField": "Field of highest education completed.",
    "EnvironmentSatisfaction": "Satisfaction with work environment (1-4).",
    "Gender": "Employee gender.",
    "HourlyRate": "Hourly rate of pay (USD).",
    "JobInvolvement": "Degree of job involvement (1-4).",
    "JobLevel": "Job level within the organization (1-5).",
    "JobRole": "Current job role/title category.",
    "JobSatisfaction": "Overall job satisfaction score (1-4).",
    "MaritalStatus": "Marital status of the employee.",
    "MonthlyIncome": "Monthly income in USD.",
    "MonthlyRate": "Monthly rate of pay.",
    "NumCompaniesWorked": "Number of companies previously worked at.",
    "OverTime": "Whether the employee works overtime.",
    "PercentSalaryHike": "Most recent percentage salary increase.",
    "PerformanceRating": "Most recent performance rating (1-4).",
    "RelationshipSatisfaction": "Satisfaction with relationships at work (1-4).",
    "StockOptionLevel": "Stock option level (0-3).",
    "TotalWorkingYears": "Total years of professional experience.",
    "TrainingTimesLastYear": "Number of training sessions in the last year.",
    "WorkLifeBalance": "Perceived work-life balance (1-4).",
    "YearsAtCompany": "Total years with the current company.",
    "YearsInCurrentRole": "Years in the current role.",
    "YearsSinceLastPromotion": "Years since last promotion.",
    "YearsWithCurrManager": "Years with current manager.",
}

# HR recommendation engine content
HR_RECOMMENDATIONS: Dict[str, List[str]] = {
    "High Risk": [
        "Salary Review – Evaluate compensation against market benchmarks.",
        "Reduce Overtime – Review workload and overtime assignments.",
        "Career Growth – Discuss promotion paths and development plans.",
        "Employee Engagement – Increase one-on-one check-ins and pulse surveys.",
        "Recognition – Implement immediate recognition for contributions.",
        "Manager Discussion – Schedule retention conversation with direct manager.",
        "Learning Program – Enroll in upskilling or certification programs.",
        "Flexible Working – Explore remote/hybrid or flexible schedule options.",
    ],
    "Medium Risk": [
        "Monthly Feedback – Establish regular feedback cadence.",
        "Recognition – Acknowledge achievements in team meetings.",
        "Performance Monitoring – Track engagement and performance trends.",
    ],
    "Low Risk": [
        "Career Development – Offer mentorship and growth opportunities.",
        "Recognition – Continue positive reinforcement programs.",
        "Training – Provide advanced training and skill development.",
        "Employee Wellness – Promote wellness and work-life balance initiatives.",
    ],
}

NAV_ITEMS = [
    ("Dashboard", "🏠"),
    ("Employee Prediction", "🤖"),
    ("Analytics", "📊"),
    ("Feature Importance", "📈"),
    ("Model Comparison", "📉"),
    ("About", "ℹ"),
]


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Prediction | HR Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS – Enterprise theme with dark-mode support
# ---------------------------------------------------------------------------

def inject_custom_css() -> None:
    """Inject professional enterprise styling into the Streamlit app."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .main .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }}

        /* Gradient header banner */
        .gradient-header {{
            background: linear-gradient(135deg, {COLORS['gradient_start']} 0%, {COLORS['gradient_end']} 100%);
            padding: 1.75rem 2rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(30, 64, 175, 0.25);
        }}
        .gradient-header h1 {{
            margin: 0;
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}
        .gradient-header p {{
            margin: 0.35rem 0 0 0;
            opacity: 0.9;
            font-size: 0.95rem;
        }}

        /* KPI metric cards */
        div[data-testid="stMetric"] {{
            background: {COLORS['card_bg']};
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        }}
        div[data-testid="stMetric"] label {{
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            color: #64748b !important;
        }}
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
            font-size: 1.65rem !important;
            font-weight: 700 !important;
            color: {COLORS['primary']} !important;
        }}

        /* Rounded content cards */
        .custom-card {{
            background: {COLORS['card_bg']};
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        }}

        /* Risk badges */
        .badge {{
            display: inline-block;
            padding: 0.45rem 1rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.9rem;
            margin: 0.25rem 0;
        }}
        .badge-low {{ background: #d1fae5; color: #065f46; }}
        .badge-medium {{ background: #fef3c7; color: #92400e; }}
        .badge-high {{ background: #fee2e2; color: #991b1b; }}

        /* Animated progress bars */
        .progress-container {{
            background: #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
            height: 28px;
            margin: 0.5rem 0 1rem 0;
        }}
        .progress-bar {{
            height: 100%;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 0.85rem;
            transition: width 1.2s ease-in-out;
        }}
        .progress-stay {{ background: linear-gradient(90deg, #059669, #10b981); }}
        .progress-leave {{ background: linear-gradient(90deg, #dc2626, #ef4444); }}

        .prob-large {{
            font-size: 2.2rem;
            font-weight: 700;
            margin: 0.25rem 0;
        }}

        /* Prediction result banner */
        .result-stay {{
            background: linear-gradient(135deg, #ecfdf5, #d1fae5);
            border-left: 5px solid {COLORS['success']};
            padding: 1.25rem;
            border-radius: 12px;
            margin: 1rem 0;
        }}
        .result-leave {{
            background: linear-gradient(135deg, #fef2f2, #fee2e2);
            border-left: 5px solid {COLORS['danger']};
            padding: 1.25rem;
            border-radius: 12px;
            margin: 1rem 0;
        }}

        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }}
        section[data-testid="stSidebar"] .stRadio label {{
            color: #e2e8f0 !important;
            font-weight: 500;
        }}
        section[data-testid="stSidebar"] .stRadio label:hover {{
            color: #ffffff !important;
        }}

        /* Footer */
        .footer {{
            text-align: center;
            color: #94a3b8;
            font-size: 0.82rem;
            padding: 1.5rem 0 0.5rem 0;
            border-top: 1px solid rgba(148, 163, 184, 0.2);
            margin-top: 2rem;
        }}

        /* Dark mode adjustments */
        @media (prefers-color-scheme: dark) {{
            div[data-testid="stMetric"] {{
                background: rgba(30, 41, 59, 0.85);
                border-color: rgba(148, 163, 184, 0.15);
            }}
            .custom-card {{
                background: rgba(30, 41, 59, 0.85);
                border-color: rgba(148, 163, 184, 0.15);
            }}
            div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
                color: #60a5fa !important;
            }}
        }}

        /* Hide default streamlit header/footer clutter */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached Data Loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading employee dataset...")
def load_dataset() -> pd.DataFrame:
    """Load and validate the HR employee attrition dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH.name}. "
            "Please ensure WA_Fn-UseC_-HR-Employee-Attrition.csv is in the project folder."
        )
    df = pd.read_csv(DATA_PATH)
    if "Attrition" not in df.columns:
        raise ValueError("Dataset is missing the 'Attrition' target column.")
    return df


@st.cache_resource(show_spinner="Loading prediction model...")
def load_model_bundle() -> Dict[str, Any]:
    """Load the trained model bundle from disk using joblib."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH.name}. "
            "Please run model training or place attrition_model.pkl in the project folder."
        )
    bundle = joblib.load(MODEL_PATH)

    # Support both raw model objects and full bundles
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle

    return {
        "model": bundle,
        "best_model_name": "Random Forest",
        "feature_columns": None,
        "label_encoders": {},
        "categorical_columns": [],
        "categorical_options": {},
        "numerical_ranges": {},
        "model_comparison": [],
        "evaluation": {},
        "accuracy": getattr(bundle, "score", lambda *_: 0.0),
    }


@st.cache_data
def compute_dashboard_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Pre-compute dashboard KPI statistics."""
    total = len(df)
    left = int((df["Attrition"] == "Yes").sum())
    stayed = total - left
    rate = (left / total * 100) if total else 0.0
    return {
        "total": total,
        "stayed": stayed,
        "left": left,
        "rate": rate,
    }


# ---------------------------------------------------------------------------
# UI Helper Functions
# ---------------------------------------------------------------------------

def render_header(title: str, subtitle: str) -> None:
    """Render the gradient page header."""
    st.markdown(
        f"""
        <div class="gradient-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render the application footer."""
    year = datetime.now().year
    st.markdown(
        f"""
        <div class="footer">
            Employee Attrition Prediction System &copy; {year} |
            Enterprise HR Analytics Dashboard | Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(stats: Dict[str, Any], best_model: str, accuracy: float) -> None:
    """Display KPI metric cards on the dashboard."""
    cols = st.columns(6)
    metrics = [
        ("Total Employees", f"{stats['total']:,}", None),
        ("Employees Stayed", f"{stats['stayed']:,}", None),
        ("Employees Left", f"{stats['left']:,}", None),
        ("Attrition Rate", f"{stats['rate']:.1f}%", None),
        ("Best Model", best_model, None),
        ("Model Accuracy", f"{accuracy * 100:.2f}%", None),
    ]
    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            st.metric(label=label, value=value, delta=delta)


def get_risk_level(leave_probability: float) -> Tuple[str, str]:
    """Classify attrition risk based on leave probability."""
    if leave_probability >= 0.60:
        return "High Risk", "badge-high"
    if leave_probability >= 0.35:
        return "Medium Risk", "badge-medium"
    return "Low Risk", "badge-low"


def render_progress_bar(label: str, percentage: float, css_class: str) -> None:
    """Render an animated HTML progress bar."""
    st.markdown(f"**{label}**")
    st.markdown(
        f"""
        <div class="progress-container">
            <div class="progress-bar {css_class}" style="width: {percentage:.1f}%;">
                {percentage:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gauge_chart(stay_prob: float, leave_prob: float) -> go.Figure:
    """Create a Plotly gauge chart for attrition probability."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=leave_prob * 100,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": "Leave Probability", "font": {"size": 16}},
            delta={"reference": 16.1, "suffix": "%", "relative": False},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": COLORS["danger"]},
                "steps": [
                    {"range": [0, 35], "color": "#d1fae5"},
                    {"range": [35, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": COLORS["dark"], "width": 3},
                    "thickness": 0.8,
                    "value": leave_prob * 100,
                },
            },
        )
    )
    fig.update_layout(height=320, margin=dict(t=40, b=20, l=30, r=30))
    return fig


def encode_input_features(
    input_data: Dict[str, Any],
    bundle: Dict[str, Any],
) -> pd.DataFrame:
    """Encode user form inputs into model-ready feature dataframe."""
    feature_columns: List[str] = bundle["feature_columns"]
    encoders = bundle.get("label_encoders", {})
    cat_cols = bundle.get("categorical_columns", [])

    encoded: Dict[str, Any] = {}
    for col in feature_columns:
        value = input_data[col]
        if col in cat_cols:
            le = encoders.get(col)
            if le is not None:
                if value not in le.classes_:
                    raise ValueError(f"Invalid value '{value}' for {col}.")
                encoded[col] = le.transform([value])[0]
            else:
                encoded[col] = value
        else:
            encoded[col] = float(value)

    return pd.DataFrame([encoded], columns=feature_columns)


def get_default_ranges(df: pd.DataFrame, bundle: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Build numerical ranges from bundle or dataset fallback."""
    if bundle.get("numerical_ranges"):
        return bundle["numerical_ranges"]

    ranges: Dict[str, Dict[str, float]] = {}
    drop_cols = {"EmployeeCount", "EmployeeNumber", "Over18", "StandardHours", "Attrition"}
    for col in df.columns:
        if col not in drop_cols and df[col].dtype in ["int64", "float64"]:
            ranges[col] = {
                "min": int(df[col].min()),
                "max": int(df[col].max()),
                "mean": float(df[col].mean()),
            }
    return ranges


def get_categorical_options(df: pd.DataFrame, bundle: Dict[str, Any]) -> Dict[str, List[str]]:
    """Get categorical column options from bundle or dataset."""
    if bundle.get("categorical_options"):
        return bundle["categorical_options"]

    options: Dict[str, List[str]] = {}
    for col in df.select_dtypes(include=["object", "str"]).columns:
        if col != "Attrition":
            options[col] = sorted(df[col].unique().tolist())
    return options


# ---------------------------------------------------------------------------
# Chart Functions – Dashboard
# ---------------------------------------------------------------------------

def chart_attrition_distribution(df: pd.DataFrame) -> go.Figure:
    """Pie chart of overall attrition distribution."""
    counts = df["Attrition"].value_counts()
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title="Attrition Distribution",
        color=counts.index,
        color_discrete_map={"Yes": COLORS["danger"], "No": COLORS["success"]},
        hole=0.45,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label+value", hovertemplate="%{label}: %{value}<extra></extra>")
    fig.update_layout(showlegend=True, height=380)
    return fig


def chart_department_attrition(df: pd.DataFrame) -> go.Figure:
    """Bar chart of department-wise attrition rate."""
    dept = df.groupby("Department")["Attrition"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
    dept.columns = ["Department", "Attrition Rate (%)"]
    fig = px.bar(
        dept,
        x="Department",
        y="Attrition Rate (%)",
        title="Department Wise Attrition Rate",
        color="Attrition Rate (%)",
        color_continuous_scale="Reds",
        text="Attrition Rate (%)",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", hovertemplate="%{x}: %{y:.1f}%<extra></extra>")
    fig.update_layout(height=380, coloraxis_showscale=False)
    return fig


def chart_gender_attrition(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart for gender-wise attrition."""
    cross = pd.crosstab(df["Gender"], df["Attrition"])
    fig = px.bar(
        cross.reset_index(),
        x="Gender",
        y=["No", "Yes"],
        title="Gender Wise Attrition",
        barmode="group",
        color_discrete_map={"No": COLORS["success"], "Yes": COLORS["danger"]},
    )
    fig.update_layout(height=380, legend_title="Attrition")
    fig.for_each_trace(lambda t: t.update(hovertemplate=f"{t.name}: %{{y}}<extra></extra>"))
    return fig


def chart_business_travel(df: pd.DataFrame) -> go.Figure:
    """Business travel attrition analysis."""
    travel = df.groupby("BusinessTravel")["Attrition"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
    travel.columns = ["Business Travel", "Attrition Rate (%)"]
    fig = px.bar(
        travel,
        x="Business Travel",
        y="Attrition Rate (%)",
        title="Business Travel Analysis",
        color="Attrition Rate (%)",
        color_continuous_scale="Oranges",
        text="Attrition Rate (%)",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", hovertemplate="%{x}: %{y:.1f}%<extra></extra>")
    fig.update_layout(height=380, coloraxis_showscale=False)
    return fig


def chart_overtime(df: pd.DataFrame) -> go.Figure:
    """Overtime attrition analysis."""
    ot = pd.crosstab(df["OverTime"], df["Attrition"], normalize="index") * 100
    ot = ot.reset_index()
    fig = px.bar(
        ot,
        x="OverTime",
        y=["Yes", "No"],
        title="OverTime Analysis (% within group)",
        barmode="stack",
        color_discrete_map={"Yes": COLORS["danger"], "No": COLORS["success"]},
    )
    fig.update_layout(height=380, legend_title="Attrition", yaxis_title="Percentage")
    return fig


def chart_age_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of age distribution by attrition."""
    fig = px.histogram(
        df,
        x="Age",
        color="Attrition",
        nbins=25,
        title="Age Distribution",
        barmode="overlay",
        opacity=0.75,
        color_discrete_map={"Yes": COLORS["danger"], "No": COLORS["success"]},
    )
    fig.update_layout(height=380)
    fig.for_each_trace(lambda t: t.update(hovertemplate="Age: %{x}<br>Count: %{y}<extra></extra>"))
    return fig


def chart_monthly_income(df: pd.DataFrame) -> go.Figure:
    """Box plot of monthly income by attrition status."""
    fig = px.box(
        df,
        x="Attrition",
        y="MonthlyIncome",
        title="Monthly Income Distribution",
        color="Attrition",
        color_discrete_map={"Yes": COLORS["danger"], "No": COLORS["success"]},
    )
    fig.update_layout(height=380, showlegend=False)
    fig.update_traces(hovertemplate="Income: $%{y:,}<extra></extra>")
    return fig


def chart_years_at_company(df: pd.DataFrame) -> go.Figure:
    """Violin plot of years at company."""
    fig = px.violin(
        df,
        x="Attrition",
        y="YearsAtCompany",
        title="Years At Company",
        color="Attrition",
        box=True,
        color_discrete_map={"Yes": COLORS["danger"], "No": COLORS["success"]},
    )
    fig.update_layout(height=380, showlegend=False)
    return fig


def chart_education_distribution(df: pd.DataFrame) -> go.Figure:
    """Education level distribution chart."""
    edu_labels = {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}
    temp = df.copy()
    temp["EducationLabel"] = temp["Education"].map(edu_labels)
    counts = temp.groupby(["EducationLabel", "Attrition"]).size().reset_index(name="Count")
    fig = px.bar(
        counts,
        x="EducationLabel",
        y="Count",
        color="Attrition",
        title="Education Distribution",
        barmode="group",
        color_discrete_map={"Yes": COLORS["danger"], "No": COLORS["success"]},
    )
    fig.update_layout(height=380)
    return fig


def chart_job_satisfaction(df: pd.DataFrame) -> go.Figure:
    """Job satisfaction vs attrition chart."""
    js = pd.crosstab(df["JobSatisfaction"], df["Attrition"])
    js = js.reset_index()
    fig = px.bar(
        js,
        x="JobSatisfaction",
        y=["No", "Yes"],
        title="Job Satisfaction vs Attrition",
        barmode="group",
        color_discrete_map={"No": COLORS["success"], "Yes": COLORS["danger"]},
    )
    fig.update_layout(height=380, xaxis_title="Job Satisfaction (1-4)")
    return fig


def chart_environment_satisfaction(df: pd.DataFrame) -> go.Figure:
    """Environment satisfaction vs attrition chart."""
    es = pd.crosstab(df["EnvironmentSatisfaction"], df["Attrition"])
    es = es.reset_index()
    fig = px.bar(
        es,
        x="EnvironmentSatisfaction",
        y=["No", "Yes"],
        title="Environment Satisfaction vs Attrition",
        barmode="group",
        color_discrete_map={"No": COLORS["success"], "Yes": COLORS["danger"]},
    )
    fig.update_layout(height=380, xaxis_title="Environment Satisfaction (1-4)")
    return fig


def chart_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Interactive correlation heatmap for numerical features."""
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if "EmployeeCount" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["EmployeeCount", "EmployeeNumber", "StandardHours"], errors="ignore")
    numeric_df["Attrition_Binary"] = (df["Attrition"] == "Yes").astype(int)
    corr = numeric_df.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    fig.update_layout(height=600)
    fig.update_traces(hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>")
    return fig


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------

def page_dashboard(df: pd.DataFrame, bundle: Dict[str, Any]) -> None:
    """Render the main HR analytics dashboard."""
    render_header(
        "HR Analytics Dashboard",
        "Real-time insights into workforce attrition patterns and model performance",
    )

    stats = compute_dashboard_stats(df)
    best_model = bundle.get("best_model_name", "Random Forest")
    accuracy = bundle.get("accuracy", 0.7993)

    render_kpi_cards(stats, best_model, accuracy)
    st.markdown("---")

    # Row 1 charts
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_attrition_distribution(df), use_container_width=True)
    with c2:
        st.plotly_chart(chart_department_attrition(df), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(chart_gender_attrition(df), use_container_width=True)
    with c4:
        st.plotly_chart(chart_business_travel(df), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(chart_overtime(df), use_container_width=True)
    with c6:
        st.plotly_chart(chart_age_distribution(df), use_container_width=True)

    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(chart_monthly_income(df), use_container_width=True)
    with c8:
        st.plotly_chart(chart_years_at_company(df), use_container_width=True)

    c9, c10 = st.columns(2)
    with c9:
        st.plotly_chart(chart_education_distribution(df), use_container_width=True)
    with c10:
        st.plotly_chart(chart_job_satisfaction(df), use_container_width=True)

    c11, c12 = st.columns(2)
    with c11:
        st.plotly_chart(chart_environment_satisfaction(df), use_container_width=True)
    with c12:
        st.plotly_chart(chart_correlation_heatmap(df), use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Employee Prediction
# ---------------------------------------------------------------------------

def build_prediction_form(
    df: pd.DataFrame,
    bundle: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build the employee attrition prediction input form."""
    cat_options = get_categorical_options(df, bundle)
    num_ranges = get_default_ranges(df, bundle)

    st.markdown("### Employee Profile")
    st.info("Fill in all employee attributes below. Hover over labels for field descriptions.")

    input_data: Dict[str, Any] = {}

    # --- Personal Information ---
    with st.expander("Personal & Demographic Information", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            input_data["Age"] = st.slider(
                "Age",
                min_value=int(num_ranges["Age"]["min"]),
                max_value=int(num_ranges["Age"]["max"]),
                value=int(num_ranges["Age"]["mean"]),
                help=FEATURE_TOOLTIPS["Age"],
            )
            input_data["Gender"] = st.selectbox(
                "Gender",
                options=cat_options.get("Gender", ["Male", "Female"]),
                help=FEATURE_TOOLTIPS["Gender"],
            )
        with c2:
            input_data["MaritalStatus"] = st.selectbox(
                "Marital Status",
                options=cat_options.get("MaritalStatus", ["Single", "Married", "Divorced"]),
                help=FEATURE_TOOLTIPS["MaritalStatus"],
            )
            input_data["DistanceFromHome"] = st.slider(
                "Distance From Home (miles)",
                min_value=int(num_ranges["DistanceFromHome"]["min"]),
                max_value=int(num_ranges["DistanceFromHome"]["max"]),
                value=int(num_ranges["DistanceFromHome"]["mean"]),
                help=FEATURE_TOOLTIPS["DistanceFromHome"],
            )
        with c3:
            input_data["Education"] = st.selectbox(
                "Education Level",
                options=[1, 2, 3, 4, 5],
                index=2,
                format_func=lambda x: {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}.get(x, str(x)),
                help=FEATURE_TOOLTIPS["Education"],
            )
            input_data["EducationField"] = st.selectbox(
                "Education Field",
                options=cat_options.get("EducationField", []),
                help=FEATURE_TOOLTIPS["EducationField"],
            )

    # --- Job Information ---
    with st.expander("Job & Role Information", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            input_data["Department"] = st.selectbox(
                "Department",
                options=cat_options.get("Department", []),
                help=FEATURE_TOOLTIPS["Department"],
            )
            input_data["JobRole"] = st.selectbox(
                "Job Role",
                options=cat_options.get("JobRole", []),
                help=FEATURE_TOOLTIPS["JobRole"],
            )
        with c2:
            input_data["JobLevel"] = st.slider(
                "Job Level",
                min_value=1,
                max_value=5,
                value=1,
                help=FEATURE_TOOLTIPS["JobLevel"],
            )
            input_data["JobInvolvement"] = st.slider(
                "Job Involvement",
                min_value=1,
                max_value=4,
                value=3,
                help=FEATURE_TOOLTIPS["JobInvolvement"],
            )
        with c3:
            input_data["BusinessTravel"] = st.selectbox(
                "Business Travel",
                options=cat_options.get("BusinessTravel", []),
                help=FEATURE_TOOLTIPS["BusinessTravel"],
            )
            input_data["OverTime"] = st.selectbox(
                "Over Time",
                options=cat_options.get("OverTime", ["Yes", "No"]),
                help=FEATURE_TOOLTIPS["OverTime"],
            )

    # --- Compensation ---
    with st.expander("Compensation & Benefits", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            input_data["MonthlyIncome"] = st.number_input(
                "Monthly Income ($)",
                min_value=int(num_ranges["MonthlyIncome"]["min"]),
                max_value=int(num_ranges["MonthlyIncome"]["max"]),
                value=int(num_ranges["MonthlyIncome"]["mean"]),
                step=100,
                help=FEATURE_TOOLTIPS["MonthlyIncome"],
            )
            input_data["DailyRate"] = st.number_input(
                "Daily Rate ($)",
                min_value=int(num_ranges["DailyRate"]["min"]),
                max_value=int(num_ranges["DailyRate"]["max"]),
                value=int(num_ranges["DailyRate"]["mean"]),
                help=FEATURE_TOOLTIPS["DailyRate"],
            )
        with c2:
            input_data["HourlyRate"] = st.number_input(
                "Hourly Rate ($)",
                min_value=int(num_ranges["HourlyRate"]["min"]),
                max_value=int(num_ranges["HourlyRate"]["max"]),
                value=int(num_ranges["HourlyRate"]["mean"]),
                help=FEATURE_TOOLTIPS["HourlyRate"],
            )
            input_data["MonthlyRate"] = st.number_input(
                "Monthly Rate",
                min_value=int(num_ranges["MonthlyRate"]["min"]),
                max_value=int(num_ranges["MonthlyRate"]["max"]),
                value=int(num_ranges["MonthlyRate"]["mean"]),
                help=FEATURE_TOOLTIPS["MonthlyRate"],
            )
        with c3:
            input_data["PercentSalaryHike"] = st.slider(
                "Percent Salary Hike",
                min_value=int(num_ranges["PercentSalaryHike"]["min"]),
                max_value=int(num_ranges["PercentSalaryHike"]["max"]),
                value=int(num_ranges["PercentSalaryHike"]["mean"]),
                help=FEATURE_TOOLTIPS["PercentSalaryHike"],
            )
            input_data["StockOptionLevel"] = st.slider(
                "Stock Option Level",
                min_value=0,
                max_value=3,
                value=0,
                help=FEATURE_TOOLTIPS["StockOptionLevel"],
            )

    # --- Satisfaction Scores ---
    with st.expander("Satisfaction & Performance", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            input_data["JobSatisfaction"] = st.slider(
                "Job Satisfaction",
                1, 4, 3,
                help=FEATURE_TOOLTIPS["JobSatisfaction"],
            )
            input_data["EnvironmentSatisfaction"] = st.slider(
                "Environment Satisfaction",
                1, 4, 3,
                help=FEATURE_TOOLTIPS["EnvironmentSatisfaction"],
            )
        with c2:
            input_data["RelationshipSatisfaction"] = st.slider(
                "Relationship Satisfaction",
                1, 4, 3,
                help=FEATURE_TOOLTIPS["RelationshipSatisfaction"],
            )
            input_data["WorkLifeBalance"] = st.slider(
                "Work Life Balance",
                1, 4, 3,
                help=FEATURE_TOOLTIPS["WorkLifeBalance"],
            )
            input_data["PerformanceRating"] = st.slider(
                "Performance Rating",
                1, 4, 3,
                help=FEATURE_TOOLTIPS["PerformanceRating"],
            )

    # --- Tenure & Experience ---
    with st.expander("Tenure & Experience", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            input_data["TotalWorkingYears"] = st.slider(
                "Total Working Years",
                min_value=int(num_ranges["TotalWorkingYears"]["min"]),
                max_value=int(num_ranges["TotalWorkingYears"]["max"]),
                value=int(num_ranges["TotalWorkingYears"]["mean"]),
                help=FEATURE_TOOLTIPS["TotalWorkingYears"],
            )
            input_data["YearsAtCompany"] = st.slider(
                "Years At Company",
                min_value=int(num_ranges["YearsAtCompany"]["min"]),
                max_value=int(num_ranges["YearsAtCompany"]["max"]),
                value=int(num_ranges["YearsAtCompany"]["mean"]),
                help=FEATURE_TOOLTIPS["YearsAtCompany"],
            )
        with c2:
            input_data["YearsInCurrentRole"] = st.slider(
                "Years In Current Role",
                min_value=int(num_ranges["YearsInCurrentRole"]["min"]),
                max_value=int(num_ranges["YearsInCurrentRole"]["max"]),
                value=int(num_ranges["YearsInCurrentRole"]["mean"]),
                help=FEATURE_TOOLTIPS["YearsInCurrentRole"],
            )
            input_data["YearsSinceLastPromotion"] = st.slider(
                "Years Since Last Promotion",
                min_value=int(num_ranges["YearsSinceLastPromotion"]["min"]),
                max_value=int(num_ranges["YearsSinceLastPromotion"]["max"]),
                value=int(num_ranges["YearsSinceLastPromotion"]["mean"]),
                help=FEATURE_TOOLTIPS["YearsSinceLastPromotion"],
            )
        with c3:
            input_data["YearsWithCurrManager"] = st.slider(
                "Years With Current Manager",
                min_value=int(num_ranges["YearsWithCurrManager"]["min"]),
                max_value=int(num_ranges["YearsWithCurrManager"]["max"]),
                value=int(num_ranges["YearsWithCurrManager"]["mean"]),
                help=FEATURE_TOOLTIPS["YearsWithCurrManager"],
            )
            input_data["NumCompaniesWorked"] = st.slider(
                "Number of Companies Worked",
                min_value=int(num_ranges["NumCompaniesWorked"]["min"]),
                max_value=int(num_ranges["NumCompaniesWorked"]["max"]),
                value=int(num_ranges["NumCompaniesWorked"]["mean"]),
                help=FEATURE_TOOLTIPS["NumCompaniesWorked"],
            )
            input_data["TrainingTimesLastYear"] = st.slider(
                "Training Times Last Year",
                min_value=int(num_ranges["TrainingTimesLastYear"]["min"]),
                max_value=int(num_ranges["TrainingTimesLastYear"]["max"]),
                value=int(num_ranges["TrainingTimesLastYear"]["mean"]),
                help=FEATURE_TOOLTIPS["TrainingTimesLastYear"],
            )

    # Validation: years consistency check
    if input_data["YearsAtCompany"] > input_data["TotalWorkingYears"]:
        st.warning("Years At Company cannot exceed Total Working Years. Please adjust the values.")
        return None

    submitted = st.button("Predict Attrition", type="primary", use_container_width=True)
    if submitted:
        return input_data
    return None


def display_prediction_results(
    input_data: Dict[str, Any],
    bundle: Dict[str, Any],
) -> None:
    """Run prediction and display results with probabilities and recommendations."""
    try:
        model = bundle["model"]
        features_df = encode_input_features(input_data, bundle)

        with st.spinner("Analyzing employee profile..."):
            prediction = model.predict(features_df)[0]
            probabilities = model.predict_proba(features_df)[0]

        stay_prob = float(probabilities[0])
        leave_prob = float(probabilities[1])
        confidence = float(max(probabilities))
        risk_label, badge_class = get_risk_level(leave_prob)

        # Store in session for download
        result_record = {**input_data}
        result_record["Prediction"] = "Leave" if prediction == 1 else "Stay"
        result_record["Stay_Probability"] = round(stay_prob * 100, 2)
        result_record["Leave_Probability"] = round(leave_prob * 100, 2)
        result_record["Confidence_Score"] = round(confidence * 100, 2)
        result_record["Risk_Level"] = risk_label
        st.session_state["last_prediction"] = result_record

        st.success("Prediction completed successfully!")

        # Result banner
        if prediction == 1:
            st.markdown(
                """
                <div class="result-leave">
                    <h2 style="margin:0; color:#991b1b;">Employee Likely to Leave</h2>
                    <p style="margin:0.5rem 0 0 0;">This employee shows elevated attrition risk based on the trained model.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-stay">
                    <h2 style="margin:0; color:#065f46;">Employee Likely to Stay</h2>
                    <p style="margin:0.5rem 0 0 0;">This employee profile indicates a lower likelihood of attrition.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Stay Probability", f"{stay_prob * 100:.1f}%")
        with m2:
            st.metric("Leave Probability", f"{leave_prob * 100:.1f}%")
        with m3:
            st.metric("Confidence Score", f"{confidence * 100:.1f}%")

        st.markdown(
            f'<span class="badge {badge_class}">{risk_label}</span>',
            unsafe_allow_html=True,
        )

        # Probability section
        st.markdown("### Probability Analysis")
        col_bar, col_gauge = st.columns([1, 1])
        with col_bar:
            st.markdown(
                f'<p class="prob-large" style="color:{COLORS["success"]};">{stay_prob * 100:.1f}%</p>',
                unsafe_allow_html=True,
            )
            render_progress_bar("Stay Probability", stay_prob * 100, "progress-stay")
            st.markdown(
                f'<p class="prob-large" style="color:{COLORS["danger"]};">{leave_prob * 100:.1f}%</p>',
                unsafe_allow_html=True,
            )
            render_progress_bar("Leave Probability", leave_prob * 100, "progress-leave")

        with col_gauge:
            st.plotly_chart(render_gauge_chart(stay_prob, leave_prob), use_container_width=True)

        # HR Recommendations
        st.markdown("### HR Recommendation Engine")
        st.markdown(
            f'<div class="custom-card"><strong>Risk Classification:</strong> {risk_label}</div>',
            unsafe_allow_html=True,
        )
        for rec in HR_RECOMMENDATIONS.get(risk_label, []):
            st.markdown(f"- {rec}")

        # Download section
        st.markdown("### Download Results")
        result_df = pd.DataFrame([result_record])
        csv_data = result_df.to_csv(index=False).encode("utf-8")
        REPORTS_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Download Prediction Result (CSV)",
            data=csv_data,
            file_name=f"prediction_result_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except ValueError as exc:
        st.error(f"Validation Error: {exc}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.warning("Please verify all input values and ensure the model file is compatible.")


def page_prediction(df: pd.DataFrame, bundle: Dict[str, Any]) -> None:
    """Render the employee attrition prediction page."""
    render_header(
        "Employee Attrition Prediction",
        "Enter employee details to predict attrition risk using the trained machine learning model",
    )

    input_data = build_prediction_form(df, bundle)
    if input_data is not None:
        display_prediction_results(input_data, bundle)


# ---------------------------------------------------------------------------
# Page: Feature Importance
# ---------------------------------------------------------------------------

def page_feature_importance(bundle: Dict[str, Any]) -> None:
    """Display Random Forest feature importance analysis."""
    render_header(
        "Feature Importance Analysis",
        "Top predictive features from the Random Forest attrition model",
    )

    model = bundle.get("model")
    feature_columns = bundle.get("feature_columns", [])

    if model is None or not hasattr(model, "feature_importances_"):
        st.error("Feature importance is unavailable. The loaded model does not support this analysis.")
        return

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    importance_df["Importance (%)"] = (
        importance_df["Importance"] / importance_df["Importance"].sum() * 100
    ).round(2)

    top10 = importance_df.head(10)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = px.bar(
            top10.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Feature Importance",
            color="Importance",
            color_continuous_scale="Blues",
            text="Importance (%)",
        )
        fig.update_traces(
            texttemplate="%{text:.1f}%",
            hovertemplate="Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>",
        )
        fig.update_layout(height=480, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Importance Table")
        st.dataframe(
            top10[["Feature", "Importance", "Importance (%)"]].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        total_top10 = top10["Importance (%)"].sum()
        st.metric("Top 10 Combined Importance", f"{total_top10:.1f}%")


# ---------------------------------------------------------------------------
# Page: Analytics (Filtered)
# ---------------------------------------------------------------------------

def page_analytics(df: pd.DataFrame) -> None:
    """Render filtered analytics dashboard."""
    render_header(
        "Workforce Analytics",
        "Explore attrition patterns with dynamic filters across departments and demographics",
    )

    st.markdown("### Filters")
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        dept_filter = st.multiselect("Department", options=sorted(df["Department"].unique()), default=[])
    with f2:
        gender_filter = st.multiselect("Gender", options=sorted(df["Gender"].unique()), default=[])
    with f3:
        role_filter = st.multiselect("Job Role", options=sorted(df["JobRole"].unique()), default=[])
    with f4:
        ot_filter = st.multiselect("OverTime", options=sorted(df["OverTime"].unique()), default=[])
    with f5:
        travel_filter = st.multiselect("Business Travel", options=sorted(df["BusinessTravel"].unique()), default=[])

    filtered = df.copy()
    if dept_filter:
        filtered = filtered[filtered["Department"].isin(dept_filter)]
    if gender_filter:
        filtered = filtered[filtered["Gender"].isin(gender_filter)]
    if role_filter:
        filtered = filtered[filtered["JobRole"].isin(role_filter)]
    if ot_filter:
        filtered = filtered[filtered["OverTime"].isin(ot_filter)]
    if travel_filter:
        filtered = filtered[filtered["BusinessTravel"].isin(travel_filter)]

    if filtered.empty:
        st.warning("No records match the selected filters. Please adjust your filter criteria.")
        return

    st.success(f"Showing analytics for {len(filtered):,} employees")

    stats = compute_dashboard_stats(filtered)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Filtered Employees", f"{stats['total']:,}")
    with c2:
        st.metric("Attrition Rate", f"{stats['rate']:.1f}%")
    with c3:
        st.metric("Employees Left", f"{stats['left']:,}")
    with c4:
        st.metric("Employees Stayed", f"{stats['stayed']:,}")

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(chart_department_attrition(filtered), use_container_width=True)
        st.plotly_chart(chart_overtime(filtered), use_container_width=True)
    with c6:
        st.plotly_chart(chart_gender_attrition(filtered), use_container_width=True)
        st.plotly_chart(chart_monthly_income(filtered), use_container_width=True)

    st.plotly_chart(chart_correlation_heatmap(filtered), use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model Comparison
# ---------------------------------------------------------------------------

def page_model_comparison(bundle: Dict[str, Any]) -> None:
    """Display model comparison metrics, confusion matrices, and ROC curves."""
    render_header(
        "Model Comparison",
        "Performance evaluation of Logistic Regression, Decision Tree, and Random Forest classifiers",
    )

    comparison = bundle.get("model_comparison", [])
    evaluation = bundle.get("evaluation", {})
    best_name = bundle.get("best_model_name", "Random Forest")

    if not comparison:
        st.warning("Model comparison data not available in the model bundle.")
        return

    comp_df = pd.DataFrame(comparison)
    comp_df["Accuracy (%)"] = (comp_df["Accuracy"] * 100).round(2)
    comp_df["Precision (%)"] = (comp_df["Precision"] * 100).round(2)
    comp_df["Recall (%)"] = (comp_df["Recall"] * 100).round(2)
    comp_df["F1 Score (%)"] = (comp_df["F1 Score"] * 100).round(2)

    st.markdown(f"**Best Model:** {best_name} (highlighted in comparison table)")

    display_df = comp_df[
        ["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"]
    ].copy()

    def highlight_best(row: pd.Series) -> List[str]:
        """Highlight the best model row in the comparison table."""
        if row["Model"] == best_name:
            return ["background-color: #dbeafe; font-weight: 600"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # Accuracy comparison chart
    fig_acc = px.bar(
        comp_df,
        x="Model",
        y="Accuracy (%)",
        title="Model Accuracy Comparison",
        color="Model",
        color_discrete_sequence=[COLORS["primary"], COLORS["secondary"], COLORS["accent"]],
        text="Accuracy (%)",
    )
    fig_acc.update_traces(texttemplate="%{text:.2f}%", hovertemplate="%{x}: %{y:.2f}%<extra></extra>")
    fig_acc.add_hline(y=comp_df["Accuracy (%)"].max(), line_dash="dash", line_color=COLORS["success"])
    st.plotly_chart(fig_acc, use_container_width=True)

    # Precision, Recall, F1 grouped chart
    metrics_melt = comp_df.melt(
        id_vars="Model",
        value_vars=["Precision (%)", "Recall (%)", "F1 Score (%)"],
        var_name="Metric",
        value_name="Score",
    )
    fig_metrics = px.bar(
        metrics_melt,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Precision, Recall & F1 Score Comparison",
        text="Score",
    )
    fig_metrics.update_traces(texttemplate="%{text:.1f}", hovertemplate="%{x}<br>%{fullData.name}: %{y:.1f}%<extra></extra>")
    st.plotly_chart(fig_metrics, use_container_width=True)

    # Confusion matrices and ROC curves per model
    model_names = ["Logistic Regression", "Decision Tree", "Random Forest"]
    tabs = st.tabs(model_names)

    for tab, name in zip(tabs, model_names):
        with tab:
            if name not in evaluation:
                st.info(f"No evaluation data for {name}.")
                continue

            eval_data = evaluation[name]
            cm = np.array(eval_data["confusion_matrix"])
            roc_data = eval_data.get("roc", {})

            c1, c2 = st.columns(2)
            with c1:
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    title=f"{name} – Confusion Matrix",
                    labels={"x": "Predicted", "y": "Actual"},
                    x=["Stay", "Leave"],
                    y=["Stay", "Leave"],
                    color_continuous_scale="Blues",
                )
                fig_cm.update_layout(height=400)
                st.plotly_chart(fig_cm, use_container_width=True)

            with c2:
                if roc_data:
                    fig_roc = go.Figure()
                    fig_roc.add_trace(
                        go.Scatter(
                            x=roc_data["fpr"],
                            y=roc_data["tpr"],
                            mode="lines",
                            name=f"ROC (AUC={roc_data['auc']:.3f})",
                            line=dict(color=COLORS["primary"], width=3),
                        )
                    )
                    fig_roc.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode="lines",
                            name="Random",
                            line=dict(dash="dash", color="gray"),
                        )
                    )
                    fig_roc.update_layout(
                        title=f"{name} – ROC Curve",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        height=400,
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------

def page_about() -> None:
    """Render project information and credits."""
    render_header(
        "About This Project",
        "Employee Attrition Prediction System – Enterprise HR Analytics Platform",
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=180)
        st.markdown("### Project Title")
        st.markdown("**Employee Attrition Prediction System**")

    with c2:
        st.markdown(
            """
            ### Project Description
            This enterprise HR analytics dashboard leverages machine learning to predict
            employee attrition risk. It empowers HR teams with actionable insights,
            interactive visualizations, and data-driven retention recommendations.

            ### Dataset Information
            - **Source:** IBM HR Analytics Employee Attrition Dataset
            - **Records:** 1,470 employees
            - **Features:** 35 attributes covering demographics, job role, satisfaction, and compensation
            - **Target:** Employee attrition (Yes/No)

            ### Algorithms Used
            - Logistic Regression
            - Decision Tree Classifier
            - Random Forest Classifier (Best Model)
            - SMOTE for class imbalance handling

            ### Technologies Used
            Python | Streamlit | Pandas | NumPy | Scikit-learn | Plotly | Matplotlib | Joblib

            ### Developer Information
            Built as an enterprise-grade HR analytics solution for workforce planning,
            retention strategy, and predictive people analytics.
            """
        )


# ---------------------------------------------------------------------------
# Sidebar & Main Application Router
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    """Render the professional sidebar navigation."""
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
        st.markdown("## HR Analytics")
        st.markdown("*Attrition Prediction System*")
        st.markdown("---")

        page_labels = [f"{icon} {name}" for name, icon in NAV_ITEMS]
        page_map = {label: name for (name, icon), label in zip(NAV_ITEMS, page_labels)}

        selected_label = st.radio(
            "Navigation",
            options=page_labels,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Quick Stats**")
        try:
            df = load_dataset()
            stats = compute_dashboard_stats(df)
            st.caption(f"Workforce Size: {stats['total']:,}")
            st.caption(f"Attrition Rate: {stats['rate']:.1f}%")
        except Exception:
            st.caption("Dataset unavailable")

        st.markdown("---")
        st.caption("v1.0.0 | Enterprise Edition")

    return page_map[selected_label]


def main() -> None:
    """Main application entry point."""
    inject_custom_css()

    # Load resources with error handling
    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        st.stop()

    try:
        bundle = load_model_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(
            "To generate the model, run the training script or ensure "
            "`attrition_model.pkl` exists in the project directory."
        )
        bundle = None
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        bundle = None

    page = render_sidebar()

    # Route to selected page
    if page == "Dashboard":
        page_dashboard(df, bundle or {})
    elif page == "Employee Prediction":
        if bundle is None:
            st.error("Prediction unavailable – model file is missing or corrupted.")
        else:
            page_prediction(df, bundle)
    elif page == "Analytics":
        page_analytics(df)
    elif page == "Feature Importance":
        if bundle is None:
            st.error("Feature importance unavailable – model file is missing.")
        else:
            page_feature_importance(bundle)
    elif page == "Model Comparison":
        if bundle is None:
            st.error("Model comparison unavailable – model file is missing.")
        else:
            page_model_comparison(bundle)
    elif page == "About":
        page_about()

    render_footer()


if __name__ == "__main__":
    main()
