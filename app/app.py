import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.express as px
from forecasting import forecast_load

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="UAC Healthcare Analytics",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

with open("app/styles.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():

    # Read dataset
    df = pd.read_csv("data/uac_data.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Rename columns
    df = df.rename(columns={
        'Children apprehended and placed in CBP custody*': 'Apprehended_CBP',
        'Children in CBP custody': 'CBP_Custody',
        'Children transferred out of CBP custody': 'Transferred_Out_CBP',
        'Children in HHS Care': 'HHS_Care',
        'Children discharged from HHS Care': 'Discharged_HHS'
    })

    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])

    # Numeric columns
    numeric_cols = [
        'Apprehended_CBP',
        'CBP_Custody',
        'Transferred_Out_CBP',
        'HHS_Care',
        'Discharged_HHS'
    ]

    # Clean numeric columns
    for col in numeric_cols:

        # Remove commas
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '', regex=False)
        )

        # Convert safely
        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

        # Replace missing values
        df[col] = df[col].fillna(0)

        # Convert to integer
        df[col] = df[col].astype(int)

    # Sort properly
    df = df.sort_values('Date')

    # Reset index
    df = df.reset_index(drop=True)

    return df

# ---------------------------------------------------
# LOAD DATAFRAME
# ---------------------------------------------------

df = load_data()

# ---------------------------------------------------
# DERIVED METRICS
# ---------------------------------------------------

# Total System Load
df['Total_System_Load'] = (
    df['CBP_Custody'] +
    df['HHS_Care']
)

# Net Intake Pressure
df['Net_Intake'] = (
    df['Transferred_Out_CBP'] -
    df['Discharged_HHS']
)

# Growth Rate
df['Growth_Rate'] = (
    df['Total_System_Load']
    .pct_change() * 100
)

# Rolling Average
df['Rolling_7_Day'] = (
    df['Total_System_Load']
    .rolling(7)
    .mean()
)

# Safe Division Ratio
df['Discharge_Offset_Ratio'] = (
    df['Discharged_HHS'] /
    df['Transferred_Out_CBP'].replace(0, 1)
)

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------

st.sidebar.header("Dashboard Filters")

start_date = st.sidebar.date_input(
    "Start Date",
    value=df['Date'].min()
)

end_date = st.sidebar.date_input(
    "End Date",
    value=df['Date'].max()
)

metric_option = st.sidebar.selectbox(
    "Select Metric",
    [
        'Total_System_Load',
        'CBP_Custody',
        'HHS_Care',
        'Net_Intake',
        'Rolling_7_Day'
    ]
)

# ---------------------------------------------------
# FILTER DATA
# ---------------------------------------------------

filtered_df = df[
    (df['Date'] >= pd.to_datetime(start_date)) &
    (df['Date'] <= pd.to_datetime(end_date))
]

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("Humanitarian Healthcare Capacity Intelligence System")

st.markdown("""
### U.S. Department of Health & Human Services  
Operational Monitoring & Care Capacity Analytics Dashboard
""")

# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------

st.markdown("""
<div style="
padding:20px;
border-radius:15px;
background: linear-gradient(90deg,#1E3A5F,#0E7490);
margin-bottom:25px;
">

<h2 style="color:white;">
Healthcare Capacity Intelligence & Forecasting Platform
</h2>

<p style="color:white;font-size:17px;">
Real-time operational analytics system for monitoring humanitarian healthcare capacity, intake pressure, discharge effectiveness, and predictive care load forecasting.
</p>

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# KPI SECTION
# ---------------------------------------------------

st.markdown("---")

st.subheader("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total System Load",
        f"{filtered_df['Total_System_Load'].iloc[-1]:,}"
    )

    st.caption(
        "Combined operational care burden across CBP and HHS systems."
    )

with col2:
    st.metric(
        "HHS Care Load",
        f"{filtered_df['HHS_Care'].iloc[-1]:,}"
    )

    st.caption(
        "Children actively receiving care under HHS supervision."
    )

with col3:
    st.metric(
        "CBP Custody",
        f"{filtered_df['CBP_Custody'].iloc[-1]:,}"
    )

    st.caption(
        "Children currently awaiting transfer within CBP custody."
    )

with col4:
    st.metric(
        "Net Intake Pressure",
        f"{filtered_df['Net_Intake'].iloc[-1]:,}"
    )

    st.caption(
        "Difference between transfers into HHS care and successful discharges."
    )

# ---------------------------------------------------
# ALERT ENGINE
# ---------------------------------------------------

st.markdown("---")

st.subheader("Operational Alerts")

threshold = 1000

if filtered_df['Net_Intake'].iloc[-1] > threshold:

    st.warning(
        "⚠ Sustained intake pressure detected. Potential backlog accumulation risk."
    )

else:

    st.success(
        "✅ System currently operating within manageable intake levels."
    )

# ---------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------

st.markdown("---")

st.subheader("Dataset Preview")

st.dataframe(filtered_df.head())

# ---------------------------------------------------
# DYNAMIC METRIC TREND
# ---------------------------------------------------

st.markdown("---")

st.subheader("Dynamic Trend Analysis")

fig_dynamic = px.line(
    filtered_df,
    x='Date',
    y=metric_option,
    title=f'{metric_option} Trend Over Time'
)

fig_dynamic.update_layout(
    xaxis_title="Date",
    yaxis_title="Metric Value",
    plot_bgcolor='#0F172A',
    paper_bgcolor='#0F172A',
    font=dict(color='white'),
    title_font=dict(size=22),
    xaxis=dict(showgrid=False),
    yaxis=dict(
        showgrid=True,
        gridcolor='#1E293B'
    )
)

st.plotly_chart(fig_dynamic, use_container_width=True)

# ---------------------------------------------------
# TOTAL SYSTEM LOAD GRAPH
# ---------------------------------------------------

st.markdown("---")

st.subheader("Total Healthcare System Load")

fig1 = px.line(
    filtered_df,
    x='Date',
    y='Total_System_Load',
    title='Total System Load Over Time'
)

fig1.update_layout(
    xaxis_title="Date",
    yaxis_title="Children Under Care",
    plot_bgcolor='#0F172A',
    paper_bgcolor='#0F172A',
    font=dict(color='white'),
    title_font=dict(size=22),
    xaxis=dict(showgrid=False),
    yaxis=dict(
        showgrid=True,
        gridcolor='#1E293B'
    )
)

st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------
# CBP VS HHS GRAPH
# ---------------------------------------------------

st.markdown("---")

st.subheader("CBP vs HHS Operational Comparison")

fig2 = px.line(
    filtered_df,
    x='Date',
    y=['CBP_Custody', 'HHS_Care'],
    title='CBP Custody vs HHS Care Load'
)

fig2.update_layout(
    xaxis_title="Date",
    yaxis_title="Children Count",
    plot_bgcolor='#0F172A',
    paper_bgcolor='#0F172A',
    font=dict(color='white'),
    title_font=dict(size=22),
    xaxis=dict(showgrid=False),
    yaxis=dict(
        showgrid=True,
        gridcolor='#1E293B'
    )
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------
# ROLLING AVERAGE
# ---------------------------------------------------

st.markdown("---")

st.subheader("Rolling Care Load Stability")

fig3 = px.line(
    filtered_df,
    x='Date',
    y='Rolling_7_Day',
    title='7-Day Rolling Average of Total System Load'
)

fig3.update_layout(
    xaxis_title="Date",
    yaxis_title="Rolling Average",
    plot_bgcolor='#0F172A',
    paper_bgcolor='#0F172A',
    font=dict(color='white'),
    title_font=dict(size=22),
    xaxis=dict(showgrid=False),
    yaxis=dict(
        showgrid=True,
        gridcolor='#1E293B'
    )
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------
# FORECASTING SECTION
# ---------------------------------------------------

st.markdown("---")

st.subheader("Forecasting & Predictive Analytics")

forecast_df = forecast_load(filtered_df)

fig4 = px.line(
    forecast_df,
    x='ds',
    y='yhat',
    title='30-Day Forecasted Healthcare System Load'
)

fig4.update_layout(
    xaxis_title="Future Timeline",
    yaxis_title="Projected Children Under Care",
    plot_bgcolor='#0F172A',
    paper_bgcolor='#0F172A',
    font=dict(color='white'),
    title_font=dict(size=22),
    xaxis=dict(showgrid=False),
    yaxis=dict(
        showgrid=True,
        gridcolor='#1E293B'
    )
)

st.plotly_chart(fig4, use_container_width=True)

st.info(
    "Forecasting helps estimate future operational burden and supports proactive staffing and shelter planning."
)

# ---------------------------------------------------
# OPERATIONAL INSIGHTS
# ---------------------------------------------------

st.markdown("---")

st.subheader("Operational Insights")

st.markdown("""
### Key Observations

- The healthcare care pipeline demonstrates periods of elevated intake pressure during high-transfer intervals.
- Sustained increases in Total System Load indicate temporary stress on sheltering and care infrastructure.
- Declining trends observed after early 2025 may reflect improved discharge coordination, policy stabilization, or reduced intake surges.
- HHS Care consistently represents the largest operational responsibility within the system.
- Variability spikes indicate periods requiring rapid healthcare resource allocation.
""")

# ---------------------------------------------------
# POLICY RECOMMENDATIONS
# ---------------------------------------------------

st.markdown("---")

st.subheader("Policy Recommendations")

st.markdown("""
### Recommended Operational Strategies

#### Improve Capacity Management
- Expand temporary shelter capacity during high-intake periods.
- Increase healthcare staffing during surge intervals.

#### Reduce Backlog Pressure
- Improve sponsor verification workflows to accelerate discharge efficiency.
- Enhance coordination between CBP and HHS transfer systems.

#### Strengthen Predictive Planning
- Monitor rolling intake trends for proactive operational planning.
- Use forecasting outputs for resource allocation and emergency preparedness.

#### Future Enhancements
- Integrate region-wise operational data for localized analysis.
- Add category-based filtering for demographic and care-type segmentation.
""")

# ---------------------------------------------------
# EXPORT ANALYTICS OUTPUTS
# ---------------------------------------------------

# Export cleaned dataset
df.to_csv(
    "exports/cleaned_dataset.csv",
    index=False
)

# Export forecast results
forecast_df.to_csv(
    "exports/forecast_output.csv",
    index=False
)

# KPI Summary Export
summary_df = pd.DataFrame({

    "Metric": [
        "Total System Load",
        "HHS Care Load",
        "CBP Custody",
        "Net Intake Pressure"
    ],

    "Value": [
        filtered_df['Total_System_Load'].iloc[-1],
        filtered_df['HHS_Care'].iloc[-1],
        filtered_df['CBP_Custody'].iloc[-1],
        filtered_df['Net_Intake'].iloc[-1]
    ]
})

summary_df.to_csv(
    "exports/analytics_summary.csv",
    index=False
)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "Humanitarian Healthcare Capacity Intelligence Dashboard | Healthcare Operations Analytics Platform"
)
