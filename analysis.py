import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ────────────────────────────────────────────────
#  Config & Data loading
# ────────────────────────────────────────────────
st.set_page_config(page_title="Motor Insurance Explorer + Yearly", layout="wide")
st.title("Motor Insurance Data – Interactive Summary with Yearly View")

@st.cache_data
def load_data():
    try:
        df = pd.read_excel('Motor Data.xlsx')
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

df = load_data()
if df is None:
    st.stop()

# ────────────────────────────────────────────────
#  Prepare year column (using policy effective date)
# ────────────────────────────────────────────────
year_col_name = 'policy_year' 

date_candidates = ['policy_effect_date', 'policy_effective_date', 'policy_effect_date']
main_date_col = next((c for c in date_candidates if c in df.columns), None)

if main_date_col:
    df[main_date_col] = pd.to_datetime(df[main_date_col], errors='coerce')
    df[year_col_name] = df[main_date_col].dt.year
    st.success(f"Using '{main_date_col}' → created 'policy_year' column")
else:
    st.warning("No policy effective date column found → yearly view disabled")
    df[year_col_name] = pd.NA
    year_col_name = None

# ────────────────────────────────────────────────
#  Column groups
# ────────────────────────────────────────────────
categorical_cols = [
    'vehicle_type', 'transaction_type', 'policyholder_gender',
    'policyholder_occupation', 'vehicle_brand', 'vehicle_model'
]

numerical_cols = [
    'sum_insured', 'basic_prem', 'actual_writtern_prem',
    'policy_excess', 'gross_weight', 'manufacture_year'
]

date_cols = [c for c in ['policy_effect_date', 'driver_DOB'] if c in df.columns]

all_displayable = [c for c in categorical_cols + numerical_cols + date_cols if c in df.columns]

# ────────────────────────────────────────────────
#  Controls
# ────────────────────────────────────────────────
st.markdown("### Display Controls")

colA, colB, colC, colD = st.columns([2, 2.5, 2, 1.5])

with colA:
    view_mode = st.radio("View mode", ["📊 Charts", "📋 Numbers"], horizontal=True)

with colB:
    selected_items = st.multiselect(
        "Columns to show",
        options=all_displayable
    )

with colC:
    if year_col_name and df[year_col_name].notna().any():
        available_years = sorted(df[year_col_name].dropna().unique().astype(int))
        selected_years = st.multiselect(
            "Filter years",
            options=available_years
        )
    else:
        selected_years = None

with colD:
    yearly_mode = st.checkbox("Show by year", value=True) if year_col_name else False

if not selected_items:
    st.info("Select at least one column.")
    st.stop()

# Filter years if selected
if selected_years and year_col_name:
    df_view = df[df[year_col_name].isin(selected_years)].copy()
else:
    df_view = df.copy()

# ────────────────────────────────────────────────
#  Render each selected column
# ────────────────────────────────────────────────
show_charts = "Charts" in view_mode
show_yearly = yearly_mode and year_col_name and df_view[year_col_name].notna().any()

for col in selected_items:
    st.subheader(f"{col}" + (f" – yearly" if show_yearly else ""))

    if col in categorical_cols:
        # ───── Categorical ─────
        if show_yearly:
            # Group by year + category
            agg = df_view.groupby([year_col_name, col]).size().reset_index(name='Count')
            agg['Year'] = agg[year_col_name].astype(str)

            if show_charts:
                fig = px.bar(
                    agg,
                    x='Year', y='Count',
                    color=col,
                    title=f"{col} distribution by policy year",
                    barmode='stack' if len(agg[col].unique()) <= 8 else 'group',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                pivot = agg.pivot(index='Year', columns=col, values='Count').fillna(0).astype(int)
                st.dataframe(pivot.style.background_gradient(cmap='YlGn'), use_container_width=True)

        else:
            # Overall (no yearly)
            vc = df_view[col].value_counts(dropna=False).reset_index()
            vc.columns = [col, 'Count']
            vc['%'] = (vc['Count'] / len(df_view) * 100).round(1)

            if show_charts:
                fig = px.bar(vc, x=col, y='Count', text='%', title=f"Top categories – {col}")
                fig.update_traces(texttemplate='%{text}%')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(vc, use_container_width=True)

    elif col in numerical_cols:
        st.subheader(f"{col}" + (f" – yearly" if show_yearly else ""))

        if show_yearly and year_col_name:
            # ───── Yearly summary statistics ─────
            summary = df_view.groupby(year_col_name)[col].describe().round(2)

            # Optional: rename index to look nicer
            summary.index = summary.index.astype(str) + " (year)"
            summary.index.name = "Year"

            # Make it look better in Streamlit
            st.dataframe(
                summary.style
                    .format(precision=2, thousands=",", decimal=".")
                    .set_caption(f"Summary statistics of {col} by policy year")
                    .set_properties(**{'text-align': 'right'}),
                use_container_width=True
            )

            # Optional: small line chart of mean/median over years
            if len(summary) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=summary.index,
                    y=summary['mean'],
                    mode='lines+markers',
                    name='Mean'
                ))
                fig.add_trace(go.Scatter(
                    x=summary.index,
                    y=summary['50%'],
                    mode='lines+markers',
                    name='Median'
                ))
                fig.update_layout(
                    title=f"Mean & Median of {col} over years",
                    xaxis_title="Year",
                    yaxis_title=col,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            # ───── Overall (no yearly) ─────
            if show_charts:
                fig = px.histogram(
                    df_view,
                    x=col,
                    title=f"Distribution of {col}",
                    marginal="box",
                    opacity=0.75
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Simple overall stats
                stats = df_view[col].describe().round(2).to_frame().T
                st.dataframe(
                    stats.style.format(precision=2, thousands=","),
                    use_container_width=True
                )

    elif col in date_cols:
        # ───── Dates ───── (almost always yearly makes sense)
        df_view[col] = pd.to_datetime(df_view[col], errors='coerce')

        if show_charts:
            yearly_counts = df_view[col].dt.year.value_counts().sort_index().reset_index()
            yearly_counts.columns = ['Year', 'Count']
            fig = px.bar(yearly_counts, x='Year', y='Count', title=f"Records per year – {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            yearly = df_view[col].dt.year.value_counts().sort_index().to_frame(name='Count')
            yearly['%'] = (yearly['Count'] / len(df_view) * 100).round(1)
            st.dataframe(yearly)

    st.markdown("---")

# ────────────────────────────────────────────────
#  Footer
# ────────────────────────────────────────────────
st.markdown("---")
st.caption(f"Dataset: {len(df):,} rows | Years available: {df[year_col_name].min()} – {df[year_col_name].max()}" if year_col_name else "")