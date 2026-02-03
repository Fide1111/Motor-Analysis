import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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
else:
    st.warning("No policy effective date column found → yearly view disabled")
    df[year_col_name] = pd.NA
    year_col_name = None

if 'driver_age' in df.columns:
    df['driver_age'] = pd.to_numeric(df['driver_age'], errors='coerce')

if 'vehicle_age' in df.columns:
    df['vehicle_age'] = pd.to_numeric(df['vehicle_age'], errors='coerce')

# Calculate missing ages
if 'driver_DOB' in df.columns and main_date_col:
    df['driver_DOB'] = pd.to_datetime(df['driver_DOB'], errors='coerce')
    ref_date = df[main_date_col].max()                      # or .mean(), or pd.Timestamp('2025-12-31')
    df['driver_age'] = (ref_date - df['driver_DOB']).dt.days / 365.25
    df['driver_age'] = df['driver_age'].round(1).clip(16, 100)

# Same for vehicle_age if missing / unreliable
if 'manufacture_year' in df.columns and main_date_col:
    df['vehicle_age'] = df[main_date_col].dt.year - df['manufacture_year']
    df['vehicle_age'] = df['vehicle_age'].clip(0, 40)      # reasonable bounds

# ────────────────────────────────────────────────
#  Column groups
# ────────────────────────────────────────────────
categorical_cols = [
    'vehicle_type', 'transaction_type', 'policyholder_gender',
    'policyholder_occupation', 'vehicle_brand', 'vehicle_model'
]

numerical_cols = [
    'sum_insured', 'basic_prem', 'actual_written_prem',
    'policy_excess', 'gross_weight', 'manufacture_year',
    'vehicle_age'
]

expensive_brand = [
    'ALFA ROMEO', 'ASTON MARTIN', 'AUDI', 'BENTLEY', 'BMW', 'FERRARI', 'LAMBORGHINI', 'LAND ROVER', 'LOTUS', 'MASERATI', 
    'MERCEDES BENZ', 'PORSCHE', 'ROLLS-ROYCE', 'McLaren'
]

ch_brand = [
    'BAOJUN', 'CHANGAN', 'DENZA', 'DONG FENG', 'GEELY', 'FORTHING', 'Great Wall', 'HONGQI', 'LEAPMOTOR', 'NETA',
    'NIO', 'ORA', 'TRUMPCHI', 'WELTMEISTER', 'XIAOMI', 'XPENG', 'ZEEKR'
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
    # Only show columns that can reasonably be averaged / trended
    trendable_columns = numerical_cols + ['driver_age']  # add more if needed

    # Filter to columns actually present in the data
    available_trend_cols = [c for c in trendable_columns if c in df.columns]

    if not available_trend_cols:
        st.warning("No numeric/trendable columns found in the dataset")
        selected_items = None
    else:
        selected_items = st.selectbox(
            "Select column to analyze / trend",
            options=available_trend_cols,
            index=available_trend_cols.index('actual_written_prem') 
                  if 'actual_written_prem' in available_trend_cols 
                  else 0,
            help="Only numeric columns are shown here (for average trend calculation)"
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

brand_col = next((c for c in ['vehicle_brand', 'Vehicle_Brand', 'brand'] if c in df_view.columns), None)

@st.cache_data
def compute_insights(df):
    insights = {}
    
    # Driver age (using pre-calculated column from Excel)
    if 'driver_age' in df.columns:
        df_temp = df[df['driver_age'].notna() & df['driver_age'].between(16, 100)].copy()
        if not df_temp.empty:
            age_stats = df_temp['driver_age'].describe(percentiles=[0.25, 0.5, 0.75]).round(1)
            age_bands = pd.cut(df_temp['driver_age'], bins=[0,18,25,30,35,40,45,50,55,60,65,70,100],
                               labels=['<18','18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70+'])
            age_dist = age_bands.value_counts().sort_index()
            age_pct = (age_dist / len(df_temp) * 100).round(1)
            
            insights['driver_age'] = {
                'stats': age_stats.to_dict(),
                'distribution': pd.concat([age_dist, age_pct], axis=1, keys=['Count', '%']).to_dict(orient='index')
            }
        else:
            insights['driver_age'] = {}
    
    # % with no_claim_discount > 0
    if 'no_claim_discount' in df.columns:
        has_ncd = df['no_claim_discount'] > 0
        ncd_count = has_ncd.value_counts()
        ncd_pct = (ncd_count / len(df) * 100).round(1)
        insights['ncd'] = {
            'with': int(ncd_count.get(True, 0)),
            'pct_with': ncd_pct.get(True, 0.0),
            'without': int(ncd_count.get(False, 0)),
            'pct_without': ncd_pct.get(False, 0.0)
        }
    
    # % with no_intermediary_discount > 0
    if 'no_intermediary_discount' in df.columns:
        has_inter = df['no_intermediary_discount'] > 0
        inter_count = has_inter.value_counts()
        inter_pct = (inter_count / len(df) * 100).round(1)
        insights['intermediary_discount'] = {
            'with': int(inter_count.get(True, 0)),
            'pct_with': inter_pct.get(True, 0.0),
            'without': int(inter_count.get(False, 0)),
            'pct_without': inter_pct.get(False, 0.0)
        }
    
    # Vehicle age – calculated from manufacture_year and policy_effect_date / policy_year
    if 'manufacture_year' in df.columns:
        df_temp = df.copy()
        if 'policy_effect_date' in df_temp.columns:
            df_temp['policy_effect_date'] = pd.to_datetime(df_temp['policy_effect_date'], errors='coerce')
            df_temp['vehicle_age'] = df_temp['policy_effect_date'].dt.year - df_temp['manufacture_year']
        elif 'policy_year' in df_temp.columns:
            df_temp['vehicle_age'] = df_temp['policy_year'] - df_temp['manufacture_year']
        else:
            df_temp['vehicle_age'] = np.nan
        
        df_temp = df_temp[df_temp['vehicle_age'].between(0, 40)]
        if not df_temp.empty:
            veh_stats = df_temp['vehicle_age'].describe().round(1)
            insights['vehicle_age'] = {
                'stats': veh_stats.to_dict(),
                'data_for_plot': df_temp[['vehicle_age', 'actual_written_prem', 'sum_insured', 
                                          'basic_prem','policy_excess']].dropna()
            }
    
    # Time trend
    if 'policy_year' in df.columns:
        year_trend = df['policy_year'].value_counts().sort_index()
        year_pct = (year_trend / len(df) * 100).round(1)
        insights['time_trend'] = pd.concat([year_trend, year_pct], axis=1, keys=['Policies', '%']).to_dict(orient='index')
    
    return insights

# Call it
insights = compute_insights(df_view)

# ────────────────────────────────────────────────
#  Key Portfolio Insights
# ────────────────────────────────────────────────
st.markdown("### Key Portfolio Insights")

col1, col2, col3 = st.columns(3)

with col1:
    if 'driver_age' in df_view.columns:
        valid = df_view['driver_age'].notna() & df_view['driver_age'].between(16, 100)
        if valid.sum() > 0:
            avg_age = df_view.loc[valid, 'driver_age'].mean()
            st.metric("Average Driver Age", f"{avg_age:.1f}")
            young = df_view['driver_age'].between(18, 29).sum()
            young_pct = young / len(df_view) * 100 if len(df_view) > 0 else 0
            st.caption(f"Young drivers (18–29): {young_pct:.1f}%  |  {young:,} policies")
        else:
            st.metric("Average Driver Age", "—", delta="No valid ages")
    else:
        st.metric("Average Driver Age", "—", delta="Column missing")

with col2:
    if 'no_claim_discount' in df_view.columns:
        has_ncd = (df_view['no_claim_discount'] > 0).sum()
        pct_ncd = has_ncd / len(df_view) * 100 if len(df_view) > 0 else 0
        st.metric("% with NCD", f"{pct_ncd:.1f}%")
        st.caption(f"{has_ncd:,} policies")
    else:
        st.metric("% with NCD", "—")

with col3:
    if 'no_intermediary_discount' in df_view.columns:
        has_inter = (df_view['no_intermediary_discount'] > 0).sum()
        pct_inter = has_inter / len(df_view) * 100 if len(df_view) > 0 else 0
        st.metric("% with Intermediary Discount", f"{pct_inter:.1f}%")
        st.caption(f"{has_inter:,} policies")
    else:
        st.metric("% with Intermediary Discount", "—")

# ────────────────────────────────────────────────
#  Yearly Trend Line Chart – Select Any Numeric Variable
# ────────────────────────────────────────────────
st.markdown("### Yearly Trend – Average Written Premium & Policy Count")

if year_col_name in df_view.columns:
    trend_df = df_view.groupby(year_col_name).agg(
        Avg_Premium=('actual_written_prem', 'mean'),
        Median_Premium=('actual_written_prem', 'median'),
        Policy_Count=('actual_written_prem', 'count')
    ).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trend_df[year_col_name],
        y=trend_df['Avg_Premium'],
        name='Avg Premium',
        line=dict(color='#1f77b4', width=3),
        mode='lines+markers'
    ))

    fig.add_trace(go.Scatter(
        x=trend_df[year_col_name],
        y=trend_df['Median_Premium'],
        name='Median Premium',
        line=dict(color='#2ca02c', dash='dot', width=2),
        mode='lines+markers'
    ))

    fig.add_trace(go.Bar(
        x=trend_df[year_col_name],
        y=trend_df['Policy_Count'],
        name='Policy Count',
        yaxis='y2',
        opacity=0.3,
        marker_color='lightgray'
    ))

    fig.update_layout(
        title="Yearly Trend: Premium & Policy Volume",
        xaxis_title="Policy Year",
        yaxis_title="Premium Amount",
        yaxis2=dict(title="Number of Policies", overlaying='y', side='right'),
        hovermode='x unified',
        height=520,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        xaxis=dict(dtick=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw yearly numbers"):
        st.dataframe(trend_df.style.format({
            'Avg_Premium': '{:,.0f}',
            'Median_Premium': '{:,.0f}',
            'Policy_Count': '{:,}'
        }), use_container_width=True, hide_index=True)
else:
    st.info("Cannot show yearly trend – missing policy_year column.")

# ────────────────────────────────────────────────
#  Original per-column rendering (kept as-is)
# ────────────────────────────────────────────────
show_charts = "Charts" in view_mode
show_yearly = yearly_mode and year_col_name and df_view[year_col_name].notna().any()

for col in selected_items:

    if col in categorical_cols:
        if show_yearly:
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
        if show_yearly and year_col_name:
            summary = df_view.groupby(year_col_name)[col].describe().round(2)
            summary.index = summary.index.astype(str) + " (year)"
            summary.index.name = "Year"

            st.dataframe(
                summary.style
                    .format(precision=2, thousands=",", decimal=".")
                    .set_caption(f"Summary statistics of {col} by policy year")
                    .set_properties(**{'text-align': 'right'}),
                use_container_width=True
            )

            if len(summary) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=summary.index, y=summary['mean'], mode='lines+markers', name='Mean'))
                fig.add_trace(go.Scatter(x=summary.index, y=summary['50%'], mode='lines+markers', name='Median'))
                fig.update_layout(title=f"Mean & Median of {col} over years", height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            if show_charts:
                fig = px.histogram(df_view, x=col, title=f"Distribution of {col}", marginal="box", opacity=0.75)
                st.plotly_chart(fig, use_container_width=True)
            else:
                stats = df_view[col].describe().round(2).to_frame().T
                st.dataframe(stats.style.format(precision=2, thousands=","), use_container_width=True)

    elif col in date_cols:
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

# Footer
st.markdown("---")
st.caption(f"Dataset: {len(df):,} rows | Years available: {df[year_col_name].min()} – {df[year_col_name].max()}" if year_col_name else "")

# ────────────────────────────────────────────────
#  Flexible Group-by + Distribution Chart (with binning for premium)
# ────────────────────────────────────────────────
st.markdown("### Group by One Column & Compare Distribution of Another")

# Grouping columns
group_candidates = [
    'vehicle_type', 'vehicle_usage', 'policyholder_gender',
    'policyholder_occupation', 'vehicle_brand', 'vehicle_model',
    'transaction_type', 'policyholder_type'
]

# Comparison columns
compare_candidates = [
    'actual_written_prem', 'sum_insured', 'basic_prem',
    'policy_excess', 'no_claim_discount',
    'no_intermediary_discount', 'vehicle_age',
    'driver_age'
]

col_g, col_c = st.columns(2)

with col_g:
    selected_group = st.selectbox(
        "Group by:",
        [c for c in group_candidates if c in df_view.columns],
        index=0,
        key="group_select_flex"
    )

with col_c:
    selected_compare = st.selectbox(
        "Show distribution of:",
        [c for c in compare_candidates if c in df_view.columns],
        index=0,
        key="compare_select_flex"
    )

# ── Prepare data ──
if selected_group in df_view.columns and selected_compare in df_view.columns:
    df_plot = df_view.copy()

    # Top 15 for certain groups
    if selected_group in ['vehicle_model', 'vehicle_brand', 'policyholder_occupation']:
        top15 = df_plot[selected_group].value_counts().head(15).index
        df_plot = df_plot[df_plot[selected_group].isin(top15)].copy()
        st.caption(f"Showing top 15 {selected_group} by policy count")

    # ── Custom / auto bins ──
    bins = None
    labels = None
    compare_col = selected_compare
    title_suffix = ""

    if selected_compare == 'actual_written_prem':
        # Auto-bins: 10 equal-width based on min/max
        prem_min = df_plot[selected_compare].min()
        prem_max = df_plot[selected_compare].max()
        
        if pd.notna(prem_min) and pd.notna(prem_max) and prem_max > prem_min:
            # Create 10 bins from min to max
            bin_edges = np.linspace(prem_min, prem_max, 11)  # 11 edges → 10 bins
            bins = bin_edges
            # Nice labels (rounded to nearest integer)
            labels = [f'{int(bin_edges[i]):,}–{int(bin_edges[i+1]):,}' for i in range(10)]
            df_plot['prem_bin'] = pd.cut(
                df_plot[selected_compare],
                bins=bins,
                labels=labels,
                right=False,
                include_lowest=True
            )
            compare_col = 'prem_bin'
            title_suffix = f" (auto-binned: {len(labels)} groups)"
        else:
            st.caption("No valid range for binning (single value or all missing)")
            compare_col = selected_compare
            title_suffix = ""

    elif selected_compare == 'sum_insured':
        bins = [0, 1500000, 3000000, np.inf]
        labels = ['0–1.5M', '1.5M–3M', '3M+']
        compare_col = 'sum_bin'
        df_plot[compare_col] = pd.cut(df_plot[selected_compare], bins=bins, labels=labels, right=False, include_lowest=True)
        title_suffix = " (binned)"

    elif selected_compare == 'policy_excess':
        bins = [0, 2000, 4000, 6000, 8000, 10000, np.inf]
        labels = ['0–2k', '2k–4k', '4k–6k', '6k–8k', '8k–10k', '10k+']
        compare_col = 'excess_bin'
        df_plot[compare_col] = pd.cut(df_plot[selected_compare], bins=bins, labels=labels, right=False, include_lowest=True)
        title_suffix = " (binned)"

    elif selected_compare in ['no_claim_discount', 'no_intermediary_discount']:
        # No binning – use raw values directly
        compare_col = selected_compare
        title_suffix = ""

    elif selected_compare == 'driver_age':
        bins = [16, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 100]
        labels = ['16-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70+']
        compare_col = 'age_bin'
        df_plot[compare_col] = pd.cut(df_plot[selected_compare], bins=bins, labels=labels, right=False)
        title_suffix = ""

    # If no binning applied
    if bins is None:
        compare_col = selected_compare
        title_suffix = ""

    # ── Compute distribution ──
    dist_df = df_plot.groupby([selected_group, compare_col], observed=True).size().reset_index(name='count')
    total_per_group = dist_df.groupby(selected_group)['count'].transform('sum')
    dist_df['percentage'] = (dist_df['count'] / total_per_group * 100).round(1)

    if dist_df.empty:
        st.info("No valid data after filtering / binning")
    else:
        # Stacked percentage bar
        fig = px.bar(
            dist_df,
            x=selected_group,
            y='percentage',
            color=compare_col,
            title=f"Distribution of {selected_compare.replace('_', ' ').title()}{title_suffix} by {selected_group.replace('_', ' ').title()}",
            barmode='stack',
            height=600,
            text='percentage',
            category_orders={compare_col: labels if labels else None}
        )

        fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
        fig.update_layout(
            xaxis_title=selected_group.replace('_', ' ').title(),
            yaxis_title="% of Policies",
            legend_title=compare_col.replace('_', ' ').title(),
            hovermode='x unified',
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── Count table ──
    with st.expander("Detailed Count & Percentage Table"):
        if not dist_df.empty:
            col_left, col_right = st.columns(2)

            # Count table
            pivot_count = dist_df.pivot(
                index=selected_group,
                columns=compare_col,
                values='count'
            ).fillna(0).astype(int)

            if labels:
                pivot_count = pivot_count.reindex(columns=labels, fill_value=0)

            with col_left:
                st.markdown("**Counts**")
                st.dataframe(
                    pivot_count.style
                        .format("{:,.0f}")
                        .background_gradient(cmap='YlOrRd', axis=None),
                    use_container_width=True
                )

            # Percentage table
            pivot_pct = dist_df.pivot(
                index=selected_group,
                columns=compare_col,
                values='percentage'
            ).fillna(0).round(1)

            if labels:
                pivot_pct = pivot_pct.reindex(columns=labels, fill_value=0.0)

            with col_right:
                st.markdown("**Row %**")
                st.dataframe(
                    pivot_pct.style
                        .format("{:.1f}%")
                        .background_gradient(cmap='YlGn', axis=None),
                    use_container_width=True
                )
        else:
            st.info("No data available")

# ────────────────────────────────────────────────
#  Special Brand Groups Analysis (including Tesla)
# ────────────────────────────────────────────────

# Find the actual brand column
st.markdown("### Development Trend – Premium/Luxury, Chinese/New Energy & Tesla")

# Make sure we have the brand_group column from earlier code
# (If not already created → recreate it here)
brand_col = next((c for c in ['vehicle_brand', 'Vehicle_Brand', 'brand'] if c in df_view.columns), None)

if brand_col and 'policy_year' in df_view.columns:
    df_trend = df_view.copy()

    df_trend['brand_group'] = 'Other'
    df_trend.loc[df_trend[brand_col].isin(expensive_brand), 'brand_group'] = 'Premium/Luxury'
    df_trend.loc[df_trend[brand_col].isin(ch_brand), 'brand_group'] = 'Chinese/New Energy'
    # Tesla (robust matching)
    df_trend.loc[df_trend[brand_col].astype(str).str.contains('tesla', case=False, na=False), 'brand_group'] = 'Tesla'

    # ── Prepare yearly counts ──
    yearly_trend = (
    df_trend
        .groupby(['policy_year', 'brand_group'])
        .size()
        .reset_index(name='Number of Policies')
    )

    if yearly_trend.empty:
        st.info("No policies found in any brand group.")
    else:
        # Optional: sort years just in case
        yearly_trend = yearly_trend.sort_values('policy_year')

        # Line chart – policy count (now includes Other)
        fig_count = px.line(
            yearly_trend,
            x='policy_year',
            y='Number of Policies',
            color='brand_group',
            markers=True,
            line_shape='linear',
            title="Number of Policies per Year by Brand Group (including Others)",
            color_discrete_map={
                'Premium/Luxury': '#1f77b4',          # blue
                'Chinese/New Energy': '#ff7f0e',      # orange
                'Tesla': '#d62728',                   # red
                'Other': '#7f7f7f'                    # gray – add this
            }
        )

        fig_count.update_traces(
            marker=dict(size=10, line=dict(width=2)),
            line=dict(width=3)
        )

        fig_count.update_layout(
            xaxis_title="Policy Year",
            yaxis_title="Number of Policies",
            hovermode="x unified",
            legend_title="Brand Group",
            xaxis=dict(dtick=1, type='linear'),
            height=580,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_count, use_container_width=True)


        # Quick summary table
        with st.expander("Yearly Policy Counts Table (including Others) – with % breakdown"):
            # ── Include ALL brand groups (including Other) ──
            yearly_trend_all = (
                df_trend
                .groupby(['policy_year', 'brand_group'])
                .size()
                .reset_index(name='Number of Policies')
            )

            if yearly_trend_all.empty:
                st.info("No data available for yearly brand group breakdown.")
            else:
                # Pivot for counts
                pivot_count = yearly_trend_all.pivot(
                    index='policy_year',
                    columns='brand_group',
                    values='Number of Policies'
                ).fillna(0).astype(int)

                # Ensure desired column order
                desired_order = ['Premium/Luxury', 'Chinese/New Energy', 'Tesla', 'Other']
                available = [c for c in desired_order if c in pivot_count.columns]
                pivot_count = pivot_count[available]

                # Add total column
                pivot_count['Total'] = pivot_count.sum(axis=1)

                # Row percentages
                pivot_pct = pivot_count.drop(columns=['Total']).div(
                    pivot_count['Total'], axis=0
                ) * 100
                pivot_pct = pivot_pct.round(1)

                # ── Display side by side ──
                col_left, col_right = st.columns(2)

                with col_left:
                    st.markdown("**Policy Counts per Year**")
                    st.dataframe(
                        pivot_count.style
                            .format("{:,}")
                            .format("{:,}", subset=['Total'])
                            .set_properties(**{'text-align': 'center'}),
                        use_container_width=True,
                        hide_index=False
                    )

                with col_right:
                    st.markdown("**Percentage of Policies per Year** (row sums to 100%)")
                    st.dataframe(
                        pivot_pct.style
                            .format("{:.1f}%")
                            .set_properties(**{'text-align': 'center'}),
                        use_container_width=True,
                        hide_index=False
                    )

                # Summary line
                grand_total = pivot_count['Total'].sum()
                years_shown = len(pivot_count)
                st.caption(
                    f"Total policies in view: {grand_total:,} across {years_shown} year{'s' if years_shown != 1 else ''} | "
                    f"Showing groups: {', '.join(available)}"
                )

if st.button("Force Reload"):
    st.cache_data.clear()
    st.rerun()
