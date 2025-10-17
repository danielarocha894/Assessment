# app_streamlit.py
# ======================================================
# Which Sources Drive Lasting Engagement? — UTM Attribution Report
# Streamlit + Altair • Graphs only • Same metrics and logic as the attached notebook
# ======================================================

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import duckdb
from collections import Counter

# ---------------- Page & minimal CSS ----------------
st.set_page_config(page_title="Working Student - Data - dltHub", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: .6rem; }
.header { display:flex; align-items:center; gap:18px; }
.header h1 { margin:0; font-size: 28px; }
.sub { color:#6b7280; margin-top:2px; }
.section { margin: 10px 0 8px; text-transform: uppercase; letter-spacing:.03em; font-size:12px; color:#374151; }
.expl { color:#4b5563; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
c1, c2 = st.columns([0.12, 0.88])
with c1:
    st.image("https://dlthub.com/docs/img/dlthub-logo.png", use_container_width=True)
with c2:
    st.markdown('<div class="header"><h1>Working Student Assessment- Data - dltHub</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Understand where users come from and which acquisition sources sustain engagement.</div>', unsafe_allow_html=True)
st.divider()

# ---------------- File upload (no sidebar) ----------------
file = st.file_uploader("Upload the anonymized CSV (contacts + events)", type=["csv"])
if not file:
    st.stop()

# ===================== 1) Load & clean (identical column handling) =====================
df = pd.read_csv(file)

df = df.rename(columns={
    "Contact ID": "contact_id",
    "Field ID": "field_id",
    "Field Value": "field_value",
    "Contact Create date": "contact_created_at",
    "Contact Update date": "contact_updated_at",
    "Fields Title": "field_title",
    "Fields Type": "field_type",
    "Fields Update date": "field_updated_at",
    "Events Category": "event_category",
    "Events Create date": "event_created_at",
    "Events Hash": "event_hash",
    "Events ID": "event_id"
})

for col in ["contact_created_at","contact_updated_at","field_updated_at","event_created_at"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# ===================== 2) Processing — exact join/compose for field_title_value =====================
result = df[[
    "contact_id",
    "field_value",
    "field_title",
    "event_category",
    "event_created_at",
    "field_updated_at",
    "contact_created_at",
    "contact_updated_at"
]].copy()

result["field_value"] = result["field_value"].apply(
    lambda x: "undefined" if pd.notnull(x) and len(str(x)) > 64 else x
)

result["field_title_value"] = result.apply(
    lambda row: f"{row['field_value']} - {row['field_title']}"
    if pd.notnull(row["field_value"]) and len(str(row["field_value"])) < 64
    else row["field_title"],
    axis=1
)

# ===================== 3) Grouping — same anchor-word heuristic =====================
def find_common_prefixes(series, min_occurrences=2):
    all_values = series.dropna().astype(str)
    words = []
    for v in all_values:
        words.extend(v.split())
    counter = Counter(words)
    anchors = []
    for word, count in counter.most_common(50):
        if count >= min_occurrences and len(word) > 3:
            values_with_word = [v for v in all_values.unique() if word in v]
            if len(values_with_word) >= min_occurrences:
                anchors.append(word)
                return anchors[:10]
    return []

def group_by_prefix(value, anchor_words):
    s = str(value)
    if "utm" in s.lower():
        return s
    for a in anchor_words:
        if a in s:
            return f"category_{a}"
    parts = s.split()
    return f"category_{parts[0]}" if parts else "other_values"

anchors = find_common_prefixes(result["field_title_value"])
result["field_title_grouped"] = result["field_title_value"].apply(lambda x: group_by_prefix(x, anchors))

# ===================== 4) Final datasets — contacts_df / events_df =====================
utm_data = (
    result
    .sort_values(by=["contact_id","contact_created_at","field_updated_at"])
    .assign(acq_rank=lambda d: d.groupby("contact_id")["contact_created_at"]
            .rank(method="first", ascending=True).astype(int))
)

contacts_df = utm_data[[
    "contact_id",
    "contact_created_at",
    "field_title_grouped",
    "acq_rank"
]].rename(columns={"field_title_grouped": "utm_source"}).reset_index(drop=True)

events_df = (
    result.loc[result["event_created_at"].notna(),
               ["contact_id","event_category","event_created_at","contact_created_at"]]
    .copy()
)
events_df = events_df[events_df["event_created_at"] >= events_df["contact_created_at"]]
events_df = events_df.sort_values(["contact_id","event_created_at"]).reset_index(drop=True)

# ===================== 5) DuckDB — same schema and SQL as notebook =====================
con = duckdb.connect(database=":memory:")
con.execute("CREATE SCHEMA silver;")
con.register("contacts_df", contacts_df)
con.register("events_df", events_df)
con.execute("CREATE TABLE silver.contacts AS SELECT * FROM contacts_df;")
con.execute("CREATE TABLE silver.events AS SELECT * FROM events_df;")

general_acquisition = con.execute("""
    SELECT
        utm_source as source_utm,
        COUNT(DISTINCT contact_id) as total_contacts,
        MIN(contact_created_at) as first_acquisition,
        MAX(contact_created_at) as last_acquisition
    FROM silver.contacts
    GROUP BY utm_source
    ORDER BY total_contacts DESC
""").fetchdf()

utm_only_acquisition = con.execute("""
    SELECT
        utm_source as source_utm,
        COUNT(DISTINCT contact_id) as total_contacts,
        MIN(contact_created_at) as first_acquisition,
        MAX(contact_created_at) as last_acquisition
    FROM silver.contacts
    WHERE LOWER(utm_source) LIKE '%utm%'
    GROUP BY utm_source
    ORDER BY total_contacts DESC
""").fetchdf()

non_utm_acquisition = con.execute("""
    SELECT
        utm_source as source_utm,
        COUNT(DISTINCT contact_id) as total_contacts,
        MIN(contact_created_at) as first_acquisition,
        MAX(contact_created_at) as last_acquisition
    FROM silver.contacts
    WHERE LOWER(utm_source) NOT LIKE '%utm%'
    GROUP BY utm_source
    ORDER BY total_contacts DESC
""").fetchdf()

engagement_metrics_utm = con.execute("""
    WITH contact_events AS (
        SELECT
            c.contact_id as contact_id,
            c.utm_source as source_utm,
            c.contact_created_at as contact_creation_date,
            COUNT(e.event_created_at) as total_events,
            COUNT(DISTINCT e.event_category) as unique_event_types,
            MIN(e.event_created_at) as first_event_date,
            MAX(e.event_created_at) as last_event_date
        FROM silver.contacts c
        LEFT JOIN silver.events e ON c.contact_id = e.contact_id
        WHERE LOWER(c.utm_source) LIKE '%utm%'
        GROUP BY c.contact_id, c.utm_source, c.contact_created_at
    )
    SELECT
        source_utm,
        COUNT(contact_id) as acquired_contacts,
        SUM(CASE WHEN total_events > 0 THEN 1 ELSE 0 END) as contacts_with_events,
        ROUND(SUM(CASE WHEN total_events > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(contact_id), 2) as engagement_rate_percent,
        AVG(total_events) as avg_events_per_contact
    FROM contact_events
    GROUP BY source_utm
    ORDER BY engagement_rate_percent DESC
""").fetchdf()

# ===================== 6) Chart helpers (with data labels everywhere) =====================
def barh_with_labels(df, x, y, title, fmt=None, height=360):
    base = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{x}:Q", title=None),
        y=alt.Y(f"{y}:N", sort="-x", title=None),
        tooltip=[alt.Tooltip(f"{y}:N", title="Source"),
                 alt.Tooltip(f"{x}:Q", title="Value", format=fmt if fmt else ",")]
    ).properties(title=title, width="container", height=height)

    labels = alt.Chart(df).mark_text(
        align="left", baseline="middle", dx=3
    ).encode(
        x=alt.X(f"{x}:Q"),
        y=alt.Y(f"{y}:N", sort="-x"),
        text=alt.Text(f"{x}:Q", format=fmt if fmt else ",")
    )
    return base + labels

def scatter_with_labels(df, x, y, title="", height=360):
    if df.empty:
        return alt.Chart(pd.DataFrame({"x":[0],"y":[0]})).mark_circle().encode(x="x",y="y")
    points = alt.Chart(df).mark_circle(size=120).encode(
        x=alt.X(f"{x}:Q", title=None),
        y=alt.Y(f"{y}:Q", title=None),
        tooltip=[
            alt.Tooltip("source_utm:N", title="Source"),
            alt.Tooltip(f"{x}:Q", title=x),
            alt.Tooltip(f"{y}:Q", title=y)
        ]
    ).properties(title=title, width="container", height=height)

    # Data label: show y-value and source name
    labels = alt.Chart(df).mark_text(dx=8, dy=-8).encode(
        x=alt.X(f"{x}:Q"),
        y=alt.Y(f"{y}:Q"),
        text=alt.Text("source_utm:N")
    )
    return points + labels

def line_with_point_labels(df, x, y, title="", fmt=None, height=360):
    if df.empty:
        return alt.Chart(pd.DataFrame({"x":[0],"y":[0]})).mark_line().encode(x="x",y="y")
    ln = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X(f"{x}:Q", title=None),
        y=alt.Y(f"{y}:Q", title=None),
        tooltip=[alt.Tooltip(f"{x}:Q", title=x),
                 alt.Tooltip(f"{y}:Q", title=y, format=fmt if fmt else ",")]
    ).properties(title=title, width="container", height=height)

    labels = alt.Chart(df).mark_text(dy=-8).encode(
        x=alt.X(f"{x}:Q"),
        y=alt.Y(f"{y}:Q"),
        text=alt.Text(f"{y}:Q", format=fmt if fmt else ",")
    )
    return ln + labels

def acquisition_dashboard(df, title):
    if df.empty:
        st.warning("No data for this view.")
        return
    d = df.sort_values("total_contacts", ascending=False).copy()
    d["percentage"] = 100 * d["total_contacts"] / d["total_contacts"].sum()
    d["operation_days"] = (d["last_acquisition"] - d["first_acquisition"]).dt.days

    d_cum = d.sort_values("total_contacts", ascending=False).copy()
    d_cum["cumulative"] = d_cum["total_contacts"].cumsum()
    d_cum["cumulative_percentage"] = 100 * d_cum["cumulative"] / d_cum["total_contacts"].sum()
    d_cum["rank"] = np.arange(1, len(d_cum) + 1)

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(barh_with_labels(d.tail(15), "total_contacts", "source_utm", f"{title} · Acquisition Volume"), use_container_width=True)
    with col2:
        st.altair_chart(barh_with_labels(d.tail(10), "percentage", "source_utm", f"{title} · Percentage Distribution (Top 10)", fmt=".1f"), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        d_temporal = d[d["operation_days"] > 0].head(10).copy()
        st.altair_chart(scatter_with_labels(d_temporal, "total_contacts", "operation_days", f"{title} · Duration vs Volume"), use_container_width=True)
    with col4:
        st.altair_chart(line_with_point_labels(d_cum.head(10), "rank", "cumulative_percentage", f"{title} · Cumulative Distribution (%)", fmt=".1f"), use_container_width=True)

def engagement_panels_utm(df):
    if df.empty:
        st.warning("No UTM engagement data.")
        return
    d_rate = df.sort_values("engagement_rate_percent", ascending=False).head(10).copy()
    d_avg = df.sort_values("avg_events_per_contact", ascending=False).head(10).copy()

    col1, col2 = st.columns(2)
    with col1:
        g1 = alt.Chart(d_rate).mark_bar().encode(
            x=alt.X("source_utm:N", sort="-y", title=None),
            y=alt.Y("engagement_rate_percent:Q", title=None),
            tooltip=[alt.Tooltip("source_utm:N", title="Source"),
                     alt.Tooltip("engagement_rate_percent:Q", title="Engagement %", format=".1f"),
                     alt.Tooltip("acquired_contacts:Q", title="Contacts", format=",")]
        ).properties(title="Engagement Rate by UTM Source (Top 10)", width="container", height=360)
        g1_lbl = alt.Chart(d_rate).mark_text(dy=-5).encode(
            x=alt.X("source_utm:N", sort="-y"),
            y=alt.Y("engagement_rate_percent:Q"),
            text=alt.Text("engagement_rate_percent:Q", format=".1f")
        )
        st.altair_chart(g1 + g1_lbl, use_container_width=True)

    with col2:
        g2 = alt.Chart(d_avg).mark_bar().encode(
            x=alt.X("source_utm:N", sort="-y", title=None),
            y=alt.Y("avg_events_per_contact:Q", title=None),
            tooltip=[alt.Tooltip("source_utm:N", title="Source"),
                     alt.Tooltip("avg_events_per_contact:Q", title="Avg events", format=".2f"),
                     alt.Tooltip("acquired_contacts:Q", title="Contacts", format=",")]
        ).properties(title="Average Events per Contact by UTM Source (Top 10)", width="container", height=360)
        g2_lbl = alt.Chart(d_avg).mark_text(dy=-5).encode(
            x=alt.X("source_utm:N", sort="-y"),
            y=alt.Y("avg_events_per_contact:Q"),
            text=alt.Text("avg_events_per_contact:Q", format=".2f")
        )
        st.altair_chart(g2 + g2_lbl, use_container_width=True)

def concentration_panel(df, title):
    if df.empty:
        return
    top8 = df.head(8).copy()
    total_contacts = int(df["total_contacts"].sum()) if not df.empty else 0
    top5_pct = (df.nlargest(5, "total_contacts")["total_contacts"].sum() / total_contacts * 100) if total_contacts else 0
    subtitle = f"{title} — Total Sources: {len(df)} | Top 5: {top5_pct:.1f}%"

    g = alt.Chart(top8).mark_bar().encode(
        x=alt.X("source_utm:N", sort="-y", title=None),
        y=alt.Y("total_contacts:Q", title=None),
        tooltip=[alt.Tooltip("source_utm:N", title="Source"),
                 alt.Tooltip("total_contacts:Q", title="Contacts", format=",")]
    ).properties(title=subtitle, width="container", height=360)
    g_lbl = alt.Chart(top8).mark_text(dy=-5).encode(
        x=alt.X("source_utm:N", sort="-y"),
        y=alt.Y("total_contacts:Q"),
        text=alt.Text("total_contacts:Q", format=",")
    )
    st.altair_chart(g + g_lbl, use_container_width=True)

# ===================== 7) Tabs — same views, with concise corporate explanations =====================
tabs = st.tabs([
    "Acquisition — All Sources",
    "Acquisition — UTM Only",
    "Acquisition — Non-UTM",
    "Engagement — UTM",
    "Concentration"
])

with tabs[0]:
    acquisition_dashboard(general_acquisition, "All Sources")
    st.markdown(
        "<p class='expl'><b>Why this view:</b> This panel quantifies the overall acquisition landscape by source. "
        "Volume, percentage split, operating span, and cumulative contribution help identify meaningful channels versus long-tail noise.</p>",
        unsafe_allow_html=True
    )

with tabs[1]:
    acquisition_dashboard(utm_only_acquisition, "UTM Only")
    st.markdown(
        "<p class='expl'><b>Why this view:</b> UTM-tagged sources are controllable and attributable. "
        "Comparing their volume, longevity, and cumulative weight allows prioritizing campaigns that consistently deliver qualified signups.</p>",
        unsafe_allow_html=True
    )

with tabs[2]:
    acquisition_dashboard(non_utm_acquisition, "Non-UTM")
    st.markdown(
        "<p class='expl'><b>Why this view:</b> Non-UTM sources often hide organic or referral traffic. "
        "This view surfaces organic clusters worth formalizing with UTMs or pruning if they bring low-quality volume.</p>",
        unsafe_allow_html=True
    )

with tabs[3]:
    engagement_panels_utm(engagement_metrics_utm)
    st.markdown(
        "<p class='expl'><b>Why this view:</b> Engagement rate and average events per contact reveal which UTM sources "
        "convert signups into active users. High rate + high activity = durable value; low rate suggests volume without depth.</p>",
        unsafe_allow_html=True
    )

with tabs[4]:
    colA, colB, colC = st.columns(3)
    with colA: concentration_panel(general_acquisition, "All Sources — Top 8")
    with colB: concentration_panel(utm_only_acquisition, "UTM — Top 8")
    with colC: concentration_panel(non_utm_acquisition, "Non-UTM — Top 8")
    st.markdown(
        "<p class='expl'><b>Why this view:</b> Concentration highlights dependency risk and focus. "
        "If the top few sources dominate, we either invest further (if they are healthy) or diversify to reduce exposure.</p>",
        unsafe_allow_html=True
    )
