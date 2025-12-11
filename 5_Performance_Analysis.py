import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Performance Analysis")

base = os.getcwd()
file_path = os.path.join(base, "HR", "HR_Analytics_Dataset_1000.csv")
df = pd.read_csv(file_path)

st.subheader("📌 Performance Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Avg Performance Score", round(df["Performance_Score"].mean(), 2))
col2.metric("High Performers (Score ≥ 4)", df[df["Performance_Score"] >= 4].shape[0])
col3.metric("Promoted Employees", df[df["Promotion_Status"] == "Promoted"].shape[0])

st.markdown("---")


# Performance Score Distribution
st.subheader("📊 Performance Score Distribution")

fig1 = px.histogram(
    df,
    x="Performance_Score",
    nbins=5,
    title="Performance Score Distribution",
    color="Performance_Score"
)
st.plotly_chart(fig1, use_container_width=True)


# Performance by Department
st.subheader("🏢 Average Performance by Department")

perf_dept = df.groupby("Department")["Performance_Score"].mean().reset_index()

fig2 = px.bar(
    perf_dept,
    x="Department",
    y="Performance_Score",
    title="Avg Performance Score per Department",
    text="Performance_Score",
    color="Performance_Score"
)
st.plotly_chart(fig2, use_container_width=True)


# Salary vs Performance
st.subheader("💰 Salary vs Performance Score")

fig3 = px.scatter(
    df,
    x="Performance_Score",
    y="Salary",
    color="Department",
    title="Salary vs Performance Score",
    trendline="ols"
)
st.plotly_chart(fig3, use_container_width=True)


# Experience vs Performance
st.subheader("⌛ Experience vs Performance Score")

fig4 = px.scatter(
    df,
    x="Experience_Years",
    y="Performance_Score",
    color="Department",
    title="Experience vs Performance Score",
    hover_data=["Name"],
    trendline="ols"
)
st.plotly_chart(fig4, use_container_width=True)


# Promotion vs Performance
st.subheader("🎓 Promotion vs Performance")

promo_data = df.groupby("Promotion_Status")["Performance_Score"].mean().reset_index()

fig5 = px.bar(
    promo_data,
    x="Promotion_Status",
    y="Performance_Score",
    title="Performance Score by Promotion Status",
    text="Performance_Score",
    color="Performance_Score"
)
st.plotly_chart(fig5, use_container_width=True)
