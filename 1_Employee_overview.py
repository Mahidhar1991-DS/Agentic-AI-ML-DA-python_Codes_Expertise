import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.write("Current working directory:", os.getcwd())
st.write("Files in current directory:", os.listdir())


st.title("👥 Employee Overview")
base = os.getcwd()
file_path = os.path.join(base, "HR", "HR_Analytics_Dataset_1000.csv")
df = pd.read_csv(file_path)


# KPI Metrics
st.subheader("📌 Key Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", df.shape[0])
col2.metric("Avg Age", int(df["Age"].mean()))
col3.metric("Avg Salary", f"₹{int(df['Salary'].mean()):,}")

st.markdown("---")

# Gender Distribution
st.subheader("🧑‍🤝‍🧑 Gender Distribution")
gender_count = df["Gender"].value_counts().reset_index()
fig = px.pie(gender_count, names="Gender", values="count", title="Gender Ratio")
st.plotly_chart(fig, use_container_width=True)

# Department Distribution
st.subheader("🏢 Department Count")
dept_count = df["Department"].value_counts().reset_index()
fig2 = px.bar(dept_count, x="count", y="Department", title="Department Distribution")
st.plotly_chart(fig2, use_container_width=True)
