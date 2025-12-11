import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("💰 Salary Insights")

base = os.getcwd()
file_path = os.path.join(base, "HR", "HR_Analytics_Dataset_1000.csv")
df = pd.read_csv(file_path)

st.subheader("📌 Salary Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Avg Salary", f"₹{int(df['Salary'].mean()):,}")
col2.metric("Min Salary", f"₹{int(df['Salary'].min()):,}")
col3.metric("Max Salary", f"₹{int(df['Salary'].max()):,}")

st.markdown("---")


# Salary by Department
st.subheader("🏢 Salary by Department")

dept_salary = df.groupby("Department")["Salary"].mean().reset_index()

fig1 = px.bar(
    dept_salary,
    x="Department",
    y="Salary",
    color="Salary",
    title="Average Salary by Department",
    text="Salary"
)

st.plotly_chart(fig1, use_container_width=True)


# Salary by Job Role
st.subheader("💼 Salary by Job Role")

role_salary = df.groupby("Job_Role")["Salary"].mean().reset_index()

fig2 = px.bar(
    role_salary,
    x="Job_Role",
    y="Salary",
    color="Salary",
    title="Average Salary by Job Role",
)

st.plotly_chart(fig2, use_container_width=True)


# Salary vs Experience
st.subheader("📈 Salary vs Experience")

fig3 = px.scatter(
    df,
    x="Experience_Years",
    y="Salary",
    color="Department",
    title="Relationship: Experience vs Salary",
    trendline="ols"
)

st.plotly_chart(fig3, use_container_width=True)


# Salary Distribution
st.subheader("📊 Salary Distribution (Histogram)")

fig4 = px.histogram(df, x="Salary", nbins=40, title="Salary Distribution")
st.plotly_chart(fig4, use_container_width=True)
