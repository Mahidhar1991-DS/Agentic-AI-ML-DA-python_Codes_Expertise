import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🏢 Department Insights")

base = os.getcwd()
file_path = os.path.join(base, "HR", "HR_Analytics_Dataset_1000.csv")
df = pd.read_csv(file_path)

# Dropdown to select department
department_list = df["Department"].unique()
selected_dept = st.selectbox("Select Department:", department_list)

df_dept = df[df["Department"] == selected_dept]

st.markdown(f"## 📌 Insights for: **{selected_dept}**")
st.markdown("---")

# KPIs
col1, col2, col3 = st.columns(3)

col1.metric("Employees", df_dept.shape[0])
col2.metric("Avg Age", int(df_dept["Age"].mean()))
col3.metric("Avg Salary", f"₹{int(df_dept['Salary'].mean()):,}")

# Job Role Count
st.subheader("💼 Job Roles in Department")

role_count = df_dept["Job_Role"].value_counts().reset_index()
fig1 = px.bar(role_count, x="count", y="Job_Role", title="Job Role Distribution", text="Job_Role")
st.plotly_chart(fig1, use_container_width=True)

# Experience Distribution
st.subheader("📈 Experience Distribution")

fig2 = px.histogram(df_dept, x="Experience_Years", nbins=20, title="Experience Levels")
st.plotly_chart(fig2, use_container_width=True)

# Salary Distribution
st.subheader("💰 Salary Spread")

fig3 = px.box(df_dept, y="Salary", points="all", title="Salary Range")
st.plotly_chart(fig3, use_container_width=True)

# Age vs Salary
st.subheader("👤 Age vs Salary")

fig4 = px.scatter(
    df_dept,
    x="Age",
    y="Salary",
    color="Job_Role",
    title="Age vs Salary (colored by Role)",
    hover_data=["Name"]
)
st.plotly_chart(fig4, use_container_width=True)
