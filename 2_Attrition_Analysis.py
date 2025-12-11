import os
import streamlit as st 
import pandas as pd 
import plotly.express as px

st.title("Attrition Analysis")

base = os.getcwd()
file_path = os.path.join(base, "HR", "HR_Analytics_Dataset_1000.csv")
df = pd.read_csv(file_path)

df['Attrition_Flag'] = df["Attrition"].apply(lambda x: 1 if x=="Yes" else 0)


st.subheader("Attrition Key Metrics")

total_emp = df.shape[0]
total_attrition = df["Attrition_Flag"].sum()
attrition_rate = round((total_attrition/total_emp)*100,2)


col1,col2,col3 = st.columns(3)
col1.metric("Total Employees", total_emp)
col2.metric("Employees Left", total_attrition)
col3.metric("Attrition Rate", f'{attrition_rate}%')

st.markdown("_____")


#Attrition by Department 

dept = df.groupby("Department")["Attrition_Flag"].mean().reset_index()
dept["Attrition %"] = dept["Attrition_Flag"]*100

fig1 = px.bar(dept, x="Department",y="Attrition %", color = "Attrition %", title = "Department Attrition %",text = "Attrition %")

st.plotly_chart(fig1, use_container_width = True)

#Attrition by Age Group

st.subheader("Attrition by Age Group")

age = df["Age_Group"] = pd.cut(df["Age"], bins=[18,25,35,45,60], labels=["18-25", "26-35", "36-45", "46-60"])

age = df.groupby("Age_Group")["Attrition_Flag"].mean().reset_index()
age["Attrition %"] = age["Attrition_Flag"]*100

fig2 = px.line(
    x=age["Age_Group"],
    y=age["Attrition %"],
    markers=True,
    title="Age Group vs Attrition %"
)

# fig2 = px.line(age, x="Age Group", y="Attrition %", markers = True, title = "Age Group vs Attrition %")
st.plotly_chart(fig2, us_container_width = True )

# Attrition by Education Level
st.subheader("🎓 Attrition by Education Level")

edu = df.groupby("Education")["Attrition_Flag"].mean().reset_index()
edu["Attrition %"] = edu["Attrition_Flag"] * 100

fig3 = px.bar(
    edu,
    x="Education",
    y="Attrition %",
    color="Attrition %",
    title="Education Level vs Attrition %",
    text="Attrition %"
)
st.plotly_chart(fig3, use_container_width=True)







