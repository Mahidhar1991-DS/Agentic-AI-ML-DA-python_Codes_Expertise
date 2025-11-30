import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

#1st part

st.set_page_config(page_title="Iris Dashboard", layout="wide")

def load_data():
    return sns.load_dataset('iris')

iris = load_data()

st.title("Iris Dataset Dashboard")
st.write("Interactive Visualizations using Streamlit + Seaborn + Matplotlib")

#second part

st.sidebar.header("🔎 Filters")

species = st.sidebar.multiselect(
    "Select Species",
    iris['species'].unique(),
    iris['species'].unique()
)

filtered = iris[iris['species'].isin(species)]

#Data table

st.subheader("📄 Data Table")
st.dataframe(filtered)

#third part

st.subheader("📊 Dataset Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Total Rows", len(filtered))
col2.metric("Species Selected", len(species))
col3.metric("Mean Petal Length", round(filtered['petal_length'].mean(), 2))


#visualisation

st.subheader("Sepal Length vs Width")

fig1, ax1 = plt.subplots()
sns.scatterplot(data=filtered, x="sepal_length", y="sepal_width", hue="species", ax=ax1)
st.pyplot(fig1)


st.subheader("Petal Length by Species")

fig2, ax2 = plt.subplots()
sns.boxplot(data=filtered, x="species", y="petal_length", ax=ax2)
st.pyplot(fig2)

st.subheader("Heatmap")

corr = filtered.select_dtypes(include='number').corr()

fig3, ax3 = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)





